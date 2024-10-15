from __future__ import annotations

import traceback
import functools
import itertools
import dataclasses
import threading
import time
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
import logging
from pprint import pformat

import jax
from jax import numpy as jnp
from jax import random
from jax.interpreters import pxla
from jax.sharding import (PartitionSpec, NamedSharding, Sharding, 
                          SingleDeviceSharding)
from jax.experimental.pallas.ops.gpu import attention
import numpy as np
from tqdm import tqdm

__all__ = ["tune"]

logger = logging.getLogger("tune_jax")
if not logger.handlers:
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
  logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

_global_tuning_lock = threading.Lock()

@dataclasses.dataclass
class CompileResult:
  status: bool
  error_msg: str | None = None

@dataclasses.dataclass
class TimingResult:
  hyperparams: dict[Any, Any]
  t_mean: float
  t_std: float

class _UnspecifiedT:
  pass

UNSPECIFIED = _UnspecifiedT()

def _get_global_mesh():
  env = pxla.thread_resources.env
  mesh = env.physical_mesh
  return None if mesh.empty else mesh

def _get_default_device():
  if jax.default_device.value is not None:
    return jax.default_device.value
  return jax.devices()[0]

def _try_call(fn: Callable[[], None]) -> CompileResult:
  """Attempt to call the function and return whether it compiles and runs."""
  try:
    _ = fn()
    _ = jax.tree.map(
      lambda x: getattr(x, "block_until_ready", lambda: x)(), fn()
    )
    return CompileResult(True, None)
  except Exception as _:
    msg = traceback.format_exc()
    return CompileResult(False, msg)


def _time_fn(fn: Callable[[], None], repeats: int = 10):
  """Time a function in a global single-threaded lock, so system is unloaded."""
  with _global_tuning_lock:
    times = [None for _ in range(repeats)]

    def _blocked_call():
      return jax.tree.map(
        lambda x: getattr(x, "block_until_ready", lambda: x)(), fn())

    start = time.time_ns()
    for i in range(repeats):
      _blocked_call()
      times[i] = time.time_ns()

    times = np.diff(np.array([start] + times) - start) / 1e9  # in seconds
    times = np.sort(times)[:-1]  # drop the slowest time
    t_mean, t_std = np.mean(times), np.std(times)
    return t_mean, t_std

def _timing_loss(result: TimingResult):
  """Compute a loss function for the timing result."""
  return result.t_mean + 0.1 * result.t_std

def _make_fn_to_time(fn_to_tune: Callable[..., Any], 
                     hyperparams: dict[str, Any], 
                     out_shardings: Sharding | _UnspecifiedT = UNSPECIFIED):
  """Embed hyperparameters into a function to time."""

  jit_opts = dict()
  if out_shardings is not UNSPECIFIED:
    jit_opts = dict(out_shardings=out_shardings)

  @functools.partial(jax.jit, **jit_opts)
  def _fn(*args, **kws):
    return fn_to_tune(*args, **dict(kws, **hyperparams))

  return _fn

def _normalize_sharding(sharding_or_spec: PartitionSpec | Sharding | None):
  if sharding_or_spec is None:
    return None
  elif (isinstance(sharding_or_spec, PartitionSpec) 
        and _get_global_mesh() is not None):
    return NamedSharding(mesh=_get_global_mesh(), spec=sharding_or_spec)
  else:
    return sharding_or_spec

def tune(
  fn_to_tune: Callable[..., Any],
  hyperparams: dict[Any, Any],
  max_workers: int = 32,
  in_shardings: Any = UNSPECIFIED,
  out_shardings: Any = UNSPECIFIED,
  device: jax.Device | _UnspecifiedT = UNSPECIFIED,
  example_args: tuple[Any] | None = None,
  example_kws: dict[Any, Any] | None = None,
):
  """Tune a function with hyperparameters, even if some fail to compile.

  Args:
      fn_to_tune (Callable[..., Any]): A jax function to tune.
      hyperparams (dict[Any, Any]): A flat dictionary of hyperparameter lists.
      max_workers (int, optional): Max workers for parallel compilation.
      in_shardings (Any, optional): in_shardings for timing (see jax.jit).
      out_shardings (Any, optional): out_shardings for timing (see jax.jit).
      device (jax.Device | _UnspecifiedT, optional): device to tune on if shardings are unspecified.
      example_args (tuple[Any] | None, optional): Exact example_args to tune with, on correct device.
      example_kws (dict[Any, Any] | None, optional): Exact example_kws to tune with, on correct device.
  """

  def _get_random_value(arr, sharding):
    if isinstance(arr, jax.Array):
      if jnp.issubdtype(arr.dtype, jnp.floating):
        return jax.device_put(random.normal(random.key(0), arr.shape, 
                                            dtype=arr.dtype), sharding)
      elif jnp.issubdtype(arr.dtype, jnp.integer):
        return jax.device_put(np.zeros(arr.shape, dtype=arr.dtype), sharding)
      else:
        raise ValueError(f"Unsupported dtype {arr.dtype}")
    else:
      return arr

  def _get_best_hyperparams(args, kws):
    if example_args is not None:
      args_val = example_args 
    else:
      if _get_global_mesh() is not None and in_shardings is not UNSPECIFIED:
        shardings = jax.tree.map(_normalize_sharding, in_shardings)
      elif device is not UNSPECIFIED:
        shardings = jax.tree.map(lambda _: SingleDeviceSharding(device), args)
      else:
        shardings = jax.tree.map(lambda _: SingleDeviceSharding(
          _get_default_device()), args)
      args_val = jax.tree.map(_get_random_value, args, shardings)

    if example_kws is not None:
      kws_val = example_kws 
    else: 
      kws_val = jax.tree.map(_get_random_value, kws)

    hyperparams_norm = {
      k: (v if isinstance(v, (tuple, list)) else (v,))
      for k, v in hyperparams.items()
    }
    executor = ThreadPoolExecutor(max_workers=max_workers)

    fns = dict()
    hyperparam_settings = {i: hs for i, hs in enumerate(
      itertools.product(*hyperparams_norm.values()))}

    with _global_tuning_lock:
      while True:
        compiles: dict[str, Future[CompileResult]] = dict()
        for i, kws in hyperparam_settings.items():
          hs = {k: v for k, v in zip(hyperparams_norm.keys(), kws)}
          fns[i] = _make_fn_to_time(fn_to_tune, hs, out_shardings=out_shardings)
          compiles[i] = executor.submit(
            lambda fn: _try_call(lambda: fn(*args_val, **kws_val)), fns[i])
        future_pbar = tqdm(compiles.items(), total=len(compiles), 
                           disable=logger.level > logging.DEBUG,
                           desc="Compiling...")
        successful_compiles = {k: x.result() for (k, x) in future_pbar
                               if x.result().status}
        if len(successful_compiles) == len(hyperparam_settings):
          break
        if len(successful_compiles) == 0:
          for i, compile_result in compiles.items():
            logger.error(f"Hyperparameters {hyperparam_settings[i]} failed to" 
                         f" compile with message:\n"
                         f" {compile_result.result().error_msg}")
          raise ValueError("No hyperparameters compiled successfully")
        logger.debug("Down to %d hyperparameters", len(successful_compiles))
        # cleanup
        hyperparam_settings = {i: hyperparam_settings[i] 
                               for i in successful_compiles.keys()}
        fns = {i: fns[i] for i in successful_compiles.keys()}

    # sequentially time the remaining hyperparameters
    results = dict()
    hs_pbar = tqdm(hyperparam_settings.items(), total=len(hyperparam_settings), 
                   disable=logger.level > logging.DEBUG, desc="Timing...")
    for i, hs in hs_pbar:
      hs = {k: v for k, v in zip(hyperparams_norm.keys(), hs)}
      results[i] = TimingResult(
        hs, *_time_fn(lambda: fns[i](*args_val, **kws_val), repeats=10))

    results = sorted(results.items(), key=lambda x: _timing_loss(x[1]))
    idx, optimal_hyperparams = results[0][0], results[0][1].hyperparams
    logger.debug(pformat(results))
    logger.debug(f"optimal hyperparams: {optimal_hyperparams}")
    return fns[idx], optimal_hyperparams

  @functools.wraps(fn_to_tune)
  def wrapped_fn(*args, **kws):
    with jax.ensure_compile_time_eval():
      _, optimal_hyperparameters = _get_best_hyperparams(args=args, kws=kws)
    return fn_to_tune(*args, **dict(kws, **optimal_hyperparameters))

  return wrapped_fn

# -------------------------------------------------------------------------------

def test_main():
  hyperparams = {
    "block_q": [4, 8, 16, 32, 64, 128],
    "block_k": [4, 8, 16, 32, 64, 128],
  }

  b, qt, h, d = 8, 32, 8, 512
  kt = 128

  q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

  tuned_mha = tune(attention.mha, hyperparams=hyperparams)
  tuned_mha_jit = jax.jit(tuned_mha)

  if False:
    @functools.partial(jax.jit, in_shardings=SingleDeviceSharding(jax.devices("cuda")[0]))
    @functools.partial(tune, hyperparams=hyperparams, device=jax.devices("cuda")[0])
    def forward_twice(q, k, v, **hyperparams):
      x1 = attention.mha(q, k, v, segment_ids=None, **hyperparams)
      x2 = attention.mha(q, k, v, segment_ids=None, **hyperparams)
      return x1 + x2

    with jax.default_device(jax.devices("cpu")[0]):
      forward_twice(q, k, v).block_until_ready()
      forward_twice(q, k, v).block_until_ready()

    return 

  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()

  return

if __name__ == "__main__":
  test_main()
