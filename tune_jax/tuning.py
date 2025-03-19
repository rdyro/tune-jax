from __future__ import annotations

import sys
import traceback
import functools
import itertools
import dataclasses
import threading
import tempfile
import time
import re
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
import logging
from pprint import pformat
from pathlib import Path
from warnings import warn
import contextlib

import jax
import jax.core
from jax import numpy as jnp
from jax import random
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, Sharding, SingleDeviceSharding
from jax.experimental.pallas.ops.gpu import attention
import numpy as np
from tqdm import tqdm

try:
  from . import profile_reader
except ImportError:
  if str(Path(__file__).parent.absolute()) not in sys.path:
    sys.path.append(str(Path(__file__).parent.absolute()))

  import profile_reader

__all__ = ["tune"]

TUNE_FN_PREFIX_FMT = "tune_jax_fn_{}"

logger = logging.getLogger("tune_jax")
if not logger.handlers:
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
  logger.addHandler(handler)
logger.setLevel(logging.WARNING)


_timer_indent = 0


@contextlib.contextmanager
def Timer(task_name: str):
  global _timer_indent
  _timer_indent += 2
  __t = time.time()
  try:
    yield
  finally:
    _timer_indent -= 2
    logger.debug(("  " * _timer_indent) + f"{task_name} took {time.time() - __t:.4e} s")


context_escape_pool_executor = ThreadPoolExecutor(max_workers=8)
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
    _ = jax.tree.map(
      lambda x: getattr(x, "block_until_ready", lambda: x)(),
      fn(),  # type: ignore
    )
    return CompileResult(True, None)
  except Exception as _:
    msg = traceback.format_exc()
    return CompileResult(False, msg)


def _time_fn(fn: Callable[[], None], repeat: int = 5, number: int = 3) -> tuple[float, float]:
  """Time a function in a global single-threaded lock, so system is unloaded."""
  assert repeat >= 2, f"{repeat = } must be >= 2, we discard slowest result."
  with _global_tuning_lock:

    def _blocked_call():
      return jax.tree.map(lambda x: getattr(x, "block_until_ready", lambda: x)(), fn())

    times_raw = []
    start = time.perf_counter()
    for r in range(repeat):
      for i in range(number):
        _blocked_call()
      times_raw.append(time.perf_counter())
    # in seconds
    times = np.diff(np.array([start] + times_raw) - start) / number

    sorted_times = np.sort(times)[:-1]  # drop the slowest time
    t_mean, t_std = np.mean(sorted_times), np.std(sorted_times)
    return float(t_mean), float(t_std)


def _timing_loss(result: TimingResult):
  """Compute a loss function for the timing result."""
  return result.t_mean + 0.1 * result.t_std


def _make_fn_to_time(
  fn_to_tune: Callable[..., Any],
  hyperparams: dict[str, Any],
  out_shardings: Sharding | _UnspecifiedT = UNSPECIFIED,
  name_id: int = 0,
) -> Callable[..., Any]:
  """Embed hyperparameters into a function to time."""

  jit_wrapper = functools.partial(jax.jit, out_shardings=out_shardings if out_shardings is not UNSPECIFIED else None)

  def _fn(*args, **kws):
    return fn_to_tune(*args, **dict(kws, **hyperparams))

  _fn.__name__ = TUNE_FN_PREFIX_FMT.format(name_id)

  return jit_wrapper(_fn)


def _normalize_sharding(
  arg: jax.Array | np.ndarray | Any,
  sharding_or_spec: PartitionSpec | Sharding | None,
  default_device: jax.Device,
):
  if not isinstance(arg, (jax.Array, np.ndarray)):
    return None
  if isinstance(sharding_or_spec, Sharding):
    return sharding_or_spec
  global_mesh = _get_global_mesh()
  if isinstance(sharding_or_spec, PartitionSpec) and global_mesh is not None:
    return jax.NamedSharding(global_mesh, sharding_or_spec)
  elif isinstance(sharding_or_spec, PartitionSpec) and global_mesh is None:
    raise ValueError("If specifying shardings via ParitionSpec, a global mesh must be defined")
  else:
    return SingleDeviceSharding(default_device)


def _experimental_time_with_profiler(
  _timing_closure: Callable[[], None], platform: str, total_calls_number: int = 0
) -> dict[int, tuple[float, float]]:
  with tempfile.TemporaryDirectory(delete=False) as tempdir:
    profile_path = Path(tempdir).absolute()
    logger.info("Saving optimization profile to `%s`", profile_path)
    profile_path.mkdir(exist_ok=True)

    with Timer("Running the code to profile and generating the profile"):
      with jax.profiler.trace(str(profile_path)):
        _timing_closure()

    with Timer("Finding latest profile"):
      profile_files = sorted(profile_path.glob("**/*.xplane.pb"), key=lambda f: f.stat().st_mtime)
      if len(profile_files) == 0:
        raise RuntimeError("No profile was created.")
      latest_profile = profile_files[-1]
    with Timer("Parsing proto"):
      profile_proto = profile_reader.parse_profile_from_bytes(latest_profile.read_bytes())
    with Timer("Finding the device plane"):
      device_plane_id = profile_reader.find_device_plane_ids(profile_proto, platform)[0]
    with Timer("Parsing the events"):
      profile_events = profile_reader.get_events_from_plane(profile_proto, device_plane_id, verbose=False)
    with Timer("Extracting timing statistics"):
      fn_format = f"jit\\({TUNE_FN_PREFIX_FMT.format('([0-9]+)')}\\)"
      function_timings = {}
      for k, v in profile_events.items():
        if not re.match(fn_format, k):
          continue
        key = int(re.match(fn_format, k).group(1))
        durations = [repeat.duration for repeat in v]
        if total_calls_number > 0:
          if total_calls_number != len(durations):
            raise RuntimeError(
              f"Obtained different number of scopes than expected. Expected {total_calls_number} vs {len(durations)}"
            )
        if len(durations) > 1:
          durations = sorted(durations)[: -max(round(0.2 * len(durations)), 1)]  # discard slowest / warmup
        function_timings[key] = (float(np.mean(durations)), float(np.std(durations)))
    return function_timings


def tune(
  fn_to_tune: Callable[..., Any],
  hyperparams: dict[Any, Any],
  max_workers: int = 32,
  in_shardings: Any = UNSPECIFIED,
  out_shardings: Any = UNSPECIFIED,
  device: jax.Device | _UnspecifiedT = UNSPECIFIED,
  example_args: tuple[Any] | None = None,
  example_kws: dict[Any, Any] | None = None,
  store_timing_results: bool = True,
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
      store_results (bool): Attach the timining results to the function handle (i.e., fn.timing_results)?
  """

  def _get_random_value(arr, sharding):
    """Random values based on the tracer shape and dtype, and the sharding."""

    # TODO(rdyro): these values get initialized on the default device before
    # being moved to the correct sharding, maybe use direct on-device generation
    if isinstance(arr, jax.Array):
      if jnp.issubdtype(arr.dtype, jnp.floating):
        return jax.device_put(random.normal(random.key(0), arr.shape, dtype=arr.dtype), sharding)
      elif jnp.issubdtype(arr.dtype, jnp.integer):
        return jax.device_put(np.zeros(arr.shape, dtype=arr.dtype), sharding)
      else:
        raise ValueError(f"Unsupported dtype {arr.dtype}")
    else:
      return arr

  def _get_best_hyperparams(args, kws):
    """Main tuning method."""

    # resolve sharding and/or device placement #################################
    if example_args is not None:
      args_val = example_args
      if in_shardings is not UNSPECIFIED or device is not UNSPECIFIED:
        raise ValueError(
          "`example_args` cannot be used with in_shardings or"
          " device. `example_args` should already be correctly"
          " sharded."
        )
    else:
      if isinstance(device, jax.Device):
        resolved_device = device
      else:
        resolved_device = _get_default_device()
      shardings = in_shardings if in_shardings is not UNSPECIFIED else jax.tree.map(lambda _: None, args)
      shardings = (shardings,) if len(args) == 1 else shardings
      shardings = jax.tree.map(
        functools.partial(_normalize_sharding, default_device=resolved_device),
        tuple(args),
        tuple(shardings),
      )
      args_val = jax.tree.map(_get_random_value, args, shardings)

    if example_kws is not None:
      kws_val = example_kws
    else:
      kws_val = jax.tree.map(_get_random_value, kws)

    hyperparams_norm = {k: (v if isinstance(v, (tuple, list)) else (v,)) for k, v in hyperparams.items()}
    executor = ThreadPoolExecutor(max_workers=max_workers)

    fns = dict()
    hyperparam_settings = dict(enumerate(itertools.product(*hyperparams_norm.values())))

    with _global_tuning_lock:
      # filter hyperparameters for those that compile ##########################
      # repeat until the number stabilizes
      # sometimes a kernel compiles once, but not twice
      while True:
        compiles: dict[int, Future[CompileResult]] = dict()
        for i, kws in hyperparam_settings.items():
          hs = dict(zip(hyperparams_norm.keys(), kws))
          fns[i] = _make_fn_to_time(fn_to_tune, hs, out_shardings=out_shardings, name_id=i)
          compiles[i] = executor.submit(lambda fn: _try_call(lambda: fn(*args_val, **kws_val)), fns[i])
        future_pbar = tqdm(
          compiles.items(),
          total=len(compiles),
          disable=logger.level > logging.INFO,
          desc="Compiling...",
        )
        successful_compiles = {k: x.result() for (k, x) in future_pbar if x.result().status}
        if len(successful_compiles) == len(hyperparam_settings):
          break
        if len(successful_compiles) == 0:
          for i, compile_result in compiles.items():
            logger.error(
              f"Hyperparameters {hyperparam_settings[i]} failed to"
              f" compile with message:\n"
              f" {compile_result.result().error_msg}"
            )
          raise ValueError("No hyperparameters compiled successfully")
        logger.debug("Down to %d hyperparameters", len(successful_compiles))
        # cleanup
        hyperparam_settings = {i: hyperparam_settings[i] for i in successful_compiles.keys()}
        fns = {i: fns[i] for i in successful_compiles.keys()}

    # sequentially time the remaining hyperparameters ##########################
    # the _time_fn will acquire the lock on its own
    results = dict()
    hs_pbar = tqdm(
      hyperparam_settings.items(),
      total=len(hyperparam_settings),
      disable=logger.level > logging.INFO,
      desc="Timing...",
    )

    try:
      repeats = 10

      def _timing_closure():
        for i, hs in hs_pbar:
          hs = dict(zip(hyperparams_norm.keys(), hs))
          _time_fn(lambda: fns[i](*args_val, **kws_val), repeat=repeats, number=1)

      if len(jax.tree.leaves(args_val)) > 0:
        platform = list(jax.tree.leaves(args_val)[0].devices())[0].platform
      else:
        platform = _get_default_device().platform

      with Timer("Profile all together"):
        profiler_timings = _experimental_time_with_profiler(_timing_closure, platform, total_calls_number=repeats)
      for i, hs in hs_pbar:
        hs = dict(zip(hyperparams_norm.keys(), hs))
        results[i] = TimingResult(hs, *profiler_timings[i])
    except Exception as _:
      # old timing fallback
      logger.warning(traceback.format_exc())
      warn("Could not time with the profiler, falling back to Python-level timing")
      for i, hs in hs_pbar:
        hs = dict(zip(hyperparams_norm.keys(), hs))
        results[i] = TimingResult(hs, *_time_fn(lambda: fns[i](*args_val, **kws_val), repeat=10))

    results = sorted(results.items(), key=lambda x: _timing_loss(x[1]))
    idx, optimal_hyperparams = results[0][0], results[0][1].hyperparams
    logger.info("\n" + pformat(results, width=300))
    logger.info(f"optimal hyperparams: {optimal_hyperparams}")
    return fns[idx], optimal_hyperparams, results

  @functools.wraps(fn_to_tune)
  def wrapped_fn(*args, **kws):
    with jax.core.eval_context():
      _, optimal_hyperparameters, results = _get_best_hyperparams(args, kws)
    if store_timing_results:
      wrapped_fn.timing_results.clear()
      wrapped_fn.timing_results.update(results)
    return fn_to_tune(*args, **dict(kws, **optimal_hyperparameters))

  wrapped_fn.timing_results = {}
  return wrapped_fn


# -------------------------------------------------------------------------------


def test_main():
  hyperparams = {
    # anything below 16 will fail since the smallest matmul block is 16x16
    "block_q": [4, 8, 16, 32, 64, 128],
    "block_k": [4, 8, 16, 32, 64, 128],
  }

  b, qt, h, d = 8, 32, 8, 512
  kt = 128

  q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

  attention_wrapper = lambda *args, block_q=None, block_k=None, **kw: attention.mha(
    *args,
    **dict(kw, block_sizes=attention.BlockSizes(block_q=block_q, block_k=block_k)),
  )

  tuned_mha = tune(attention_wrapper, hyperparams=hyperparams)
  tuned_mha_jit = jax.jit(tuned_mha)

  logger.setLevel("DEBUG")

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

  print(tuned_mha_jit.timing_results)  # to get access to timing results

  return


if __name__ == "__main__":
  test_main()
