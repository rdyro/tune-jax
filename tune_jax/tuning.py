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
import random as pyrandom

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


@dataclasses.dataclass
class _Config:
  allow_fallback_timing: bool = True


CONFIG = _Config()

logger = logging.getLogger("tune_jax")
if not logger.handlers:
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
  logger.addHandler(handler)
logger.setLevel(logging.WARNING)

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
    _ = jax.tree.map(lambda x: getattr(x, "block_until_ready", lambda: x)(), fn())
    return CompileResult(True, None)
  except Exception as _:
    msg = traceback.format_exc()
    return CompileResult(False, msg)


def _time_fn(fn: Callable[[], None], repeat: int = 5, number: int = 3) -> tuple[float, float]:
  """Time a function in a global single-threaded lock, so system is unloaded."""
  # assert repeat >= 2, f"{repeat = } must be >= 2, we discard slowest result."
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

    sorted_times = np.sort(times)[:-1] if repeat > 1 else np.sort(times)  # maybe drop the slowest time
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
  _timing_closure: Callable[[], None], platform: str, total_calls_number: int
) -> dict[int, tuple[float, float]]:
  function_timings = {}
  for it in tqdm(range(total_calls_number), desc=f"Profiling {platform}"):
    with tempfile.TemporaryDirectory(prefix="tuning_profile_", delete=False) as tempdir:
      profile_path = Path(tempdir).absolute()
      if it == 0:
        logger.info("Saving optimization profile to `%s`", profile_path)
      profile_path.mkdir(exist_ok=True)
      with jax.profiler.trace(str(profile_path)):
        _timing_closure(1)
      profile_files = sorted(profile_path.glob("**/*.xplane.pb"), key=lambda f: f.stat().st_mtime)
      if len(profile_files) == 0:
        raise RuntimeError("No profile was created.")
      latest_profile = profile_files[-1]
      profile_proto = profile_reader.parse_profile_from_bytes(latest_profile.read_bytes())
      device_plane_id = profile_reader.find_device_plane_ids(profile_proto, platform)[0]
      profile_events = profile_reader.get_events_from_plane(profile_proto, device_plane_id, verbose=False)
      fn_format = f"jit_{TUNE_FN_PREFIX_FMT.format('([0-9]+)')}.*"
      for k, durations in profile_events.items():
        if not re.match(fn_format, k):
          continue
        key = int(re.match(fn_format, k).group(1))
        assert len(durations) == 1, "We are expecting a single call per profile"
        function_timings.setdefault(key, []).extend(durations)

  for key, durations in function_timings.items():
    if len(durations) > 2:
      durations = sorted(durations)[1:-1]  # discard slowest and fastest
    function_timings[key] = (float(np.mean(durations)), float(np.std(durations)))

  return function_timings


@functools.partial(jax.jit, static_argnames=("sds", "sharding"))
def _get_random_value(sds, sharding):
  """Random values based on the tracer shape and dtype, and the sharding."""

  if hasattr(sds, "shape") and hasattr(sds, "dtype"):
    if jnp.issubdtype(sds.dtype, jnp.floating):
      return jax.jit(lambda key: random.normal(key, sds.shape, sds.dtype), out_shardings=sharding)(random.key(0))
    elif jnp.issubdtype(sds.dtype, jnp.integer):
      return jax.jit(lambda: jnp.zeros(sds.shape, sds.dtype), out_shardings=sharding)()
    else:
      raise ValueError(f"Unsupported dtype {sds.dtype}")
  else:
    return sds


def _try_hash_input(args, kws):
  """For eager mode tunable, hash the shape, dtype and sharding of the inputs."""

  flat_vals, struct = jax.tree.flatten((args, kws))
  all_concrete = all(jax.core.is_concrete(x) for x in flat_vals if isinstance(x, jax.Array))
  if not all_concrete:
    return None
  array_to_hashable = lambda x: x if not isinstance(x, jax.Array) else hash((jax.typeof(x), x.sharding))
  try:
    return hash((struct, tuple(array_to_hashable(x) for x in flat_vals)))
  except:
    return None


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

  def _get_best_hyperparams(args, kws):
    """Main tuning method."""

    # resolve sharding and/or device placement #################################
    if len(args) == 0 or all(x is None or jax.core.is_concrete(x) for x in jax.tree.leaves(args)):
      logger.info("All arguments are concrete, no need to pick random values.")
      args_val = args
    elif example_args is not None:
      logger.info("Example arguments provided")
      args_val = example_args
      if in_shardings is not UNSPECIFIED or device is not UNSPECIFIED:
        raise ValueError(
          "`example_args` cannot be used with in_shardings or"
          " device. `example_args` should already be correctly"
          " sharded."
        )
    else:
      logger.info("Selecting random input arguments.")
      resolved_device = device if isinstance(device, jax.Device) else _get_default_device()
      shardings = in_shardings if in_shardings is not UNSPECIFIED else jax.tree.map(lambda _: None, args)
      shardings = (shardings,) if len(args) == 1 else shardings
      shardings = jax.tree.map(
        functools.partial(_normalize_sharding, default_device=resolved_device),
        tuple(args),
        tuple(shardings),
      )
      _maybe_aval = lambda x: x if not isinstance(x, jax.Array) else x.aval
      args_val = jax.tree.map(lambda x, s: _get_random_value(_maybe_aval(x), s), args, shardings)

    if len(kws) == 0 or all(v is None or jax.core.is_concrete(v) for v in kws.values()):
      logger.info("All keyword arguments are concrete, no need to pick random values.")
      kws_val = kws
    elif example_kws is not None:
      logger.info("Example keyword arguments provided")
      kws_val = example_kws
    else:
      logger.info("Selecting random keyword arguments.")
      kws_val = jax.tree.map(lambda x: _get_random_value(_maybe_aval(x)), kws)

    hyperparams_norm = {k: (v if isinstance(v, (tuple, list)) else (v,)) for k, v in hyperparams.items()}
    executor = ThreadPoolExecutor(max_workers=max_workers)

    fns = dict()
    hyperparam_settings = dict(enumerate(itertools.product(*hyperparams_norm.values())))

    with _global_tuning_lock:
      # filter hyperparameters for those that compile ##########################
      for _ in range(2):  # sometimes a kernel compiles once, but not twice
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
    try:
      repeats = 5

      def _timing_closure(repeats: int):
        hs = list(hyperparam_settings.items())
        pyrandom.shuffle(hs)
        for i, hs in hs:
          hs = dict(zip(hyperparams_norm.keys(), hs))
          _time_fn(lambda: fns[i](*args_val, **kws_val), repeat=repeats, number=1)

      if len(jax.tree.leaves(args_val)) > 0:
        platform = list(jax.tree.leaves(args_val)[0].devices())[0].platform
      else:
        platform = _get_default_device().platform

      profiler_timings = _experimental_time_with_profiler(_timing_closure, platform, total_calls_number=repeats)
      for i, hs in hyperparam_settings.items():
        hs = dict(zip(hyperparams_norm.keys(), hs))
        results[i] = TimingResult(hs, *profiler_timings[i])
    except Exception as _:
      if not CONFIG.allow_fallback_timing:
        raise RuntimeError(f"Need to fall back to the python-level timing, but {CONFIG=} prohibits it.")
      # old timing fallback
      logger.warning(traceback.format_exc())
      warn("Could not time with the profiler, falling back to Python-level timing")
      _opts = dict(total=len(hyperparam_settings), disable=logger.level > logging.INFO, desc="Timing...")
      hs_pbar = tqdm(hyperparam_settings.items(), **_opts)
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
    maybe_hash = _try_hash_input(args, kws)
    if maybe_hash is not None and maybe_hash in wrapped_fn.hyperparams_cache:
      optimal_hyperparameters, results = wrapped_fn.hyperparams_cache[maybe_hash]
    else:
      with jax.core.eval_context():
        _, optimal_hyperparameters, results = _get_best_hyperparams(args, kws)
      if maybe_hash is not None:
        wrapped_fn.hyperparams_cache[maybe_hash] = (optimal_hyperparameters, results)
    if store_timing_results:
      wrapped_fn.timing_results.clear()
      wrapped_fn.timing_results.update(results)
    return fn_to_tune(*args, **dict(kws, **optimal_hyperparameters))

  wrapped_fn.timing_results = {}
  wrapped_fn.hyperparams_cache = {}
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
