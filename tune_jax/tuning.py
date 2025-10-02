from __future__ import annotations

import os
import traceback
import itertools
import contextlib
import dataclasses
from datetime import datetime
import threading
import tempfile
import time
import re
from functools import partial, wraps
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import logging
from pprint import pformat
from pathlib import Path
import random as pyrandom
from collections import defaultdict

import jax
import jax.core
from jax import numpy as jnp
from jax import random
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, Sharding, SingleDeviceSharding
import numpy as np
from tqdm import tqdm

from tune_jax import profile_reader

try:
  import tabulate as tabulate_mod
except ModuleNotFoundError:
  tabulate_mod = None

__all__ = ["tune"]

TUNE_FN_PREFIX_FMT = "tune_jax_fn_{}"


@dataclasses.dataclass
class _Config:
  # whether to error if we need to fall back to python timing
  allow_fallback_timing: bool = True
  # valid parsed event ratio threshold at which to abort profile parsing
  # for instance if only 3 out of 10 profiled events parsed correctly
  # and the fraction is set to 0.4, we abort and fall back to Python timing
  must_find_at_least_profiler_result_fraction: float = 0.5
  # how many profiling samples to take for profile parsing
  profiling_samples: int = 5
  # whether to attempt to compute optimal layouts for the function
  find_optimal_layouts_automatically: bool = False
  # whether to mark events that come back as 0.0 seconds as invalid
  _reject_zero_time_events: bool = True


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
  optimal_formats: Any = None


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
  if jax.config.values["jax_default_device"] is not None:
    return jax.config.values["jax_default_device"]
  return jax.devices()[0]


@contextlib.contextmanager
def suppress_stdout_stderr():
  devnull, stdout, stderr = open(os.devnull, "w+"), os.dup(1), os.dup(2)
  os.dup2(devnull.fileno(), 1), os.dup2(devnull.fileno(), 2)
  yield
  os.dup2(stdout, 1), os.dup2(stderr, 2)


def _try_call(
  fn: Callable[[], None],
  args_val,
  kws_val,
  compile_only: bool = False,
  compute_layouts: bool = False,
  optimal_formats: Any | None = None,
) -> CompileResult:
  """Attempt to call the function and return whether it compiles and runs."""
  try:
    if compile_only:
      if compute_layouts:
        to_shape = (
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if isinstance(x, jax.Array) else x
        )
        (args_shapes, kws_shapes) = jax.tree.map(to_shape, (args_val, kws_val))
        optimal_formats = jax.jit(fn).lower(*args_shapes, **kws_shapes).compile().input_formats
        print(f"Optimal formats: {pformat(optimal_formats)}")
      else:
        _ = jax.jit(fn).lower(*args_val, **kws_val).compile()
    else:
      if optimal_formats is not None:
        place_if_array = lambda x, f: jax.device_put(x, f) if isinstance(x, jax.Array) else x
        (optimal_args, optimal_kws) = jax.tree.map(place_if_array, (args_val, kws_val), optimal_formats)
        _ = jax.block_until_ready(fn(*optimal_args, **optimal_kws))
      else:
        _ = jax.block_until_ready(fn(*args_val, **kws_val))
    return CompileResult(True, None, optimal_formats)
  except Exception as _:
    msg = traceback.format_exc()
    return CompileResult(False, msg, optimal_formats)


def _time_fn(fn: Callable[[], None], repeat: int = 5, number: int = 3) -> tuple[float, float]:
  """Time a function in a global single-threaded lock, so system is unloaded."""
  with _global_tuning_lock:
    _blocked_call = lambda: jax.block_until_ready(fn())
    times_raw = []
    start = time.perf_counter()
    for _ in range(repeat):
      for _ in range(number):
        _blocked_call()
      times_raw.append(time.perf_counter())
    times = np.diff(np.array([start] + times_raw) - start) / number  # in seconds
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

  jit_wrapper = partial(jax.jit, out_shardings=out_shardings if out_shardings is not UNSPECIFIED else None)

  def _fn(*args, **kws):
    return fn_to_tune(*args, **dict(kws, **hyperparams))

  _fn.__name__ = TUNE_FN_PREFIX_FMT.format(name_id)
  _fn.__qualname__ = TUNE_FN_PREFIX_FMT.format(name_id)
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
  _timing_closure: Callable[[], None], platform: str, total_calls_number: int, event_filter_regex: str | None = None
) -> dict[int, tuple[float, float]]:
  function_timings = defaultdict(list)
  pbar = tqdm(range(total_calls_number), desc=f"Profiling {platform}", disable=logger.level > logging.INFO)
  for it in pbar:
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    profile_path = Path(tempfile.mkdtemp(prefix=f"tuning_profile_{now}_")).absolute()
    if it == 0:
      pbar.write(f"Saving optimization profile to `{profile_path}`")
    profile_path.mkdir(exist_ok=True)
    with suppress_stdout_stderr():
      with jax.profiler.trace(str(profile_path)):
        _timing_closure()
    profile_files = sorted(profile_path.glob("**/*.xplane.pb"), key=lambda f: f.stat().st_mtime)
    if len(profile_files) == 0:
      raise RuntimeError("No profile was created.")
    latest_profile = profile_files[-1]
    profile_proto = profile_reader.parse_profile_from_bytes(latest_profile.read_bytes())
    device_plane_id = profile_reader.find_device_plane_ids(profile_proto, platform)[0]
    profile_events = profile_reader.get_events_from_plane(
      profile_proto, device_plane_id, prefix_filter="jit_", event_filter_regex=event_filter_regex
    )
    fn_format = f"jit_{TUNE_FN_PREFIX_FMT.format('([0-9]+)')}.*"
    for k, duration in profile_events.items():
      if not re.match(fn_format, k):
        continue
      key = int(re.match(fn_format, k)[1])
      if (not CONFIG._reject_zero_time_events) or duration > 0:
        function_timings[key].append(duration)

  for key, durations in function_timings.items():
    if len(durations) > 2:
      durations = sorted(durations)[1:-1]  # discard slowest and fastest
    function_timings[key] = (float(np.mean(durations)), float(np.std(durations)))

  return function_timings


@partial(jax.jit, static_argnames=("sds", "sharding"))
def _get_random_value(sds, sharding=None):
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


def _try_hash_input(args, kws, must_be_concrete: bool = True):
  """For eager mode tunable, hash the shape, dtype and sharding of the inputs."""

  flat_vals, struct = jax.tree.flatten((args, kws))
  all_concrete = all(jax.core.is_concrete(x) for x in flat_vals if isinstance(x, jax.Array))
  if not all_concrete and must_be_concrete:
    return None

  def _get_sharding(x):
    try:
      return x.sharding
    except AttributeError:
      return jax.typeof(x).sharding

  array_to_hashable = lambda x: x if not isinstance(x, jax.Array) else hash((jax.typeof(x), _get_sharding(x)))
  try:
    return hash((struct, tuple(array_to_hashable(x) for x in flat_vals)))
  except:  # noqa: E722
    return None


def _tabulate_results(timing_results: dict[int, TimingResult] | Callable):
  if callable(timing_results):
    if not hasattr(timing_results, "timing_results"):
      raise ValueError("A function passed to `timing_results_to_df`, but it's missing a `timing_results` attribute.")
    timing_results: dict[int, TimingResult] = timing_results.timing_results
  if not isinstance(timing_results, dict):
    raise ValueError(f"Timing results passed to tabulate is not a dict, it's `{type(timing_results)}` instead.")
  if len(timing_results) == 0:
    return [], ["id", "t_mean (s)", "t_std (s)"]
  hyperparams_keys = list(timing_results.values())[0].hyperparams.keys()
  values = [
    [id] + [r.hyperparams[k] for k in hyperparams_keys] + [r.t_mean, r.t_std] for id, r in timing_results.items()
  ]
  columns = ["id"] + list(hyperparams_keys) + ["t_mean (s)", "t_std (s)"]
  return columns, values


def tabulate(timing_results: dict[int, TimingResult] | Callable) -> str:
  """Tabulate the timining results sorted fastest first with the `tabulate` package."""
  columns, data = _tabulate_results(timing_results)
  return tabulate_mod.tabulate(data, headers=columns, floatfmt=".4e")


def to_df(timing_results: dict[int, TimingResult] | Callable):
  """Tabulate the timining results sorted fastest first with the `pandas` package."""
  import pandas as pd

  columns, data = _tabulate_results(timing_results)
  index, data = [row[0] for row in data], [row[1:] for row in data]
  return pd.DataFrame(data, index=index, columns=columns[1:])


def tune(
  fn_to_tune: Callable[..., Any],
  hyperparams: dict[Any, Any] | None = None,
  max_workers: int = 32,
  in_shardings: Any = UNSPECIFIED,
  out_shardings: Any = UNSPECIFIED,
  device: jax.Device | _UnspecifiedT = UNSPECIFIED,
  example_args: tuple[Any] | None = None,
  example_kws: dict[Any, Any] | None = None,
  sample_num: int = 2**63 - 1,
  event_filter_regex: str | None = None,
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
      sample_num (int | float): Number of samples used for tuning. Defaults to full cartesian product (all samples).
      event_filter_regex (str | None): A regex to count only matching events into the runtime.
  """

  hyperparams_ = hyperparams if hyperparams is not None else dict()

  def _get_best_hyperparams(args, kws):
    """Main tuning method."""

    # resolve sharding and/or device placement #################################
    _maybe_aval = lambda x: x if not isinstance(x, jax.Array) else jax.typeof(x)
    if len(args) == 0 or all(x is None or jax.core.is_concrete(x) for x in jax.tree.leaves(args)):
      logger.debug("All arguments are concrete, no need to pick random values.")
      args_val = args
    elif example_args is not None:
      logger.debug("Example arguments provided")
      args_val = example_args
      if in_shardings is not UNSPECIFIED or device is not UNSPECIFIED:
        raise ValueError(
          "`example_args` can't be used with in_shardings or device. `example_args` should be correctly sharded."
        )
    else:
      logger.debug("Selecting random input arguments.")
      resolved_device = device if isinstance(device, jax.Device) else _get_default_device()
      if isinstance(resolved_device, str):  # in case the default device is a string `with jax.default_device("cpu"):`
        resolved_device = jax.devices(resolved_device)[0]
      shardings = in_shardings if in_shardings is not UNSPECIFIED else jax.tree.map(lambda _: None, args)
      shardings = (shardings,) if len(args) == 1 else shardings
      shardings = jax.tree.map(
        partial(_normalize_sharding, default_device=resolved_device), tuple(args), tuple(shardings)
      )
      args_val = jax.tree.map(lambda x, s: _get_random_value(_maybe_aval(x), s), args, shardings)

    if len(kws) == 0 or all(v is None or jax.core.is_concrete(v) for v in kws.values()):
      logger.debug("All keyword arguments are concrete, no need to pick random values.")
      kws_val = kws
    elif example_kws is not None:
      logger.debug("Example keyword arguments provided")
      kws_val = example_kws
    else:
      logger.debug("Selecting random keyword arguments.")
      kws_val = jax.tree.map(lambda x: _get_random_value(_maybe_aval(x)), kws)

    hyperparams_norm = {k: (v if isinstance(v, (tuple, list)) else (v,)) for k, v in hyperparams_.items()}
    executor = ThreadPoolExecutor(max_workers=max_workers)

    fns = dict()
    hyperparam_settings = dict(enumerate(itertools.product(*hyperparams_norm.values())))
    if sample_num < len(hyperparam_settings):
      sample_idx = sorted(pyrandom.sample(list(range(len(hyperparam_settings))), k=sample_num))
      hyperparam_settings_ = list(hyperparam_settings.items())
      hyperparam_settings = dict([hyperparam_settings_[idx] for idx in sample_idx])
    if len(hyperparam_settings) == 0:
      hyperparam_settings[0] = []  # allow no hyperparamters to tune, just time the function

    with _global_tuning_lock:
      # filter hyperparameters for those that compile ##########################
      optimal_formats = {}
      for it in range(2):  # sometimes a kernel compiles once, but not twice
        compile_only, find_optimal_layouts = (it == 0), CONFIG.find_optimal_layouts_automatically
        compiles: dict[Future[CompileResult], int] = dict()
        for i, vals in hyperparam_settings.items():
          hs = dict(zip(hyperparams_norm.keys(), vals, strict=True))
          fns[i] = _make_fn_to_time(fn_to_tune, hs, out_shardings=out_shardings, name_id=i)
          # first time, try compiling only (to check if lowering and compilation are error free)
          opts = dict(optimal_formats=optimal_formats.get(i, None), compute_layouts=find_optimal_layouts)
          compiles[executor.submit(_try_call, fns[i], args_val, kws_val, compile_only=compile_only, **opts)] = i

        # collect compiled results
        future_pbar = tqdm(total=len(compiles), disable=logger.level > logging.INFO, desc="Compiling...")
        successful_compiles = {}
        for fut in as_completed(compiles):
          result = fut.result()
          if result.status:
            successful_compiles[compiles[fut]] = result
          future_pbar.update(1)
        future_pbar.close()

        # analyze compiled results
        if compile_only and find_optimal_layouts:
          for k, x in successful_compiles.items():
            optimal_formats[k] = x.optimal_formats
        if len(successful_compiles) == 0:
          for compile_result, i in compiles.items():
            logger.error(
              f"Hyperparameters {hyperparam_settings[i]} failed to compile with message:"
              f"\n{compile_result.result().error_msg}"
            )
          raise ValueError("No hyperparameters compiled successfully")
        logger.debug("Down to %d hyperparameters", len(successful_compiles))
        # cleanup
        hyperparam_settings = {i: hyperparam_settings[i] for i in successful_compiles.keys()}
        fns = {i: fns[i] for i in successful_compiles.keys()}

    # sequentially time the remaining hyperparameters ##########################

    results = dict()
    try:
      args_with_device = [list(args.devices())[0] for args in jax.tree.leaves(args_val) if hasattr(args, "devices")]
      if len(args_with_device) > 0:
        platform = args_with_device[0].platform
      else:
        platform = _get_default_device().platform

      def _timing_closure():  # the _time_fn will acquire the lock on its own
        hs = list(hyperparam_settings.items())
        pyrandom.shuffle(hs)
        for i, _ in hs:
          _try_call(fns[i], args_val, kws_val, compile_only=False, optimal_formats=optimal_formats.get(i, None))

      profiler_timings = _experimental_time_with_profiler(
        _timing_closure, platform, CONFIG.profiling_samples, event_filter_regex=event_filter_regex
      )
      fraction_measured = sum(1 for i in hyperparam_settings.keys() if i in profiler_timings) / len(hyperparam_settings)
      if fraction_measured < CONFIG.must_find_at_least_profiler_result_fraction:
        msg = "Could not find profiler results for some hyperparameter settings:"
        for i in [i for i in hyperparam_settings.keys() if i not in profiler_timings]:
          msg += f"\n  - {i}: {hyperparam_settings[i]}"
        raise RuntimeError(msg)
      else:
        for i in hyperparam_settings.keys():
          if i not in profiler_timings:
            logger.warning(f"Could not find profiler results for hyperparameter settings: {hyperparam_settings[i]}")
            profiler_timings[i] = (float("inf"), float("inf"))  # nan doesn't work here since it sorts improperly
      for i, hs in hyperparam_settings.items():
        hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
        results[i] = TimingResult(hs, *profiler_timings[i])
    except Exception as e:
      if not CONFIG.allow_fallback_timing:
        print(traceback.format_exc())
        raise RuntimeError(f"Need to fall back to the python-level timing, but {CONFIG=} prohibits it.")
      # old timing fallback
      logger.warning(traceback.format_exc())
      logger.warning("Could not time with the profiler, falling back to Python-level timing")
      _opts = dict(total=len(hyperparam_settings), disable=logger.level > logging.INFO, desc="Timing...")
      hs_pbar = tqdm(hyperparam_settings.items(), **_opts)
      for i, hs in hs_pbar:
        hs = dict(zip(hyperparams_norm.keys(), hs, strict=True))
        results[i] = TimingResult(hs, *_time_fn(partial(lambda fn: fn(*args_val, **kws_val), fns[i]), repeat=10))

    results = sorted(results.items(), key=lambda x: _timing_loss(x[1]))
    idx, optimal_hyperparams = results[0][0], results[0][1].hyperparams
    logger.debug("\n" + (tabulate(dict(results)) if tabulate_mod is not None else pformat(dict(results), width=300)))
    logger.debug(f"optimal hyperparams: {optimal_hyperparams}")
    return fns[idx], optimal_hyperparams, results

  if hasattr(fn_to_tune, "timing_result"):
    raise ValueError("Wrapping a `tune`d function in the `tune` decorator the second time is not supported.")

  @wraps(fn_to_tune)
  def wrapped_fn(*args, **kws):
    maybe_hash = _try_hash_input(args, kws)
    if maybe_hash is not None and maybe_hash in wrapped_fn.hyperparams_cache:
      optimal_hyperparameters, results = wrapped_fn.hyperparams_cache[maybe_hash]
    else:
      with jax.core.eval_context():
        _, optimal_hyperparameters, results = _get_best_hyperparams(args, kws)
      if maybe_hash is not None:
        wrapped_fn.hyperparams_cache[maybe_hash] = (optimal_hyperparameters, results)
    wrapped_fn.timing_results.clear()
    wrapped_fn.timing_results.update(results)
    wrapped_fn.optimal_hyperparams.clear()
    wrapped_fn.optimal_hyperparams.update(optimal_hyperparameters)
    return fn_to_tune(*args, **dict(kws, **optimal_hyperparameters))

  wrapped_fn.timing_results = {}
  wrapped_fn.hyperparams_cache = {}
  wrapped_fn.optimal_hyperparams = {}
  return wrapped_fn


# -------------------------------------------------------------------------------


def test_main():
  from jax.experimental.pallas.ops.gpu import attention

  hyperparams = {
    "block_q": [4, 8, 16, 32, 64, 128],
    "block_k": [4, 8, 16, 32, 64, 128],
    "segment_ids": None,  # scalars are ok
  }

  b, qt, h, d = 8, 32, 8, 512
  kt = 128

  q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

  attention_wrapper = lambda *args, block_q, block_k, **kw: attention.mha(
    *args,
    **dict(kw, block_sizes=attention.BlockSizes(block_q=block_q, block_k=block_k)),
  )

  tuned_mha = tune(attention_wrapper, hyperparams=hyperparams, sample_num=5)
  tuned_mha_jit = jax.jit(tuned_mha)

  logger.setLevel("DEBUG")

  tuned_mha_jit(q, k, v).block_until_ready()
  tuned_mha_jit(q, k, v).block_until_ready()
  q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  tuned_mha_jit(q, k, v).block_until_ready()
  tuned_mha_jit(q, k, v).block_until_ready()

  print(tuned_mha_jit.timing_results)  # to get access to timing results

  return


if __name__ == "__main__":
  test_main()
