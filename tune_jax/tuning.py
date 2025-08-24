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
import io
from functools import partial, wraps
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
import logging
from pprint import pformat
from pathlib import Path
import random as pyrandom
import textwrap
import inspect

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

__all__ = ["tune", "record"]

TUNE_FN_PREFIX_FMT = "tune_jax_fn_{}"


@dataclasses.dataclass
class _Config:
  allow_fallback_timing: bool = True
  must_find_at_least_profiler_result_fraction: float = 0.5
  profiling_samples: int = 5


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


@contextlib.contextmanager
def suppress_stdout_stderr():
  devnull, stdout, stderr = open(os.devnull, "w+"), os.dup(1), os.dup(2)
  os.dup2(devnull.fileno(), 1), os.dup2(devnull.fileno(), 2)
  yield
  os.dup2(stdout, 1), os.dup2(stderr, 2)


def _try_call(fn: Callable[[], None], args_val, kws_val, compile_only: bool = False) -> CompileResult:
  """Attempt to call the function and return whether it compiles and runs."""
  try:
    if compile_only:
      _ = jax.jit(fn).lower(*args_val, **kws_val).compile()
    else:
      _ = jax.block_until_ready(fn(*args_val, **kws_val))
    return CompileResult(True, None)
  except Exception as _:
    msg = traceback.format_exc()
    return CompileResult(False, msg)


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
  for it in tqdm(range(total_calls_number), desc=f"Profiling {platform}", disable=logger.level > logging.INFO):
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    profile_path = Path(tempfile.mkdtemp(prefix=f"tuning_profile_{now}_")).absolute()
    if it == 0:
      logger.debug("Saving optimization profile to `%s`", profile_path)
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
    profile_events = profile_reader.get_events_from_plane(profile_proto, device_plane_id, verbose=False)
    fn_format = f"jit_{TUNE_FN_PREFIX_FMT.format('([0-9]+)')}.*"
    for k, durations in profile_events.items():
      if not re.match(fn_format, k):
        continue
      key = int(re.match(fn_format, k)[1])
      assert len(durations) == 1, "We are expecting a single call per profile"
      function_timings.setdefault(key, []).extend(durations)

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


def codegen_a_tuning_script(fn: Callable, args: tuple, kws: dict, dir_or_buffer: str | io.IOBase | None):
  qualname, module, fname, source_code = fn.__qualname__, fn.__module__, fn.__name__, inspect.getsource(fn)

  class Literal:  # allows pformat to interpolate strings like my_name instead of "my_name"
    def __init__(self, val: Any):
      self.val = val

    def __repr__(self) -> str:
      return self.val

  total_arrays = 0

  def init_obj(x):
    nonlocal total_arrays
    _normal_init = "random.normal(next(keys), {shape}, dtype='{dtype}')"
    _const_init = "jnp.full({shape}, {value}, dtype='{dtype}')"
    if isinstance(x, jax.Array):
      total_arrays += 1
      try:
        local_shape = x.sharding.shard_shape(x.shape)
      except AttributeError:
        local_shape = x.shape
      if jnp.issubdtype(x.dtype, jnp.floating):
        return Literal(_normal_init.format(shape=str(local_shape), dtype=x.dtype.name))
      else:
        return Literal(_const_init.format(shape=local_shape, value=1, dtype=x.dtype))
    else:
      return Literal(str(x))

  args_init, kw_init = jax.tree.map(init_obj, args), jax.tree.map(init_obj, kws)
  if "<locals>" not in qualname:
    import_statement = f"from {module} import {fname}"
  else:
    import_statement = textwrap.dedent(source_code)
    if "<lambda>" in qualname:  # strip all before the first "lambda" in the source code and give it a name
      fname = "my_lambda"
      import_statement = f"{fname} = {import_statement[max(import_statement.find('lambda'), 0) :].lstrip()}"
  code = f"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.layout import Format, Layout

import tune_jax

# your function #########################################
# (some imports might be missing)
{import_statement.strip()}

# hyperparameters #######################################
hyperparams = {{
    # FILL ME IN
}}

# random inputs #########################################
# (hyperparameters have been wrongly placed in the args/kws)
keys = iter(random.split(random.key(0), {total_arrays}))
args = {textwrap.indent(pformat(args_init, width=120), " " * 7).strip()}
kws = {textwrap.indent(pformat(kw_init, width=120), " " * 6).strip()}

# optimal layouts microbenchmarking #####################
xs_flat, xs_struct = jax.tree.flatten((args, kws))
xs_arr = [x if isinstance(x, jax.Array) else None for x in xs_flat]
xs_obj = [x if not isinstance(x, jax.Array) else None for x in xs_flat]

def fn_flat(*xs_arr_flat):
  xs_flat = [x if x is not None else y for x, y in zip(xs_arr_flat, xs_obj)]
  args_, kws_ = jax.tree.unflatten(xs_struct, xs_flat)
  fn_with_hyperparams = partial({fname}, )  # FILL ME IN: hyperparams that will definitely compile
  return fn_with_hyperparams(*args_, **kws_)

shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if x is not None else None for x in xs_arr]
formats_flat = jax.jit(fn_flat, in_shardings=Format(Layout.AUTO)).lower(*xs_arr).compile().input_formats[0]
formats = jax.tree.unflatten(xs_struct, formats_flat)

(args, kws) = jax.tree.map(lambda x, l: jax.device_put(x, l) if l is not None else x, (args, kws), formats)

# tuning ################################################
fn = tune_jax.tune({fname}, hyperparams=hyperparams)
fn(*args, **kws)
print(tune_jax.tabulate(fn))
"""
  if dir_or_buffer is None:
    logger.info("Auto-tuning script " + "-" * 61 + "\n" + code + "\n" + "-" * 80)
  elif isinstance(dir_or_buffer, io.TextIOBase):
    dir_or_buffer.write(code)
  elif isinstance(dir_or_buffer, io.IOBase):
    dir_or_buffer.write(code.encode())
  elif hasattr(dir_or_buffer, "write"):
    dir_or_buffer.write(code)
  else:
    path = Path(dir_or_buffer).expanduser().absolute()
    path.mkdir(exist_ok=True, parents=True)
    assert path.exists()
    tuning_filename = str(time.time_ns())
    tuning_path = path / (f"{re.sub(r'(<|>)', '_', qualname)}_{tuning_filename}.py")
    tuning_path.write_text(code)


def record(fn: Callable, dir_or_buffer: str | io.IOBase | None = None):
  """Record a function call (under jit is ok) to remember its input arguments shapes for tuning.

  Example:
    ```
    @tune_jax.record
    def my_library_function(...):
        ...
    ```

  Optionally codegen a simple tuning template script.
  """
  seen_hashes = {}

  @wraps(fn)
  def _recorded_fn(*args, **kws):
    nonlocal seen_hashes
    input_hash = _try_hash_input(args, kws, must_be_concrete=False)
    if input_hash is not None and input_hash not in seen_hashes:
      seen_hashes[input_hash] = True
      fn_: Callable = fn
      while hasattr(fn_, "_fun"):  # unpack PjitFunctions, it's jitted
        fn_ = getattr(fn_, "_fun")
      module, fname, code = fn_.__module__, fn_.__name__, fn_.__code__
      recorded_args = jax.tree.map(lambda x: x if not isinstance(x, jax.Array) else jax.typeof(x), args)
      recorded_kws = jax.tree.map(lambda x: x if not isinstance(x, jax.Array) else jax.typeof(x), kws)
      args_str = ", ".join(map(pformat, recorded_args))
      kw_str = ", ".join([f"{k}={pformat(v)}" for k, v in recorded_kws.items()])
      logger.info(f"Called {code.co_filename}:{code.co_firstlineno}\n{module}.{fname}({', '.join([args_str, kw_str])})")
      codegen_a_tuning_script(fn_, args, kws, dir_or_buffer)
    return fn(*args, **kws)

  return _recorded_fn


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
      for it in range(2):  # sometimes a kernel compiles once, but not twice
        compiles: dict[int, Future[CompileResult]] = dict()
        for i, vals in hyperparam_settings.items():
          hs = dict(zip(hyperparams_norm.keys(), vals, strict=True))
          fns[i] = _make_fn_to_time(fn_to_tune, hs, out_shardings=out_shardings, name_id=i)
          # first time, try compiling only (to check if lowering and compilation are error free)
          compiles[i] = executor.submit(
            lambda fn, args_val, kws_val: _try_call(fn, args_val, kws_val, it == 0), fns[i], args_val, kws_val
          )
        future_pbar = tqdm(
          compiles.items(), total=len(compiles), disable=logger.level > logging.INFO, desc="Compiling..."
        )
        successful_compiles = {k: x.result() for (k, x) in future_pbar if x.result().status}
        if len(successful_compiles) == len(hyperparam_settings):
          break
        if len(successful_compiles) == 0:
          for i, compile_result in compiles.items():
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
          _time_fn(partial(lambda fn: fn(*args_val, **kws_val), fns[i]), repeat=1, number=1)

      profiler_timings = _experimental_time_with_profiler(_timing_closure, platform, CONFIG.profiling_samples)
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
    except Exception as _:
      if not CONFIG.allow_fallback_timing:
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
    return fn_to_tune(*args, **dict(kws, **optimal_hyperparameters))

  wrapped_fn.timing_results = {}
  wrapped_fn.hyperparams_cache = {}
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
  tuned_mha_jit = record(jax.jit(tuned_mha), codegen_tuning_script_dir="/tmp/tuning_mha")

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
