from typing import Callable, Any
import io
import textwrap
from pathlib import Path
from functools import wraps
import inspect
from pprint import pformat
import time
import re

import jax
import jax.numpy as jnp

from ..tuning import _try_hash_input, logger


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
