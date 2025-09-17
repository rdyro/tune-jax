import itertools
import logging
from functools import partial
import math
import traceback

import jax
from jax.experimental.layout import Format, Layout
import numpy as np
from tqdm import tqdm

from .. import logger


def _optimal_formats(fn, *args, **kws):
  is_array = lambda x: isinstance(x, (jax.Array, int, float, complex, np.ndarray, jax.ShapeDtypeStruct))
  args_flat, args_struct = jax.tree.flatten((args, kws))  # combine args and kws into a tuple
  args_array = [x if is_array(x) else None for x in args_flat]  # mask array args
  args_nonarray = [x if not is_array(x) else None for x in args_flat]  # mask non-array args

  def fn_flat_args(*args_array):
    args_kws_flat = [x if x is not None else args_nonarray[i] for i, x in enumerate(args_array)]  # fill non-array args
    args_, kws_ = jax.tree.unflatten(args_struct, args_kws_flat)
    return fn(*args_, **kws_)

  in_shardings = [Format(Layout.AUTO, x.sharding) if x is not None else None for x in args_array]  # get auto in layouts
  shapes = [jax.typeof(x) if x is not None else None for x in args_array]  # shapes for tracing
  args_formats = jax.jit(fn_flat_args, in_shardings=in_shardings).trace(*shapes).lower().compile().input_formats[0]
  return jax.tree.unflatten(args_struct, args_formats)


def optimal_formats(fn, *, hyperparams=None, example_args=None, example_kwargs=None):
  example_args = () if example_args is None else example_args
  example_kwargs = {} if example_kwargs is None else example_kwargs
  hyperparams = {} if hyperparams is None else hyperparams
  hyperparams_norm = {k: (v if isinstance(v, (tuple, list)) else (v,)) for k, v in hyperparams.items()}
  if len(hyperparams) == 0:
    hyperparams_gen, total_hyperams = iter([{}]), 1
  else:
    total_hyperams = math.prod(len(v) for v in hyperparams_norm.values())
    hyperparams_gen = itertools.product(*hyperparams_norm.values())
  all_tracebacks = []
  for hyperparam_settings in tqdm(hyperparams_gen, total=total_hyperams, disable=logger.level <= logging.INFO):
    hyperparam_settings = dict(zip(hyperparams_norm.keys(), hyperparam_settings, strict=True))
    try:
      args_formats, kwargs_formats = _optimal_formats(
        partial(fn, **hyperparam_settings), *example_args, **example_kwargs
      )
      return args_formats, kwargs_formats
    except Exception as _:
      all_tracebacks.append(traceback.format_exc())
  for tb in all_tracebacks:
    logger.error(tb)
  raise ValueError("Could not find optimal formats. All hyperameters failed to compiled.")


def optimal_device_put(fn, *, hyperparams=None, example_args=None, example_kwargs=None):
  example_args = () if example_args is None else example_args
  example_kwargs = {} if example_kwargs is None else example_kwargs
  args_formats, kwargs_formats = optimal_formats(
    fn, hyperparams=hyperparams, example_args=example_args, example_kwargs=example_kwargs
  )
  example_args = jax.tree.map(lambda x, f: jax.device_put(x, f) if f is not None else x, example_args, args_formats)
  example_kwargs = jax.tree.map(
    lambda x, f: jax.device_put(x, f) if f is not None else x, example_kwargs, kwargs_formats
  )
  return example_args, example_kwargs
