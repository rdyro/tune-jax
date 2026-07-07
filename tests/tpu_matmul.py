import functools
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref):
  @pl.when(pl.program_id(2) == 0)
  def init():
    acc_ref[...] = jnp.zeros_like(acc_ref)
  acc_ref[...] += jnp.dot(x_tile_ref[...], y_tile_ref[...], preferred_element_type=acc_ref.dtype)
  o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)


@functools.partial(jax.jit, static_argnames=["block_shape", "block_k"])
def matmul(
  x: jax.Array,
  y: jax.Array,
  *,
  block_shape: tuple[int, int],
  block_k: int = 256,
) -> jax.Array:
  out_dtype = x.dtype
  acc_dtype = jnp.float32
  bm, bn = block_shape
  assert x.shape[0] % bm == 0 and y.shape[1] % bn == 0 and x.shape[1] % block_k == 0
  return pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), out_dtype),
    in_specs=[
      pl.BlockSpec((bm, block_k), lambda i, _, k: (i, k)),
      pl.BlockSpec((block_k, bn), lambda _, j, k: (k, j)),
    ],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    grid=(x.shape[0] // bm, y.shape[1] // bn, x.shape[1] // block_k),
    scratch_shapes=[pltpu.VMEM((bm, bn), acc_dtype)],
    compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
