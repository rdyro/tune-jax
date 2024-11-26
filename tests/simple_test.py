import functools

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.pallas.ops.gpu import attention
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import pytest

from tune_jax import tune, tune_logger

tune_logger.setLevel("DEBUG")

def gpu_available():
  try:
    jax.devices("cuda")
    return True
  except:
    return False

def test_matmul_size():
  hyperparams = {
    "n": [128, 1024],
    "m": [128, 2056],
  }

  def fn(n, m):
    keys = random.split(random.key(0), 2)
    A = random.normal(keys[0], (n, m), dtype=jnp.float32)
    B = random.normal(keys[1], (m, n), dtype=jnp.float32)
    C = A @ B
    return C / jnp.linalg.norm(C, axis=-1)[..., None]

  tuned_mha = tune(fn, hyperparams=hyperparams)
  tuned_mha_jit = jax.jit(tuned_mha)

  _ = tuned_mha_jit().block_until_ready()
  C = tuned_mha_jit().block_until_ready()
  assert C.shape[-1] == 128

@pytest.mark.skipif(not gpu_available(), reason="No GPU available")
def test_simple_mha():
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

  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
  tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()

@pytest.mark.skipif(not gpu_available(), reason="No GPU available")
def test_multidevice():
  hyperparams = {
    "block_q": [4, 8, 16, 32, 64, 128],
    "block_k": [4, 8, 16, 32, 64, 128],
  }

  b, qt, h, d = 8, 32, 8, 512
  kt = 128

  if len(jax.devices()) < 2:
    return

  mesh = Mesh(devices=jax.devices(), axis_names=("x",))

  q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

  in_shardings = tuple(NamedSharding(mesh, P(*(["x"] + [None] * (z.ndim - 1))))
                       for z in [q, k, v])
  q, k, v = jax.tree.map(lambda x, y: jax.device_put(x, y),
                         [q, k, v], in_shardings)

  tuned_mha = tune(functools.partial(attention.mha, segment_ids=None),
                   hyperparams=hyperparams, in_shardings=in_shardings)
  tuned_mha_jit = jax.jit(tuned_mha, in_shardings=in_shardings) # type: ignore

  tuned_mha_jit(q, k, v).block_until_ready()
  tuned_mha_jit(q, k, v).block_until_ready()
  q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
  k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
  tuned_mha_jit(q, k, v).block_until_ready()
  tuned_mha_jit(q, k, v).block_until_ready()

if __name__ == "__main__":
  test_matmul_size()
  test_simple_mha()
  test_multidevice()
