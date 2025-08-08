import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.pallas.ops.gpu import attention as attention_gpu
from absl.testing import absltest

from tune_jax import tune, tune_logger
from tune_jax import tuning

tune_logger.setLevel("DEBUG")


def platforms_available(*platforms):
  for platform in platforms:
    try:
      jax.devices(platform)
      return True
    except:
      pass
  return False


# a collection of functions to tune ################################################################


def _matmul(x, y, dummy):
  del dummy
  return x @ y


def _cond_fn(c, fn1, fn2, *args):
  return jax.lax.cond(c, lambda: fn1(*args), lambda: fn2(*args))


def _gpu_attention(q, k, v, block_q, block_k):
  if hasattr(attention_gpu, "BlockSizes"):
    attention_fn = lambda *args, block_q=None, block_k=None, **kw: attention_gpu.mha(
      *args,
      **dict(kw, block_sizes=attention_gpu.BlockSizes(block_q=block_q, block_k=block_k)),
    )
  else:
    attention_fn = attention_gpu.mha
  return attention_fn(q, k, v, segment_ids=None, block_q=block_q, block_k=block_k)


def _long_while(it, x, y):
  return jax.lax.fori_loop(0, it, lambda i, carry: (x @ y) / jnp.linalg.norm(carry + x @ y), (x @ y))


####################################################################################################

class ProfileReadingTest(absltest.TestCase):
  def test_parsing_multiple_profiles_on_gpu(self):
    if not platforms_available("gpu", "tpu"):
      self.skipTest("No GPU or TPU available")
    try:
      tuning.CONFIG.allow_fallback_timing = False
      hyperparams = {
        "block_q": [4, 8, 16, 32, 64, 128],
        "block_k": [4, 8, 16, 32, 64, 128],
      }
      b, qt, h, d, kt = 8, 32, 8, 512, 128
      q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
      k = random.normal(random.key(1), (b, kt, h, d), dtype=jnp.bfloat16)
      v = random.normal(random.key(2), (b, kt, h, d), dtype=jnp.bfloat16)
      if platforms_available("gpu"):
        tune(_gpu_attention, hyperparams=hyperparams)(q, k, v).block_until_ready()
        jax.jit(tune(_gpu_attention, hyperparams=hyperparams))(q, k, v).block_until_ready()

      x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
      y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)

      tune(_matmul, hyperparams={"dummy": [1]})(x, y)
      jax.jit(tune(_matmul, hyperparams={"dummy": [1]}))(x, y)

      _fn = lambda x, y, c: _cond_fn(c, lambda x, y: x @ y, lambda x, y: x + y, x, y)
      tune(_fn, hyperparams={"c": [0, 1, 2]})(x, y)
      jax.jit(tune(_fn, hyperparams={"c": [0, 1, 2]}))(x, y)

      _fn = lambda x, y, it: _long_while(it, x, y)
      tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]})(x, y)
      jax.jit(tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]}))(x, y)
    finally:
      tuning.CONFIG.allow_fallback_timing = True


if __name__ == "__main__":
  absltest.main()
