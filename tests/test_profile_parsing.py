import os

import jax
from jax import numpy as jnp
from jax import random
from absl.testing import absltest

import tune_jax

tune_jax.logger.setLevel("DEBUG")

TEST_WITH_PALLAS = os.environ.get("TEST_WITH_PALLAS", True)


def platforms_available(*platforms):
  for platform in platforms:
    try:
      jax.devices(platform)
      return True
    except:  # noqa: E722
      pass
  return False


# a collection of functions to tune ################################################################


def _matmul(x, y, dummy):
  del dummy
  return x @ y


def _cond_fn(c, fn1, fn2, *args):
  return jax.lax.cond(c, lambda: fn1(*args), lambda: fn2(*args))


def _tpu_matmul(x, y, block_m, block_n, block_k):
  from tests import tpu_matmul

  return tpu_matmul.matmul(x, y, block_shape=(block_m, block_n), block_k=block_k)


def _long_while(it, x, y):
  return jax.lax.fori_loop(0, it, lambda i, carry: (x @ y) / jnp.linalg.norm(carry + x @ y), (x @ y))


####################################################################################################


class ProfileReadingTest(absltest.TestCase):
  def test_parsing_multiple_profiles(self):
    if not TEST_WITH_PALLAS:
      self.skipTest(f"Skipping pallas kernels since {TEST_WITH_PALLAS=}")

    if platforms_available("tpu"):
      try:
        tune_jax.CONFIG.allow_fallback_timing = False
        hyperparams = {
          "block_m": [256, 512],
          "block_n": [256, 512],
          "block_k": [256, 512],
        }
        x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
        y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)
        tune_jax.tune(_tpu_matmul, hyperparams=hyperparams)(x, y).block_until_ready()
        jax.jit(tune_jax.tune(_tpu_matmul, hyperparams=hyperparams))(x, y).block_until_ready()
      finally:
        tune_jax.CONFIG.allow_fallback_timing = True

    try:
      tune_jax.CONFIG.allow_fallback_timing = False
      x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
      y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)

      tune_jax.tune(_matmul, hyperparams={"dummy": [1]})(x, y)
      jax.jit(tune_jax.tune(_matmul, hyperparams={"dummy": [1]}))(x, y)

      _fn = lambda x, y, c: _cond_fn(c, lambda x, y: x @ y, lambda x, y: x + y, x, y)
      tune_jax.tune(_fn, hyperparams={"c": [0, 1, 2]})(x, y)
      jax.jit(tune_jax.tune(_fn, hyperparams={"c": [0, 1, 2]}))(x, y)

      _fn = lambda x, y, it: _long_while(it, x, y)
      tune_jax.tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]})(x, y)
      jax.jit(tune_jax.tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]}))(x, y)
    finally:
      tune_jax.CONFIG.allow_fallback_timing = True

  def test_sum_events(self):
    from tune_jax.profile_reader.parse_profile import _sum_events

    events = [
      {"start_ps": 0, "end_ps": 10},
      {"start_ps": 5, "end_ps": 15},
      {"start_ps": 15, "end_ps": 20},
      {"start_ps": 25, "end_ps": 30},
      {"start_ps": 25, "end_ps": 30},
    ]
    self.assertEqual(_sum_events(events), 25)


if __name__ == "__main__":
  absltest.main()
