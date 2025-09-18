from functools import partial

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp

import tune_jax

tune_jax.logger.setLevel("DEBUG")


def _module_fn(x):
  return x


class InterfaceTest(parameterized.TestCase):
  def setUp(self):
    @jax.jit
    def fn(x):
      return x

    self.local_fn = fn
    self.lam = lambda x: x
    self.module_fn = _module_fn

    return super().setUp()

  def test_error(self):
    def fn(x):
      raise ValueError

    with self.assertRaisesRegex(ValueError, "No hyperparameters compiled successfully"):
      tune_jax.tune(fn, hyperparams={"a": [1]})(1)

  def test_scalar_hyperparam(self):
    def fn(x, a, b):
      return x

    tune_jax.tune(fn, hyperparams={"a": [1], "b": 2})(1)

  def test_empty(self):
    def fn(x):
      return x

    fn_tuned = tune_jax.tune(fn)
    fn_tuned(1)
    print(tune_jax.tabulate(fn_tuned.timing_results))
    print(tune_jax.tabulate(fn_tuned))  # this should automatically look for timing_results attribute

  @parameterized.parameters([True, False])
  def test_optimal_hyperparams_field(self, jit: bool):
    def fn(x, a, b):
      return x

    fn = jax.jit(fn) if jit else fn

    fn_tuned = tune_jax.tune(fn, hyperparams={"a": list(range(10)), "b": 2})
    fn_tuned = jax.jit(fn_tuned) if jit else fn_tuned
    fn_tuned(1)

    self.assertTrue(hasattr(fn_tuned, "timing_results"))
    self.assertTrue(hasattr(fn_tuned, "optimal_hyperparams"))
    optimal_hyperparams = sorted(fn_tuned.timing_results.items(), key=lambda x: tune_jax.tuning._timing_loss(x[1]))[0][
      1
    ].hyperparams
    self.assertEqual(tuple(optimal_hyperparams.items()), tuple(fn_tuned.optimal_hyperparams.items()))

  def test_tabulate_results(self):
    def fn(x, a, b):
      return x

    fn_tuned = tune_jax.tune(fn, hyperparams={"a": list(range(10)), "b": 2})
    fn_tuned(1)
    print(tune_jax.tabulate(fn_tuned.timing_results))
    print(tune_jax.tabulate(fn_tuned))

  def test_nested_within_jit(self):
    @partial(tune_jax.tune, hyperparams=dict(splits=[1, 2, 4]))
    def fn(A, B, splits):
      A_, B_ = jnp.split(A, splits, axis=1), jnp.split(B, splits, axis=0)
      acc = 0
      for i in range(splits):
        acc += A_[i] @ B_[i]
      return acc

    @jax.jit
    def compute_fn(X):
      return X + fn(X, X)

    X = jnp.arange(16 * 16).astype(jnp.float32).reshape((16, 16))

    self.assertEmpty(fn.timing_results)
    y = compute_fn(X)
    self.assertNotEmpty(fn.timing_results)


if __name__ == "__main__":
  absltest.main()
