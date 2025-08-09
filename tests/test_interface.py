from absl.testing import absltest

import tune_jax


class InterfaceTest(absltest.TestCase):
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

  def test_tabulate_results(self):
    def fn(x, a, b):
      return x

    fn_tuned = tune_jax.tune(fn, hyperparams={"a": list(range(10)), "b": 2})
    fn_tuned(1)
    print(tune_jax.tabulate(fn_tuned.timing_results))
    print(tune_jax.tabulate(fn_tuned))


if __name__ == "__main__":
  absltest.main()
