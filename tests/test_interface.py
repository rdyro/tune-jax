from pathlib import Path
import sys
import tempfile
import io
from functools import partial

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp

import tune_jax


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

  @parameterized.product(fn_type=["local", "lambda", "module"])
  def test_record(self, fn_type: str):
    fn = {"local": self.local_fn, "lambda": self.lam, "module": self.module_fn}[fn_type]

    fn1 = tune_jax.record(fn)
    fn1(1)

    fn2 = tune_jax.record(jax.jit(fn))
    fn2(1)

    fn3 = tune_jax.tune(tune_jax.record(jax.jit(fn)))
    fn3(1)
    print(tune_jax.tabulate(fn3))

    # record with output codegen successfully generates files
    tempdir = tempfile.mkdtemp()
    print(f"{tempdir = }")
    fn4 = tune_jax.tune(tune_jax.record(jax.jit(fn), dir_or_buffer=tempdir))
    fn4(1)
    self.assertNotEmpty(list(Path(tempdir).glob("*.py")))

    # record with output codegen directory successfully creates new directories
    fn5 = tune_jax.tune(tune_jax.record(jax.jit(fn), dir_or_buffer=Path(tempdir) / "new_folder"))
    fn5(1)
    self.assertTrue((Path(tempdir) / "new_folder").exists())
    self.assertNotEmpty(list((Path(tempdir) / "new_folder").glob("*.py")))

  @parameterized.product(buffer_type=["bytes", "text", "stdout", None])
  def test_record_with_jit(self, buffer_type):
    tune_jax.logger.setLevel("DEBUG")

    if buffer_type is not None:
      buffer = dict(bytes=io.BytesIO(), text=io.TextIOWrapper(io.BytesIO()), stdout=sys.stdout)[buffer_type]
    else:
      buffer = None

    @jax.jit
    @partial(tune_jax.record, dir_or_buffer=buffer)
    def fn1(x):
      return x

    fn1(1)

    @partial(tune_jax.record, dir_or_buffer=buffer)
    @jax.jit
    def fn2(x):
      return x

    fn2(1)

    if buffer_type in ("bytes", "text"):
      buffer.seek(0)
      print(buffer.read())


if __name__ == "__main__":
  absltest.main()
