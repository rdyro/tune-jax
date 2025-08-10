from pathlib import Path
import tempfile
import io
from functools import partial

from absl.testing import absltest, parameterized
import jax

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
    fn4 = tune_jax.tune(tune_jax.record(jax.jit(fn), codegen_dir_or_buffer=tempdir))
    fn4(1)
    self.assertNotEmpty(list(Path(tempdir).glob("*.py")))

    # record with output codegen directory successfully creates new directories
    fn5 = tune_jax.tune(tune_jax.record(jax.jit(fn), codegen_dir_or_buffer=Path(tempdir) / "new_folder"))
    fn5(1)
    self.assertTrue((Path(tempdir) / "new_folder").exists())
    self.assertNotEmpty(list((Path(tempdir) / "new_folder").glob("*.py")))

  def test_record_with_jit(self):
    tune_jax.logger.setLevel("INFO")

    buffer = io.BytesIO()

    @jax.jit
    @partial(tune_jax.record, codegen_dir_or_buffer=buffer)
    def fn1(x):
      return x

    fn1(1)

    @partial(tune_jax.record, codegen_dir_or_buffer=buffer)
    @jax.jit
    def fn2(x):
      return x

    fn2(1)

    buffer.seek(0)
    print(buffer.read().decode())


if __name__ == "__main__":
  absltest.main()
