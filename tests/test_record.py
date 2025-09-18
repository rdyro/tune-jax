from pathlib import Path
import sys
import tempfile
import io
from functools import partial

from absl.testing import absltest, parameterized
import jax

import tune_jax
from tune_jax.experimental import record


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

  @parameterized.product(fn_type=["local", "lambda", "module"])
  def test_record(self, fn_type: str):
    fn = {"local": self.local_fn, "lambda": self.lam, "module": self.module_fn}[fn_type]

    fn1 = record.record(fn)
    fn1(1)

    fn2 = record.record(jax.jit(fn))
    fn2(1)

    fn3 = tune_jax.tune(record.record(jax.jit(fn)))
    fn3(1)
    print(tune_jax.tabulate(fn3))

    # record with output codegen successfully generates files
    tempdir = tempfile.mkdtemp()
    print(f"{tempdir = }")
    fn4 = tune_jax.tune(record.record(jax.jit(fn), dir_or_buffer=tempdir))
    fn4(1)
    self.assertNotEmpty(list(Path(tempdir).glob("*.py")))

    # record with output codegen directory successfully creates new directories
    fn5 = tune_jax.tune(record.record(jax.jit(fn), dir_or_buffer=Path(tempdir) / "new_folder"))
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
    @partial(record.record, dir_or_buffer=buffer)
    def fn1(x):
      return x

    fn1(1)

    @partial(record.record, dir_or_buffer=buffer)
    @jax.jit
    def fn2(x):
      return x

    fn2(1)

    if buffer_type in ("bytes", "text"):
      buffer.seek(0)
      print(buffer.read())


if __name__ == "__main__":
  absltest.main()
