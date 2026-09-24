from functools import partial
import os
from pathlib import Path

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
    def fn(x, a):
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

  @parameterized.parameters([True, False])
  def test_wrap_unjitted_fn_in_jit(self, wrap: bool):
    traces = []

    def fn(x, a):
      traces.append(a)
      for _ in range(a):  # requires a static hyperparameter
        x = x + 1
      return x

    wrap_default, tune_jax.CONFIG.wrap_unjitted_fn_in_jit = tune_jax.CONFIG.wrap_unjitted_fn_in_jit, wrap
    try:
      fn_tuned = tune_jax.tune(fn, hyperparams={"a": [1, 2]})
      fn_tuned(jnp.ones(4))  # tunes and, if jitted, populates the jit cache
      traces.clear()
      fn_tuned(jnp.ones(4))
      self.assertEqual(len(traces), 0 if wrap else 1)
    finally:
      tune_jax.CONFIG.wrap_unjitted_fn_in_jit = wrap_default

  def test_double_tune_raises(self):
    fn_tuned = tune_jax.tune(lambda x, a: x, hyperparams={"a": [1]})
    with self.assertRaisesRegex(ValueError, "the second time"):
      tune_jax.tune(fn_tuned, hyperparams={"a": [1]})

  def test_suppress_stdout_stderr_restores_fds(self):
    fd_count = lambda: len(list(Path("/proc/self/fd").iterdir()))
    stdout_stat, fds_before = os.fstat(1), fd_count()
    for _ in range(20):
      with tune_jax.tuning.suppress_stdout_stderr():
        pass
    with self.assertRaises(ValueError), tune_jax.tuning.suppress_stdout_stderr():
      raise ValueError
    self.assertEqual(fd_count(), fds_before)
    self.assertEqual(os.fstat(1).st_ino, stdout_stat.st_ino)

  def test_alias_candidates(self):
    from tune_jax.tuning import _alias_candidates

    hlo, fingerprint = lambda v: ("hlo", v), lambda v: ("fingerprint", v)
    identities = {
      0: (hlo(b"a"),),
      1: (hlo(b"b"), fingerprint(b"F")),
      2: (hlo(b"c"), fingerprint(b"F")),  # same executable as 1 despite lowering differently
      3: (hlo(b"a"), fingerprint(b"G")),  # same lowering as 0
      4: (),  # no identity available, must stay on its own
    }
    self.assertEqual(_alias_candidates(identities, identities.keys()), {0: [0, 3], 1: [1, 2], 4: [4]})

    # aliasing is transitive across the two kinds of key
    chained = {0: (hlo(b"a"),), 1: (hlo(b"a"), fingerprint(b"F")), 2: (hlo(b"z"), fingerprint(b"F"))}
    self.assertEqual(_alias_candidates(chained, chained.keys()), {0: [0, 1, 2]})

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

  def test_kws_tracers(self):
    @partial(tune_jax.tune, hyperparams=dict(splits=[1, 2]))
    def fn(A, *, B, splits):
      A_, B_ = jnp.split(A, splits, axis=1), jnp.split(B, splits, axis=0)
      acc = 0
      for i in range(splits):
        acc += A_[i] @ B_[i]
      return acc

    @jax.jit
    def compute_fn(X):
      return X + fn(X, B=X)

    X = jnp.arange(16 * 16).astype(jnp.float32).reshape((16, 16))

    self.assertEmpty(fn.timing_results)
    y = compute_fn(X)
    self.assertNotEmpty(fn.timing_results)


if __name__ == "__main__":
  absltest.main()
