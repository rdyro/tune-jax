from functools import partial

from absl.testing import absltest
import jax
import jax.core
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AbstractMesh

from tune_jax.experimental.auto_formats import _optimal_formats, optimal_device_put, optimal_formats


class AutoLayoutTest(absltest.TestCase):
  def test_simple(self):
    @partial(jax.jit, static_argnames=("static",))
    def fn(x, A, static="hello"):
      return x @ A

    mesh = AbstractMesh((8, 8), ("x", "y"))
    mesh = jax.make_mesh((1, 1), ("x", "y"))
    x = jax.ShapeDtypeStruct((1024, 1024), jnp.float32, sharding=NamedSharding(mesh, P(None, ("x", "y"))))
    A = jax.ShapeDtypeStruct((1024, 1024), jnp.float32, sharding=NamedSharding(mesh, P(("x", "y"), None)))
    ret = _optimal_formats(fn, x, A)
    ret = _optimal_formats(fn, x, A, static="hello")
    del ret

  def test_static_args(self):
    @partial(jax.jit, static_argnames=("static",))
    def fn(x, A, static="hello"):
      return x @ A

    x, A = jnp.ones((16, 16)), jnp.ones((16, 16))
    ret = _optimal_formats(partial(fn, A=A, static="hi"), x)
    ret = _optimal_formats(fn, x, A, static="hello")
    del ret

  def test_compute_optimal_formats(self):
    x, A = jnp.ones((16, 16)), jnp.ones((16, 16))

    @jax.jit
    def fn(x, *, A):
      return x @ A

    ret = optimal_formats(fn, example_args=(x,), example_kwargs={"A": A})
    ret = optimal_formats(partial(fn, A=A), example_args=(x,))
    ret = optimal_formats(partial(fn, x), example_kwargs={"A": A})

    def fn(x, *, A):
      return x @ A

    ret = optimal_formats(fn, example_args=(x,), example_kwargs={"A": A})
    ret = optimal_formats(partial(fn, A=A), example_args=(x,))
    ret = optimal_formats(partial(fn, x), example_kwargs={"A": A})
    del ret

  def test_device_put_optimal_formats(self):
    x, A = jnp.ones((16, 16)), jnp.ones((16, 16))

    @partial(jax.jit, static_argnames=("static",))
    def fn(x, A, static="hello"):
      return x @ A

    hyperparams = {"static": ["hello", "hi", "bye"]}

    ret = optimal_device_put(fn, hyperparams=hyperparams, example_args=(x,), example_kwargs={"A": A})
    ret = optimal_device_put(partial(fn, A=A), hyperparams=hyperparams, example_args=(x,))
    ret = optimal_device_put(partial(fn, x), hyperparams=hyperparams, example_kwargs={"A": A})
    ret = optimal_device_put(partial(fn, x), example_kwargs={"A": A})
    del ret


if __name__ == "__main__":
  absltest.main()
