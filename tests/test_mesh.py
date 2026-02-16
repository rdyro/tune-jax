from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from absl.testing import absltest, parameterized
import tune_jax

try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map

tune_jax.logger.setLevel("DEBUG")


class MeshTuningTest(parameterized.TestCase):
  @parameterized.product(axis_type=[AxisType.Explicit, AxisType.Auto])
  def test_tuning_under_mesh_tracers(self, axis_type):
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("x",), axis_types=(axis_type,))

    @jax.jit
    def tuned_fn(x):
      def inner(x, h): return x * h
      tuned_inner = tune_jax.tune(inner, hyperparams={"h": [1, 2]})
      ret = tuned_inner(x)
      print(tune_jax.tabulate(tuned_inner.timing_results))
      return ret

    with jax.sharding.set_mesh(mesh):
      x = jax.device_put(jnp.ones((8, 8)), NamedSharding(mesh, P("x", None)))
      y = tuned_fn(x)
      self.assertEqual(y.shape, (8, 8))

  @parameterized.product(axis_type=[AxisType.Explicit, AxisType.Auto])
  def test_custom_sharding_random_args(self, axis_type):
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("x",), axis_types=(axis_type,))
    sharding = NamedSharding(mesh, P("x", None))

    def fn(x, h): return x + h

    @jax.jit
    def outer_fn(x):
      # x is a tracer. We pass in_shardings to tune.
      tuned_fn = tune_jax.tune(fn, hyperparams={"h": [1.0, 2.0]}, in_shardings=(sharding,))
      return tuned_fn(x)

    with jax.sharding.set_mesh(mesh):
      x = jax.device_put(jnp.ones((8, 8)), sharding)
      y = outer_fn(x)
      self.assertEqual(y.shape, (8, 8))

  @parameterized.product(axis_type=[AxisType.Explicit, AxisType.Auto])
  def test_custom_sharding_random_args_outer(self, axis_type):
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("x",), axis_types=(axis_type,))
    sharding = NamedSharding(mesh, P("x", None))

    def fn(x, h): return x + h

    with jax.sharding.set_mesh(mesh):
      x_aval = jax.ShapeDtypeStruct((8, 8), jnp.float32)
      tuned_fn = tune_jax.tune(fn, hyperparams={"h": [1.0, 2.0]}, in_shardings=(sharding,))
      res = jax.eval_shape(tuned_fn, x_aval)
      self.assertEqual(res.shape, (8, 8))
      self.assertNotEmpty(tuned_fn.timing_results)

  @parameterized.product(axis_type=[AxisType.Explicit, AxisType.Auto])
  def test_mesh_leak(self, axis_type):
    def get_current_mesh():
      try:
        return jax.sharding.get_mesh()
      except:  # noqa: E722
        return None

    initial_mesh = get_current_mesh()

    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("x",), axis_types=(axis_type,))

    @jax.jit
    def tuned_fn(x):
      def inner(x, h): return x * h
      return tune_jax.tune(inner, hyperparams={"h": [1, 2]})(x)

    with jax.sharding.set_mesh(mesh):
      x = jax.device_put(jnp.ones((8, 8)), NamedSharding(mesh, P("x", None)))
      _ = tuned_fn(x)

    final_mesh = get_current_mesh()
    self.assertEqual(initial_mesh, final_mesh)

  @parameterized.product(axis_type=[AxisType.Explicit, AxisType.Auto])
  def test_shard_map_tuning(self, axis_type):
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("x",), axis_types=(axis_type,))

    def inner_fn(x, h):
      return x * h

    # tuning under shard_map is not currently supported, tune a shard_map closure
    @partial(jax.jit, static_argnums=(1,))
    def outer_fn(x, mesh):
      @partial(tune_jax.tune, hyperparams={"h": [1, 2]})
      def fn(x, h):
        @partial(shard_map, mesh=mesh, in_specs=P("x", None), out_specs=P("x", None))
        def sharded_tuning(x_local):
          return inner_fn(x_local, h)
        return sharded_tuning(x)

      return fn(x)

    with jax.sharding.set_mesh(mesh):
      x = jnp.ones((len(devices) * 8, 8))
      sharding = NamedSharding(mesh, P("x", None))
      x = jax.device_put(x, sharding)
      y = outer_fn(x, mesh)
      self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
  absltest.main()
