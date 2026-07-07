from functools import partial
import os

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.experimental.layout import Format, Layout

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


class SplashCasesTest(absltest.TestCase):
  def test_tune_splash(self):
    if not TEST_WITH_PALLAS:
      self.skipTest(f"Skipping pallas kernels since {TEST_WITH_PALLAS=}")

    if not platforms_available("tpu"):
      self.skipTest("Splash attention requires a TPU.")

    import jax.experimental.pallas.ops.tpu.splash_attention as splash

    NUM_SAMPLES = 16
    tune_jax.logger.setLevel("DEBUG")
    tune_jax.CONFIG.find_optimal_layouts_automatically = False
    try:
      tune_jax.CONFIG.allow_fallback_timing = False

      dtype = jnp.bfloat16
      bs = 1
      q_heads = 32
      kv_heads = 8
      q_seq_len = 4096
      kv_seq_len = 4096
      qk_head_dim = 192
      v_head_dim = 128
      assert q_heads % kv_heads == 0
      init_fn = lambda shape, dtype=dtype: jnp.ones(shape, dtype)
      q = init_fn((bs, q_heads, q_seq_len, qk_head_dim), dtype=dtype)
      k = init_fn((bs, kv_heads, kv_seq_len, qk_head_dim), dtype=dtype)
      v = init_fn((bs, kv_heads, kv_seq_len, v_head_dim), dtype=dtype)

      mask = splash.MultiHeadMask([splash.CausalMask((q.shape[-2], k.shape[-2])) for _ in range(kv_heads)])

      # tune forward only ##########################################################
      tile_sizes = [512, 1024]
      hyperparams = dict(
        block_q=tile_sizes,
        block_kv=tile_sizes,
        block_kv_compute=tile_sizes,
      )

      def splash_fwd(q, k, v, **kw):
        block_sizes = splash.BlockSizes(**kw)
        attn_fn = splash.make_splash_mqa_single_device(mask, block_sizes=block_sizes)
        attn_fn = jax.vmap(attn_fn, in_axes=(0, 0, 0))  # mqa
        attn_fn = jax.vmap(attn_fn, in_axes=(0, 0, 0))  # batch
        attn_fn_ = lambda q, k, v: attn_fn(q.reshape(q.shape[:1] + (k.shape[-3], -1) + q.shape[-2:]), k, v).reshape(
          q.shape[:-1] + v.shape[-1:]
        )
        return attn_fn_(q, k, v)

      example_hyperparams = dict(block_q=1024, block_kv=1024, block_kv_compute=1024)
      formats = (
        jax.jit(partial(splash_fwd, **example_hyperparams), in_shardings=Format(Layout.AUTO))
        .lower(q, k, v)
        .compile()
        .input_formats[0]
      )
      q, k, v = jax.device_put((q, k, v), formats)

      splash_fwd = tune_jax.tune(splash_fwd, hyperparams=hyperparams, sample_num=NUM_SAMPLES)

      print("Splash fwd")
      splash_fwd(q, k, v)
      print(tune_jax.tabulate(splash_fwd))

      # tune combined forward and backward #########################################
      tile_sizes = [512, 1024]
      hyperparams = dict(
        block_q=tile_sizes,
        block_kv=tile_sizes,
        block_kv_compute=tile_sizes,
        block_q_dkv=tile_sizes,
        block_kv_dkv=tile_sizes,
        block_kv_dkv_compute=tile_sizes,
        use_fused_bwd_kernel=True,
      )

      @partial(tune_jax.tune, hyperparams=hyperparams, sample_num=NUM_SAMPLES)
      def splash_combined(q, k, v, **kw):
        block_sizes = splash.BlockSizes(**kw)
        attn_fn = splash.make_splash_mqa_single_device(mask, block_sizes=block_sizes)
        attn_fn = jax.vmap(attn_fn, in_axes=(0, 0, 0))  # mqa
        attn_fn = jax.vmap(attn_fn, in_axes=(0, 0, 0))  # batch
        attn_fn_ = lambda q, k, v: attn_fn(q.reshape(q.shape[:1] + (k.shape[-3], -1) + q.shape[-2:]), k, v).reshape(
          q.shape[:-1] + v.shape[-1:]
        )
        o, bwd_fn = jax.vjp(partial(attn_fn_), q, k, v)
        return (o, *bwd_fn(o)[:3])

      # tune combined ##############################################################
      print("Splash combined")
      splash_combined(q, k, v)
      print(tune_jax.tabulate(splash_combined))
    finally:
      tune_jax.CONFIG.allow_fallback_timing = True


class SimpleCasesTest(absltest.TestCase):
  def test_matmul_size(self):
    hyperparams = {
      "n": [64, 2048],
      "m": [64, 2048],
    }

    def fn(n, m):
      keys = random.split(random.key(0), 2)
      # A = random.normal(keys[0], (n, m), dtype=jnp.float32)
      # B = random.normal(keys[1], (m, n), dtype=jnp.float32)
      A = jnp.ones((n, m), dtype=jnp.float32)
      B = jnp.ones((m, n), dtype=jnp.float32)
      C = A @ B
      return C / jnp.linalg.norm(C, axis=-1)[..., None]

    tuned_mha = tune_jax.tune(fn, hyperparams=hyperparams)
    tuned_mha_jit = jax.jit(tuned_mha)

    _ = tuned_mha_jit().block_until_ready()
    C = tuned_mha_jit().block_until_ready()
    assert C.shape[-1] == min(hyperparams["n"])

  def test_simple_tpu_matmul(self):
    if not TEST_WITH_PALLAS:
      self.skipTest(f"Skipping pallas kernels since {TEST_WITH_PALLAS=}")
    if not platforms_available("tpu"):
      self.skipTest("No TPU available")

    from tests import tpu_matmul

    hyperparams = {
      "block_m": [256, 512],
      "block_n": [256, 512],
      "block_k": [256, 512],
    }

    x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
    y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)

    def matmul_fn(x, y, *, block_m=128, block_n=128, block_k=128):
      return tpu_matmul.matmul(x, y, block_shape=(block_m, block_n), block_k=block_k)

    tuned_matmul = tune_jax.tune(
      jax.jit(matmul_fn, static_argnames=("block_m", "block_n", "block_k")),
      hyperparams=hyperparams,
    )
    tuned_matmul_jit = jax.jit(tuned_matmul)

    tuned_matmul_jit(x, y).block_until_ready()
    print(tuned_matmul_jit.timing_results)

  def test_default_device_resolution(self):
    out = jax.jit(tune_jax.tune(lambda x: x))(1)
    with jax.default_device("cpu"):  # test that the default device resolves with a string spec "cpu"
      out = jax.jit(tune_jax.tune(lambda x: x))(1)
      self.assertEqual(list(out.devices())[0].platform.lower(), "cpu")
    with jax.default_device(jax.devices("cpu")[0]):  # test that the default device resolves with a full spec
      out = jax.jit(tune_jax.tune(lambda x: x))(1)
      self.assertEqual(list(out.devices())[0].platform.lower(), "cpu")

  def test_multidevice_tpu_matmul(self):
    if not TEST_WITH_PALLAS:
      self.skipTest(f"Skipping pallas kernels since {TEST_WITH_PALLAS=}")
    if not platforms_available("tpu"):
      self.skipTest("No TPU available")

    if len(jax.devices()) < 2:
      return

    from tests import tpu_matmul

    hyperparams = {
      "block_m": [256, 512],
      "block_n": [256, 512],
      "block_k": [256, 512],
    }

    mesh = jax.make_mesh((jax.device_count(),), ("x",))

    with jax.set_mesh(mesh):
      x = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16, out_sharding=P(None, "x"))
      y = random.normal(random.key(2), (1024, 1024), dtype=jnp.bfloat16, out_sharding=P("x", None))

    with jax.set_mesh(mesh):
      def matmul_fn(x, y, *, block_m=128, block_n=128, block_k=128):
        def inner(x, y):
          out = tpu_matmul.matmul(x, y, block_shape=(block_m, block_n), block_k=block_k)
          return jax.lax.psum(out, axis_name="x")
        return jax.shard_map(inner, out_specs=P(), check_vma=False)(x, y)
      tuned_matmul = tune_jax.tune(
        matmul_fn, hyperparams=hyperparams, example_args=(x, y),
      )
      tuned_matmul_jit = jax.jit(tuned_matmul)
      tuned_matmul_jit(x, y).block_until_ready()
    print(tuned_matmul_jit.timing_results)


if __name__ == "__main__":
  absltest.main()
