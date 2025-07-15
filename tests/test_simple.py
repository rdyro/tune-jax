import functools

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.pallas.ops.gpu import attention
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import pytest

from tune_jax import tune, tune_logger

tune_logger.setLevel("DEBUG")


def platforms_available(*platforms):
    for platform in platforms:
        try:
            jax.devices(platform)
            return True
        except:
            pass
    return False


def test_matmul_size():
    hyperparams = {
        "n": [64, 4096],
        "m": [64, 4096],
    }

    def fn(n, m):
        keys = random.split(random.key(0), 2)
        A = random.normal(keys[0], (n, m), dtype=jnp.float32)
        B = random.normal(keys[1], (m, n), dtype=jnp.float32)
        C = A @ B
        return C / jnp.linalg.norm(C, axis=-1)[..., None]

    tuned_mha = tune(fn, hyperparams=hyperparams)
    tuned_mha_jit = jax.jit(tuned_mha)

    _ = tuned_mha_jit().block_until_ready()
    C = tuned_mha_jit().block_until_ready()
    assert C.shape[-1] == min(hyperparams["n"])


@pytest.mark.skipif(not platforms_available("gpu"), reason="No GPU available")
def test_simple_mha():
    hyperparams = {
        "block_q": [4, 8, 16, 32, 64, 128],
        "block_k": [4, 8, 16, 32, 64, 128],
    }

    b, qt, h, d = 8, 32, 8, 512
    kt = 128

    q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
    k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
    v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

    if hasattr(attention, "BlockSizes"):
        attention_wrapper = lambda *args, block_q, block_k, **kw: attention.mha(
            *args,
            **dict(kw, block_sizes=attention.BlockSizes(block_q=block_q, block_k=block_k)),
        )
        attention_fn = attention_wrapper
    else:  # jax < 0.5.2
        attention_fn = attention.mha
    tuned_mha = tune(jax.jit(attention_fn, static_argnames=("block_q", "block_k")), hyperparams=hyperparams)
    tuned_mha_jit = jax.jit(tuned_mha)

    tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
    tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
    q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
    k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
    v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
    tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
    tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()

    print(tuned_mha_jit.timing_results)  # to get access to latest timing results  # ty: ignore[unresolved-attribute]


@pytest.mark.skipif(not platforms_available("gpu"), reason="No GPU available")
def test_multidevice():
    hyperparams = {
        "block_q": [4, 8, 16, 32, 64, 128],
        "block_k": [4, 8, 16, 32, 64, 128],
    }

    b, qt, h, d = 8, 32, 8, 512
    kt = 128

    if len(jax.devices()) < 2:
        return

    mesh = Mesh(devices=jax.devices(), axis_names=("x",))

    q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
    k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
    v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

    in_shardings = [NamedSharding(mesh, P(*(["x"] + [None] * (z.ndim - 1)))) for z in [q, k, v]]
    q, k, v = jax.tree.map(lambda x, y: jax.device_put(x, y), [q, k, v], in_shardings)

    if hasattr(attention, "BlockSizes"):
        attention_wrapper = lambda *args, block_q, block_k, **kw: attention.mha(
            *args,
            **dict(kw, block_sizes=attention.BlockSizes(block_q=block_q, block_k=block_k)),
        )
        tuned_mha = tune(
            functools.partial(attention_wrapper, segment_ids=None), hyperparams=hyperparams, in_shardings=in_shardings
        )
    else:  # jax < 0.5.2
        tuned_mha = tune(
            functools.partial(attention.mha, segment_ids=None), hyperparams=hyperparams, in_shardings=in_shardings
        )
    tuned_mha_jit = jax.jit(tuned_mha, in_shardings=in_shardings)  # type: ignore

    tuned_mha_jit(q, k, v).block_until_ready()
    tuned_mha_jit(q, k, v).block_until_ready()
    q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
    k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
    v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
    tuned_mha_jit(q, k, v).block_until_ready()
    tuned_mha_jit(q, k, v).block_until_ready()

    print(tuned_mha_jit.timing_results)  # to get access to latest timing results  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    test_matmul_size()
    if platforms_available("gpu"):
        test_multidevice()
        test_simple_mha()
