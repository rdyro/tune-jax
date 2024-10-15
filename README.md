# `tune-jax`

This package provides a `tune` decorator for microbenchmarking and tuning JAX functions, particularly Pallas kernels.

## Installation

```bash
pip install tune-jax
```

## Usage

```python
import functools
from tune_jax import tune

@functools.partial(tune, hyperparams={'block_q': [256, 512, 1024], 'block_k': [8, 16]})
def my_pallas_function(...):
  ...
```

This will benchmark `my_pallas_function` across all combinations of `block_q` and `block_k`, automatically handling any compilation failures. 

See the docstring of the `tune` function for details on all available options.

## Example: Tuning Attention on GPU

```python
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.pallas.ops.gpu import attention

from tune_jax import tune, tune_logger

tune_logger.setLevel("DEBUG")

hyperparams = {
"block_q": [4, 8, 16, 32, 64, 128],
"block_k": [4, 8, 16, 32, 64, 128],
}

b, qt, h, d = 8, 32, 8, 512
kt = 128

q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

tuned_mha = tune(attention.mha, hyperparams=hyperparams)
tuned_mha_jit = jax.jit(tuned_mha)

tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
```