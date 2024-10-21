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

## API

```python
def tune(
  fn_to_tune: Callable[..., Any],
  hyperparams: dict[Any, Any],
  max_workers: int = 32,
  in_shardings: Any = UNSPECIFIED,
  out_shardings: Any = UNSPECIFIED,
  device: jax.Device | _UnspecifiedT = UNSPECIFIED,
  example_args: tuple[Any] | None = None,
  example_kws: dict[Any, Any] | None = None,
):
  """Tune a function with hyperparameters, even if some fail to compile.

  Args:
      fn_to_tune (Callable[..., Any]): A jax function to tune.
      hyperparams (dict[Any, Any]): A flat dictionary of hyperparameter lists.
      max_workers (int, optional): Max workers for parallel compilation.
      in_shardings (Any, optional): in_shardings for timing (see jax.jit).
      out_shardings (Any, optional): out_shardings for timing (see jax.jit).
      device (jax.Device | _UnspecifiedT, optional): device to tune on if shardings are unspecified.
      example_args (tuple[Any] | None, optional): Exact example_args to tune with, on correct device.
      example_kws (dict[Any, Any] | None, optional): Exact example_kws to tune with, on correct device.
  """

  ...
```
