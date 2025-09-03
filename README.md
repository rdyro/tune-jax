# `tune-jax`

Compile-time runtime hyperparameter tuning for JAX functions (e.g., kernels).

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

import tune_jax
from tune_jax import tune, tune_logger

tune_logger.setLevel("INFO")

hyperparams = {
  "block_q": [4, 8, 16, 32, 64, 128],
  "block_k": [4, 8, 16, 32, 64, 128],
}

b, qt, h, d = 8, 32, 8, 512
kt = 128

q = random.normal(random.key(0), (b, qt, h, d), dtype=jnp.bfloat16)
k = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)
v = random.normal(random.key(0), (b, kt, h, d), dtype=jnp.bfloat16)

attention_wrapper = lambda *args, block_q=None, block_k=None, **kw: attention.mha(
  *args,
  **dict(kw, block_sizes=attention.BlockSizes(block_q=block_q, block_k=block_k)),
)

tuned_mha = tune(attention_wrapper, hyperparams=hyperparams)
tuned_mha_jit = jax.jit(tuned_mha)

tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
# no retuning on second call
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()

q = random.normal(random.key(0), (2 * b, qt, h, d), dtype=jnp.bfloat16)
k = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
v = random.normal(random.key(0), (2 * b, kt, h, d), dtype=jnp.bfloat16)
# retuning because data shape changed
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()
# no retuning on second call
tuned_mha_jit(q, k, v, segment_ids=None).block_until_ready()  

print(tuned_mha_jit.timing_results)  # to get access to latest timing results

print(tune_jax.tabulate(tuned_mha_jit.timing_results))  # to print nicely
# print(tune_jax.tabulate(tuned_mha_jit))  # to rely on attribute extraction

print(tuned_mha_jit.optimal_hyperparams)
```

```
  id    block_q    block_k    t_mean (s)    t_std (s)
----  ---------  ---------  ------------  -----------
  23         32        128    5.2874e-05   1.1751e-06
  35        128        128    5.4357e-05   2.1156e-07
  29         64        128    5.6255e-05   3.1701e-06
  17         16        128    5.8837e-05   6.744e-07
  16         16         64    7.728e-05    1.161e-06
  11          8        128    7.8282e-05   4.256e-07
  27         64         32    8.4714e-05   3.7561e-07
  21         32         32    8.5045e-05   1.4363e-07
  33        128         32    8.5578e-05   9.8937e-07
  15         16         32    0.00010546   1.5085e-08
  10          8         64    0.00011382   7.1118e-07
  26         64         16    0.00013777   1.3057e-06
   5          4        128    0.00013904   2.6428e-07
  20         32         16    0.0001392    4.1055e-07
  32        128         16    0.00014012   2.7692e-07
   9          8         32    0.000158     2.8738e-07
  14         16         16    0.000195     3.935e-07
   4          4         64    0.00021071   7.7064e-07
   3          4         32    0.00025267   9.3753e-08
   8          8         16    0.00026097   9.1332e-08
   2          4         16    0.00042573   7.2384e-07
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
  store_timing_results: bool = True,
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
      store_timing_results (bool): Attach the timining results to the function handle (i.e., fn.timing_results)?
  """

  ...
```
