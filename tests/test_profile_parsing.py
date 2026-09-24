import json
import os
import re
import tempfile
from pathlib import Path

import jax
from jax import numpy as jnp
from jax import random
from absl.testing import absltest

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


def xplane_pb2_or_none():
  """The vendored `xplane_pb2` gencode cannot be loaded against every protobuf runtime."""
  try:
    from tune_jax.profile_reader import xplane_pb2

    return xplane_pb2
  except Exception:  # noqa: BLE001
    return None


# a collection of functions to tune ################################################################


def _matmul(x, y, dummy):
  del dummy
  return x @ y


def _cond_fn(c, fn1, fn2, *args):
  return jax.lax.cond(c, lambda: fn1(*args), lambda: fn2(*args))


def _tpu_matmul(x, y, block_m, block_n, block_k):
  from tests import tpu_matmul

  return tpu_matmul.matmul(x, y, block_shape=(block_m, block_n), block_k=block_k)


def _long_while(it, x, y):
  return jax.lax.fori_loop(0, it, lambda i, carry: (x @ y) / jnp.linalg.norm(carry + x @ y), (x @ y))


####################################################################################################


class ProfileReadingTest(absltest.TestCase):
  def test_parsing_multiple_profiles(self):
    if not TEST_WITH_PALLAS:
      self.skipTest(f"Skipping pallas kernels since {TEST_WITH_PALLAS=}")

    if platforms_available("tpu"):
      try:
        tune_jax.CONFIG.allow_fallback_timing = False
        hyperparams = {
          "block_m": [256, 512],
          "block_n": [256, 512],
          "block_k": [256, 512],
        }
        x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
        y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)
        tune_jax.tune(_tpu_matmul, hyperparams=hyperparams)(x, y).block_until_ready()
        jax.jit(tune_jax.tune(_tpu_matmul, hyperparams=hyperparams))(x, y).block_until_ready()
      finally:
        tune_jax.CONFIG.allow_fallback_timing = True

    try:
      tune_jax.CONFIG.allow_fallback_timing = False
      x = random.normal(random.key(0), (1024, 1024), dtype=jnp.bfloat16)
      y = random.normal(random.key(1), (1024, 1024), dtype=jnp.bfloat16)

      tune_jax.tune(_matmul, hyperparams={"dummy": [1]})(x, y)
      jax.jit(tune_jax.tune(_matmul, hyperparams={"dummy": [1]}))(x, y)

      _fn = lambda x, y, c: _cond_fn(c, lambda x, y: x @ y, lambda x, y: x + y, x, y)
      tune_jax.tune(_fn, hyperparams={"c": [0, 1, 2]})(x, y)
      jax.jit(tune_jax.tune(_fn, hyperparams={"c": [0, 1, 2]}))(x, y)

      _fn = lambda x, y, it: _long_while(it, x, y)
      tune_jax.tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]})(x, y)
      jax.jit(tune_jax.tune(_fn, hyperparams={"it": [0, 1, 2, 3, 4, 5, 6, 7]}))(x, y)
    finally:
      tune_jax.CONFIG.allow_fallback_timing = True

  def test_parse_paths_agree(self):
    from tune_jax.profile_reader import parse_profile

    if (xplane_pb2 := xplane_pb2_or_none()) is None:
      self.skipTest("The vendored xplane_pb2 gencode is not loadable with the installed protobuf runtime.")

    xs = xplane_pb2.XSpace()
    plane = xs.planes.add(name="/device:GPU:0")
    plane.event_metadata[1].name = "kernel"
    names = {1: "hlo_module", 2: "program_id", 3: "unified_name", 4: "zero_stat", 5: "jit_tune_jax_fn_0"}
    for i, name in names.items():
      plane.stat_metadata[i].name = name
    for timestamp_ns in [1000, 2000]:  # two lines with different base timestamps
      event = plane.lines.add(name=f"Stream #{timestamp_ns}", timestamp_ns=timestamp_ns).events.add(
        metadata_id=1, offset_ps=0, duration_ps=500_000
      )
      event.stats.add(metadata_id=1, ref_value=5)
      event.stats.add(metadata_id=2, int64_value=7)
      event.stats.add(metadata_id=3, str_value="colliding stat name")
      event.stats.add(metadata_id=4, int64_value=0)
      event.stats.add(metadata_id=99, int64_value=1)  # unresolvable stat metadata
    profile_bytes = xs.SerializeToString()

    parsers = [parse_profile._parse_xspace_proto, parse_profile.parse_profile_from_bytes]
    for parse in parsers:
      p = parse(profile_bytes)
      plane_idx = parse_profile.find_device_plane_ids(p, "gpu")[0]
      events = parse_profile.get_events_from_plane(p, plane_idx, prefix_filter="jit_")
      self.assertEqual(list(events.keys()), ["jit_tune_jax_fn_0(7)"])
      self.assertAlmostEqual(events["jit_tune_jax_fn_0(7)"], 1.5e-6)
    stats = parse_profile._parse_stats(xs.planes[0].lines[0].events[0].stats, xs.planes[0].stat_metadata)
    self.assertEqual(stats["zero_stat"], 0)

  def test_repeated_module_gets_distinct_key(self):
    from tune_jax.profile_reader import parse_profile

    if (xplane_pb2 := xplane_pb2_or_none()) is None:
      self.skipTest("The vendored xplane_pb2 gencode is not loadable with the installed protobuf runtime.")

    xs = xplane_pb2.XSpace()
    plane = xs.planes.add(name="/device:GPU:0")
    plane.stat_metadata[1].name, plane.stat_metadata[2].name = "hlo_module", "program_id"
    line = plane.lines.add(name="Stream #1", timestamp_ns=0)
    for i, fn_idx in enumerate([0, 0, 1, 0]):  # f0 f1 f0 with two kernels in the first f0 execution
      plane.event_metadata[i + 1].name = f"kernel_{i}"
      event = line.events.add(metadata_id=i + 1, offset_ps=i * 1_000_000, duration_ps=500_000)
      event.stats.add(metadata_id=1, str_value=f"jit_tune_jax_fn_{fn_idx}")
      event.stats.add(metadata_id=2, int64_value=fn_idx)
    p = parse_profile._parse_xspace_proto(xs.SerializeToString())
    events = parse_profile.get_events_from_plane(p, 0, prefix_filter="jit_")
    self.assertEqual(sorted(events), ["jit_tune_jax_fn_0(0)", "jit_tune_jax_fn_0(0)[1]", "jit_tune_jax_fn_1(1)"])
    self.assertAlmostEqual(events["jit_tune_jax_fn_0(0)"], 1.5e-6)

  def test_repeated_execution_raises(self):
    if not platforms_available("gpu", "tpu"):
      self.skipTest("Profiler device timing requires a GPU or a TPU.")
    platform = jax.devices()[0].platform

    def make(i):
      fn = lambda x: jnp.tanh(x @ x)
      fn.__name__ = fn.__qualname__ = tune_jax.tuning.TUNE_FN_PREFIX_FMT.format(i)
      return jax.jit(fn)

    f0, f1, x = make(0), make(1), jnp.ones((1024, 1024))
    closure = lambda: [f(x).block_until_ready() for f in [f0, f1, f0]]
    closure()
    with self.assertRaisesRegex(RuntimeError, "more than once"):
      tune_jax.tuning._time_with_profiler(closure, platform, 1)

  def test_profile_files_cleanup(self):
    fn = lambda x, y, dummy: x @ y
    x = jnp.ones((128, 128))
    keep_default, tempdir_default = tune_jax.CONFIG.keep_profile_files, tempfile.tempdir
    try:
      for keep in [False, True]:
        with tempfile.TemporaryDirectory(dir=tempdir_default) as root:
          tune_jax.CONFIG.keep_profile_files, tempfile.tempdir = keep, root
          tune_jax.tune(fn, hyperparams={"dummy": [1, 2]})(x, x)
          profile_dirs = list(Path(root).glob("tuning_profile_*"))
          self.assertEqual(len(profile_dirs), tune_jax.CONFIG.profiling_samples if keep else 0)
    finally:
      tune_jax.CONFIG.keep_profile_files, tempfile.tempdir = keep_default, tempdir_default

  def test_tpu_splash_traces(self):
    from tune_jax.profile_reader import parse_profile

    # tuning profiles captured by `tune` on a TPU v5e (jax 0.11.2) for splash attention fwd and fwd + bwd
    data = Path(__file__).parent / "data"
    summary = json.loads((data / "splash_summary.json").read_text())
    for name, info in summary.items():
      p = parse_profile.parse_profile_from_bytes((data / f"{name}.xplane.pb").read_bytes())
      plane_ids = parse_profile.find_device_plane_ids(p, "tpu")
      self.assertEqual([list(p.planes)[i].name for i in plane_ids], ["/device:TPU:0"])
      events = parse_profile.get_events_from_plane(p, plane_ids[0], prefix_filter="jit_")
      fn_times = {int(m[1]): t for k, t in events.items() if (m := re.match(r"jit_tune_jax_fn_([0-9]+)\(", k))}
      expected = {int(i): r["t_mean"] for i, r in info["timing_results"].items()}
      self.assertEqual(sorted(fn_times), sorted(expected))
      for i, t_mean in expected.items():  # a single trace vs the mean over all profiling samples
        self.assertAlmostEqual(fn_times[i] / t_mean, 1.0, delta=0.01)

  def test_sum_events(self):
    from tune_jax.profile_reader.parse_profile import _sum_events

    events = [
      {"start_ps": 0, "end_ps": 10},
      {"start_ps": 5, "end_ps": 15},
      {"start_ps": 15, "end_ps": 20},
      {"start_ps": 25, "end_ps": 30},
      {"start_ps": 25, "end_ps": 30},
    ]
    self.assertEqual(_sum_events(events), 25)


if __name__ == "__main__":
  absltest.main()
