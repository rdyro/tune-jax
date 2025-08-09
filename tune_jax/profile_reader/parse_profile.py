import tempfile
from typing import Any
from pathlib import Path
from pprint import pprint

XSpace = Any

try:
  from jax.profiler import ProfileData
except ImportError:
  ProfileData = None

if ProfileData is None:
  try:
    from tune_jax.profile_reader import xplane_pb2
  except Exception as e:
    raise ValueError("Importing the profiler proto failed. Do you have the latest protobuf?") from e
else:
  xplane_pb2 = None

__all__ = ["parse_profile_from_bytes", "find_device_plane_ids", "get_events_from_plane"]


def _get_stat_value(stat, metadata):
  if stat.ref_value != 0:
    return metadata[stat.ref_value].name
  for key in ["double", "int64", "uint64", "ref"]:
    if getattr(stat, key + "_value") != 0:
      return getattr(stat, key + "_value")
  for key in ["bytes", "str"]:
    if len(getattr(stat, key + "_value")) > 0:
      return getattr(stat, key + "_value")


def _parse_stats(stats, stat_metadata):
  if stat_metadata is not None:
    return {stat_metadata[stat.metadata_id].name: _get_stat_value(stat, stat_metadata) for stat in stats}
  return dict(stats)


def _parse_event(event, event_metadata, stat_metadata):
  if event_metadata is not None:
    name = event_metadata[event.metadata_id].name
  else:
    name = event.name
  stats = _parse_stats(event.stats, stat_metadata)
  name = stats.get("hlo_module", name)  # hlo_module is GPU, name is TPU
  program_id = stats.get("program_id", stats.get("run_id"))  # program_id is GPU, run_id is TPU
  key = f"{name}({program_id})"
  if hasattr(event, "duration_ps"):
    stats["start_ps"] = event.offset_ps
    stats["end_ps"] = event.offset_ps + event.duration_ps
    stats["duration_ps"] = event.duration_ps
  else:
    stats["start_ps"] = event.start_ns * 1e3
    stats["end_ps"] = (event.start_ns + event.duration_ns) * 1e3
    stats["duration_ps"] = event.duration_ns * 1e3
  return dict(jax_fn_name=key, fusion=name, **stats)


def parse_profile_from_bytes(profile_bytes: bytes) -> ProfileData:
  if ProfileData is not None:
    return ProfileData.from_serialized_xspace(profile_bytes)
  p = xplane_pb2.XSpace()
  p.ParseFromString(profile_bytes)
  return p


def find_device_plane_ids(p: XSpace, device_str: str) -> list[int]:
  return [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]


def get_events_from_plane(p, plane_idx, verbose: bool = False, just_id: bool = True) -> dict[str, list[float]]:
  """Returns a dict of xla module names (for unique inputs) to a list of their execution time in seconds."""

  planes = list(p.planes)
  timed_events = {}
  if hasattr(planes[plane_idx], "event_metadata"):
    event_metadata, stat_metadata = planes[plane_idx].event_metadata, planes[plane_idx].stat_metadata
  else:
    event_metadata, stat_metadata = None, None
  all_parsed_events = []
  for line in planes[plane_idx].lines:
    parsed_events = [_parse_event(event, event_metadata, stat_metadata) for event in line.events]
    all_parsed_events.extend(parsed_events)

    xla_modules = {}
    for event in parsed_events:
      key = (event["jax_fn_name"], event["scope_range_id"]) if not just_id else (event["jax_fn_name"],)
      xla_modules.setdefault(key, []).append(event)
    xla_modules = {
      k: (max([e["end_ps"] for e in event_list], default=0) - min([e["start_ps"] for e in event_list], default=0))
      / 1e12
      for k, event_list in xla_modules.items()
    }
    grouped_timings = {}
    for key, duration in xla_modules.items():
      xla_module_name = key[0]
      grouped_timings.setdefault(xla_module_name, []).append(duration)
    timed_events |= grouped_timings

  # debuggging ###################################################################################
  if verbose:
    all_keys = set()
    for e in all_parsed_events:
      all_keys |= set(e.keys())
    keys_order = list(all_parsed_events[0].keys())
    all_keys = sorted(all_keys, key=lambda k: keys_order.index(k) if k in keys_order else 1e20)

    from tabulate import tabulate

    dump = tabulate([[e.get(k, "None") for k in all_keys] for e in all_parsed_events], headers=all_keys)
    with tempfile.NamedTemporaryFile(delete=False) as f:
      Path(f.name).absolute().write_text(dump)
      print(f"Dumped parsed events to {Path(f.name).absolute()}")
  # debuggging ###################################################################################

  return timed_events


# tests
def test_main():
  all_profiles = list(Path("~/profiles").expanduser().glob("**/*.xplane.pb"))
  latest_profile = sorted(all_profiles, key=lambda x: x.stat().st_mtime)[-1]
  print(f"latest profile = {latest_profile}")
  p = parse_profile_from_bytes(latest_profile.read_bytes())
  gpu_plane_ids = find_device_plane_ids(p, "gpu")
  print([p.planes[i].name for i in gpu_plane_ids])

  events = get_events_from_plane(p, gpu_plane_ids[0])
  pprint({k: sum(v) / len(v) for k, v in events.items()})


if __name__ == "__main__":
  test_main()
