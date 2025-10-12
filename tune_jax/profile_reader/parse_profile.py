from typing import Any
from pathlib import Path
import re
from pprint import pprint
from collections import defaultdict

import numpy as np

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

# profiler event times can be slightly inaccurate
# when looking whether an event is contained entirely under another event (a child), use a relaxation tolerance
EVENT_CHILD_TOLERANCE_PS = 2000  # 2 ns


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


def _parse_event(event, event_metadata, stat_metadata, prefix_filter: str = "", line_name: str = ""):
  if event_metadata is not None:
    name = event_metadata[event.metadata_id].name
  else:
    name = event.name
  stats = _parse_stats(event.stats, stat_metadata)
  name = stats.get("hlo_module", name)  # hlo_module is GPU, name is TPU
  # if not name.startswith(prefix_filter):
  #  return None
  program_id = stats.get("program_id", stats.get("run_id"))  # program_id is GPU, run_id is TPU
  scope_range_id = stats.get("scope_range_id", "None")
  key = f"{name}({program_id}-{scope_range_id})"
  if hasattr(event, "duration_ps"):
    stats["start_ps"] = int(event.offset_ps)
    stats["end_ps"] = int(event.offset_ps) + int(event.duration_ps)
    stats["duration_ps"] = int(event.duration_ps)
  else:
    stats["start_ps"] = int(event.start_ns * 1000)
    stats["end_ps"] = int(event.start_ns * 1000) + int(event.duration_ns * 1000)
    stats["duration_ps"] = int(event.duration_ns * 1000)
  return dict(unified_name=key, fusion=name, line_name=line_name, **stats)


def parse_profile_from_bytes(profile_bytes: bytes) -> ProfileData:
  if ProfileData is not None:
    return ProfileData.from_serialized_xspace(profile_bytes)
  p = xplane_pb2.XSpace()
  p.ParseFromString(profile_bytes)
  return p


def find_device_plane_ids(p: XSpace, device_str: str) -> list[int]:
  return [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]


def _find_children(own_name: str, start_ps: int, end_ps: int, events_sorted: list[dict[str, Any]]):
  """Find all events that are fully subsumed by the `start_ps` - `end_ps` range."""
  t0 = start_ps - EVENT_CHILD_TOLERANCE_PS - 1
  idx = np.searchsorted(np.sort(np.array([event["start_ps"] for event in events_sorted])), t0)
  children = []
  while idx < len(events_sorted) and events_sorted[idx]["start_ps"] <= end_ps + EVENT_CHILD_TOLERANCE_PS:
    ts, te = events_sorted[idx]["start_ps"], events_sorted[idx]["end_ps"]
    is_contained = ts >= start_ps - EVENT_CHILD_TOLERANCE_PS and te <= end_ps + EVENT_CHILD_TOLERANCE_PS
    if is_contained and events_sorted[idx]["unified_name"] != own_name:
      children.append(events_sorted[idx])
    idx += 1
  return children


def _sum_events(events):
  """Sum the time of all events as right extreme - left extreme subtracting empty space."""
  if len(events) == 0:
    return 0
  if len(events) == 1:
    return events[0]["end_ps"] - events[0]["start_ps"]
  starts, ends = np.array([e["start_ps"] for e in events]), np.array([e["end_ps"] for e in events])
  min_start, max_end = int(np.min(starts)), int(np.max(ends))
  sorted_ends = np.sort(ends)
  empty_ends = np.where(
    ~np.any((sorted_ends[None, :-1] < ends[:, None]) & (sorted_ends[None, :-1] >= starts[:, None]), axis=0)
  )[0]
  sorted_starts = np.sort(starts)
  empty_space = sum(
    int(ends[end_idx]) - int(sorted_starts[np.searchsorted(sorted_starts, ends[end_idx])]) for end_idx in empty_ends
  )
  assert empty_space < (max_end - min_start)
  return max_end - min_start - empty_space


def get_events_from_plane(
  p, plane_idx, prefix_filter: str = "", event_filter_regex: str | None = None
) -> dict[str, list[float]]:
  """Returns a dict of xla module names (for unique inputs) to a list of their execution time in seconds."""

  planes = list(p.planes)
  timed_events = {}
  if hasattr(planes[plane_idx], "event_metadata"):
    event_metadata, stat_metadata = planes[plane_idx].event_metadata, planes[plane_idx].stat_metadata
  else:
    event_metadata, stat_metadata = None, None
  all_parsed_events = []
  for line in planes[plane_idx].lines:
    parsed_events = [
      _parse_event(event, event_metadata, stat_metadata, prefix_filter, line_name=line.name) for event in line.events
    ]
    parsed_events = [event for event in parsed_events if event is not None]
    all_parsed_events.extend(parsed_events)

  sorted_events = sorted(all_parsed_events, key=lambda x: x["start_ps"])

  filtered_events = []
  for event in all_parsed_events:
    if event["unified_name"].startswith(prefix_filter):
      event["children"] = _find_children(event["unified_name"], event["start_ps"], event["end_ps"], sorted_events)
      if event_filter_regex is not None:
        # an alternative timing method, look for children based on the regex pattern
        # and sum all children events times subtracting empty space: len(|---|    |-||--|) = 6
        new_children = [ch for ch in event["children"] if re.search(event_filter_regex, ch["unified_name"]) is not None]
        event["children"] = new_children
        event["children_duration"] = _sum_events(new_children)
      filtered_events.append(event)
  timed_events = {
    event["unified_name"]: event.get("children_duration", (event["end_ps"] - event["start_ps"])) / 1e12
    for event in filtered_events
  }

  # on GPU we need to sum multiple scopes belonging to the same event based on the name and program id
  # NOTE: this assumes the program is called only once in the trace
  if "gpu" in planes[plane_idx].name.lower():
    combined_timed_events = defaultdict(lambda: 0.0)
    to_delete = []
    program_regex = r"(.*?)\(([0-9]+)-[0-9]+\)"
    for unified_name, duration in timed_events.items():
      if (m := re.match(program_regex, unified_name)) is not None:
        program_str, program_id = m[1], m[2]
        combined_timed_events[f"{program_str}({program_id})"] += duration
        to_delete.append(unified_name)
    for unified_name in to_delete:
      del timed_events[unified_name]
    timed_events |= combined_timed_events

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
