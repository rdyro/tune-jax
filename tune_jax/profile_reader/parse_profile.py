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

__all__ = ["parse_profile_from_bytes", "find_device_plane_ids", "get_events_from_plane"]

# profiler event times can be slightly inaccurate
# when looking whether an event is contained entirely under another event (a child), use a relaxation tolerance
EVENT_CHILD_TOLERANCE_PS = 2000  # 2 ns


def _get_stat_value(stat, stat_metadata):
  kind = stat.WhichOneof("value")
  if kind == "ref_value":
    return stat_metadata[stat.ref_value].name if stat.ref_value in stat_metadata else None
  return getattr(stat, kind) if kind is not None else None


def _parse_stats(stats, stat_metadata):
  if stat_metadata is not None:
    stats = [
      (stat_metadata[s.metadata_id].name if s.metadata_id in stat_metadata else None, _get_stat_value(s, stat_metadata))
      for s in stats
    ]
  return {k: v for k, v in stats if k is not None}  # ProfileData yields (None, None) for unresolved stats


def _parse_event(
  event, event_metadata, stat_metadata, line_name: str = "", line_timestamp_ns: int = 0,
  include_scope_range_id: bool = False,
):
  name = event_metadata[event.metadata_id].name if event_metadata is not None else event.name
  stats = _parse_stats(event.stats, stat_metadata)
  name = stats.get("hlo_module", name)  # hlo_module is GPU, name is TPU
  program_id = stats.get("program_id", stats.get("run_id"))  # program_id is GPU, run_id is TPU
  if include_scope_range_id:
    key = f"{name}({program_id}-{stats.get('scope_range_id', 'None')})"
  else:
    key = f"{name}({program_id})"
  if hasattr(event, "duration_ps"):  # raw XEvent proto, offsets are relative to the line timestamp
    start_ps, duration_ps = int(line_timestamp_ns) * 1000 + int(event.offset_ps), int(event.duration_ps)
  else:
    start_ps, duration_ps = round(event.start_ns * 1000), round(event.duration_ns * 1000)
  timing = dict(start_ps=start_ps, end_ps=start_ps + duration_ps, duration_ps=duration_ps)
  return {**stats, "unified_name": key, "fusion": name, "line_name": line_name, **timing}


def _parse_xspace_proto(profile_bytes: bytes):
  try:
    from tune_jax.profile_reader import xplane_pb2
  except Exception as e:
    raise ValueError("Importing the profiler proto failed. Do you have the latest protobuf?") from e
  p = xplane_pb2.XSpace()
  p.ParseFromString(profile_bytes)
  return p


def parse_profile_from_bytes(profile_bytes: bytes) -> ProfileData:
  if ProfileData is not None:
    return ProfileData.from_serialized_xspace(profile_bytes)
  return _parse_xspace_proto(profile_bytes)


def find_device_plane_ids(p: XSpace, device_str: str) -> list[int]:
  return [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]


def _find_children(own_name: str, start_ps: int, end_ps: int, events_sorted: list[dict[str, Any]], starts: np.ndarray):
  """Find all events that are fully subsumed by the `start_ps` - `end_ps` range, `starts` are sorted event starts."""
  t0, t1 = start_ps - EVENT_CHILD_TOLERANCE_PS, end_ps + EVENT_CHILD_TOLERANCE_PS
  lo, hi = np.searchsorted(starts, t0, side="left"), np.searchsorted(starts, t1, side="right")
  return [e for e in events_sorted[lo:hi] if e["end_ps"] <= t1 and e["unified_name"] != own_name]


def _sum_events(events):
  """Sum the time of all events as right extreme - left extreme subtracting empty space."""
  if not events:
    return 0
  starts, ends = np.array([e["start_ps"] for e in events]), np.array([e["end_ps"] for e in events])
  times = np.concatenate([starts, ends])
  counts = np.concatenate([np.ones_like(starts), -np.ones_like(ends)])
  order = np.argsort(times)
  active = np.cumsum(counts[order])
  return int(np.sum(np.diff(times[order]) * (active[:-1] > 0)))


def get_events_from_plane(
  p, plane_idx, prefix_filter: str = "", event_filter_regex: str | None = None
) -> dict[str, float]:
  """Returns a dict of xla module names (for unique inputs) to a list of their execution time in seconds."""

  planes = list(p.planes)
  timed_events = {}
  if hasattr(planes[plane_idx], "event_metadata"):
    event_metadata, stat_metadata = planes[plane_idx].event_metadata, planes[plane_idx].stat_metadata
  else:
    event_metadata, stat_metadata = None, None

  all_parsed_events = []
  for line in planes[plane_idx].lines:
    line_ts = getattr(line, "timestamp_ns", 0)
    parsed_events = [
      _parse_event(event, event_metadata, stat_metadata, line_name=line.name, line_timestamp_ns=line_ts)
      for event in line.events
    ]
    all_parsed_events.extend(parsed_events)

  sorted_events = sorted(all_parsed_events, key=lambda x: x["start_ps"])
  del all_parsed_events

  # on GPU we need to sum multiple scopes belonging to the same event based on the name and program id
  # NOTE: this assumes the program is called only once in the trace
  if "gpu" in planes[plane_idx].name.lower():
    # a module execution is contiguous on the device, a module interleaved with another module was executed repeatedly
    # repeated executions get distinct names (`name[1]`, `name[2]`, ...), so the caller can detect them
    run_idx, prev_name = defaultdict(lambda: -1), None
    for event in (e for e in sorted_events if e["unified_name"].startswith(prefix_filter)):
      name = event["unified_name"]
      run_idx[name], prev_name = run_idx[name] + (name != prev_name), name
      event["unified_name"] = name if run_idx[name] == 0 else f"{name}[{run_idx[name]}]"
    # the events will have the same names, we want to group them and create a fake parent event
    # later logic will aggregate the children events under this fake parent
    grouped_events = defaultdict(lambda: [])
    for event in sorted_events:
      grouped_events[event["unified_name"]].append(event)
    for unified_name, events in grouped_events.items():
      if len(events) > 1:
        start_ps, end_ps = min(event["start_ps"] for event in events), max(event["end_ps"] for event in events)
        sorted_events.append(dict(unified_name=unified_name, start_ps=start_ps, end_ps=end_ps))
        for event in events:
          event["unified_name"] = f"CHILD-{event['unified_name']}"
    sorted_events = sorted(sorted_events, key=lambda x: x["start_ps"])  # resort events since we append parents now

  filtered_events, starts = [], np.array([event["start_ps"] for event in sorted_events])
  for event in sorted_events:
    if event["unified_name"].startswith(prefix_filter):
      event["children"] = _find_children(event["unified_name"], event["start_ps"], event["end_ps"], sorted_events, starts)
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
