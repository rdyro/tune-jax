import sys
from typing import Any
from pathlib import Path
from pprint import pprint, pformat
import dataclasses

try:
  from tune_jax.profile_reader import xplane_pb2
except ImportError:
  here_path = str(Path(__file__).parent.absolute())
  if here_path not in sys.path:
    sys.path.append(here_path)

  import xplane_pb2

__all__ = [
  "parse_profile_from_bytes",
  "find_device_plane_ids",
  "get_events_from_plane",
]


def parse_profile_from_bytes(profile_bytes: bytes) -> xplane_pb2.XSpace:
  p = xplane_pb2.XSpace()
  p.ParseFromString(profile_bytes)
  return p


def find_device_plane_ids(p: xplane_pb2.XSpace, device_str: str) -> list[int]:
  return [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]


@dataclasses.dataclass(frozen=True)
class ParsedEvent:
  name: str
  scopes: str
  split_scopes: list[str | None]
  hlo_op: str
  hlo_module: str
  start: float  # in seconds
  end: float  # in seconds
  duration: float  # in seconds
  correlation_id: int
  scope_range_id: int
  program_id: int
  raw_event: xplane_pb2.XEvent
  stats: dict[str, Any] = dataclasses.field(default_factory=lambda: {})


def _extract_stat(plane: xplane_pb2.XPlane, stats: list[xplane_pb2.XStat], name: str, attr_name: str):
  valid_stats = [stat for stat in stats if plane.stat_metadata[stat.metadata_id].name == name]
  return getattr(valid_stats[0], attr_name) if len(valid_stats) == 1 else None


def _parse_event(plane, event) -> ParsedEvent:
  event_metadata = plane.event_metadata[event.metadata_id]
  all_stats = list(event.stats) + list(event_metadata.stats)  # handle both GPU and TPU

  # extract all reference stats
  ref_stats = {
    plane.stat_metadata[stat.metadata_id].name: plane.stat_metadata[stat.ref_value].name
    for stat in all_stats
    if stat.ref_value
  }

  correlation_id = _extract_stat(plane, all_stats, "correlation_id", "uint64_value")
  scope_range_id = _extract_stat(plane, all_stats, "scope_range_id", "int64_value")
  program_id = _extract_stat(plane, all_stats, "program_id", "int64_value")

  # on TPU scopes and hlo_op live in event_metadata instead
  hlo_op_tpu_fallback = event_metadata.display_name
  scopes_tpu_fallback = _extract_stat(plane, all_stats, "tf_op", "str_value")

  scopes = ref_stats.get("name", scopes_tpu_fallback)
  scope_delim = "/"
  if scopes is not None and scope_delim in scopes:
    split_scopes = scopes.split(scope_delim)
  else:
    split_scopes = [None]

  parsed_event = ParsedEvent(
    name=event_metadata.name,
    scopes=scopes,
    split_scopes=split_scopes,
    hlo_op=ref_stats.get("hlo_op", hlo_op_tpu_fallback),
    hlo_module=ref_stats.get("hlo_module", None),
    start=event.offset_ps / 1e12,
    end=(event.offset_ps + event.duration_ps) / 1e12,
    duration=event.duration_ps / 1e12,
    correlation_id=correlation_id,
    scope_range_id=scope_range_id,
    program_id=program_id,
    raw_event=event,
    stats=ref_stats,
  )
  return parsed_event


@dataclasses.dataclass
class EventScopeNode:
  """A struct holding the information about a scope, timing always and event if it's a leaf event node."""

  name: str | None = None
  event: ParsedEvent | None = None
  start: float | None = None
  end: float | None = None
  duration: float | None = None
  children: dict[str, "EventScopeNode"] = dataclasses.field(default_factory=lambda: {})


def _insert_event_into_scope_tree(event_tree: EventScopeNode, parsed_event: ParsedEvent, keys: list[str]) -> None:
  """Insert an event in a named scope tree creating nodes if necessary."""
  if len(keys) == 0:
    return
  child_name = keys[0]
  if len(keys) == 1:
    child_name = child_name + f"-{parsed_event.hlo_op}-{parsed_event.start}-{parsed_event.correlation_id}"
  if len(keys) == 1:
    new_node = EventScopeNode(
      parsed_event.name, parsed_event, parsed_event.start, parsed_event.end, parsed_event.duration
    )
    if child_name in event_tree.children:
      msg = f"This child with {child_name=} already exists in tree {pformat(event_tree)}" + "\n"
      msg += f"Existing child: {pformat(event_tree.children[child_name])}" + "\n"
      msg += f"New child: {pformat(new_node)}"
      raise ValueError(msg)
    event_tree.children[child_name] = new_node
  else:
    event_tree.children.setdefault(child_name, EventScopeNode(child_name))
    _insert_event_into_scope_tree(event_tree.children[child_name], parsed_event, keys[1:])


def _combine_scope_children_times(event_tree: EventScopeNode) -> tuple[float, float, float]:
  """Walk the event tree and combine each node's time by taking earliest child start and latest child end."""
  if event_tree.start is not None:
    return (event_tree.start, event_tree.end, event_tree.duration)
  all_children_times = [_combine_scope_children_times(child) for child in event_tree.children.values()]
  if len(all_children_times) == 0:
    raise ValueError(
      "Event without children and already pre-defined times. This should not happen, please file an issue."
    )
  t_min = min([times[0] for times in all_children_times], default=0)
  t_max = max([times[1] for times in all_children_times], default=0)
  duration = t_max - t_min
  event_tree.start, event_tree.end, event_tree.duration = t_min, t_max, duration
  return (event_tree.start, event_tree.end, event_tree.duration)


def _is_new_scope(prev_parsed_event: ParsedEvent, parsed_event: ParsedEvent, splitter_op_id: str):
  if prev_parsed_event is None:
    return False
  new_top_level_scope = prev_parsed_event.split_scopes[0] != parsed_event.split_scopes[0]
  splitter_op_hit = f"{parsed_event.scopes}-{parsed_event.hlo_op}" == splitter_op_id
  return new_top_level_scope or splitter_op_hit


def get_events_from_plane(
  p: xplane_pb2.XSpace, plane_idx: int, verbose: bool = True
) -> dict[str, list[EventScopeNode]]:
  """Get a dictionary of top level scope events. Because they can repeat, aggregate them in a list.

  Use :func:`find_device_plane_ids` to find the plane indices of planes corresponding to an accelerator.
  """
  if verbose:
    from tqdm import tqdm
  else:
    tqdm = lambda *args, **kw: args[0]

  plane = p.planes[plane_idx]
  all_raw_events = [(line_id, event) for line_id, line in enumerate(plane.lines) for event in line.events]
  all_parsed_events = [
    (line_id, _parse_event(plane, event)) for line_id, event in tqdm(all_raw_events, desc="Parsing events")
  ]
  all_parsed_events = [
    (line_id, parsed_event) for line_id, parsed_event in all_parsed_events if parsed_event.split_scopes[0] is not None
  ]
  all_parsed_events = sorted(
    all_parsed_events, key=lambda x: x[1].correlation_id if x[1].correlation_id is not None else int(1e20)
  )

  all_events, prev_parsed_event, splitter_op_id, running_event = {}, None, None, EventScopeNode()
  for line_id, parsed_event in tqdm(all_parsed_events, desc="Aggregating scopes"):
    del line_id
    op_id = f"{parsed_event.scopes}-{parsed_event.hlo_op}"

    # potentially terminate a scope
    if _is_new_scope(prev_parsed_event, parsed_event, splitter_op_id):
      _combine_scope_children_times(running_event)
      for event_node in running_event.children.values():
        all_events.setdefault(event_node.name, []).append(event_node)
      running_event, splitter_op_id = EventScopeNode(), op_id

    # update the prev_parsed_event and splitter_op_id
    splitter_op_id = op_id if splitter_op_id is None else splitter_op_id
    prev_parsed_event = parsed_event
    _insert_event_into_scope_tree(running_event, parsed_event, parsed_event.split_scopes)

  # finally, combine the last running event and save it
  _combine_scope_children_times(running_event)
  for event_node in running_event.children.values():
    all_events.setdefault(event_node.name, []).append(event_node)
  return all_events


# tests
def test_main():
  all_profiles = list(Path("~/profiles").expanduser().glob("**/*.xplane.pb"))
  latest_profile = sorted(all_profiles, key=lambda x: x.stat().st_mtime)[-1]
  print(f"latest profile = {latest_profile}")
  p = parse_profile_from_bytes(latest_profile.read_bytes())
  gpu_plane_ids = find_device_plane_ids(p, "gpu")
  print([p.planes[i].name for i in gpu_plane_ids])

  events = get_events_from_plane(p, gpu_plane_ids[0])
  _mean = lambda x: sum(x) / len(x)
  pprint({k: _mean([x.duration for x in v]) for k, v in events.items()})


if __name__ == "__main__":
  test_main()
