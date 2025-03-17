import sys
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
  hlo_op: str
  hlo_module: str
  start: float  # in seconds
  end: float  # in seconds
  duration: float  # in seconds
  correlation_id: int
  scope_range_id: int


def _parse_event(plane, event) -> ParsedEvent:
  event_metadata = plane.event_metadata[event.metadata_id]
  ref_stats = {
    plane.stat_metadata[stat.metadata_id].name: plane.stat_metadata[stat.ref_value].name
    for stat in event.stats
    if stat.ref_value
  }
  correlation_id = [
    stat.uint64_value for stat in event.stats if plane.stat_metadata[stat.metadata_id].name == "correlation_id"
  ]
  correlation_id = None if len(correlation_id) != 1 else correlation_id[0]
  scope_range_id = [
    stat.int64_value for stat in event.stats if plane.stat_metadata[stat.metadata_id].name == "scope_range_id"
  ]
  scope_range_id = None if len(scope_range_id) != 1 else scope_range_id[0]
  parsed_event = ParsedEvent(
    name=event_metadata.name,
    scopes=ref_stats.get("name", None),
    hlo_op=ref_stats.get("hlo_op", None),
    hlo_module=ref_stats.get("hlo_module", None),
    start=event.offset_ps / 1e12,
    end=(event.offset_ps + event.duration_ps) / 1e12,
    duration=event.duration_ps / 1e12,
    correlation_id=correlation_id,
    scope_range_id=scope_range_id,
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


def get_events_from_plane(p: xplane_pb2.XSpace, plane_idx: int) -> dict[str, list[EventScopeNode]]:
  """Get a dictionary of top level scope events. Because they can repeat, aggregate them in a list.

  Use :func:`find_device_plane_ids` to find the plane indices of planes corresponding to an accelerator.
  """
  plane = p.planes[plane_idx]
  all_events = {}
  for line in plane.lines:
    prev_root_key, prev_scope_range_id = None, None
    running_event = EventScopeNode()  # running event is a top level named-scope/jit-function
    for _, event in enumerate(line.events):
      parsed_event = _parse_event(plane, event)
      keys = parsed_event.scopes.split("/") if parsed_event.scopes is not None else [parsed_event.name]

      if (prev_root_key is not None and prev_root_key != keys[0]) or (
        prev_scope_range_id is not None and abs(prev_scope_range_id - parsed_event.scope_range_id) > 1
      ):
        # when the top level name changes or scope_range_id jumps by more than 1
        # we need to aggregate time and save the complete event
        _combine_scope_children_times(running_event)
        assert len(running_event.children) == 1  # we should have accumulated exactly 1 root event
        event_node = list(running_event.children.values())[0]
        all_events.setdefault(event_node.name, []).append(event_node)
        running_event = EventScopeNode()

      _insert_event_into_scope_tree(running_event, parsed_event, keys)
      prev_root_key, prev_scope_range_id = keys[0], parsed_event.scope_range_id

    # finally, combine the last running event and save it
    _combine_scope_children_times(running_event)
    assert len(running_event.children) == 1  # we should have accumulated exactly 1 root event
    event_node = list(running_event.children.values())[0]
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
