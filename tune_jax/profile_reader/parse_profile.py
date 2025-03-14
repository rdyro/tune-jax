import sys
from pathlib import Path
from pprint import pprint
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
  "get_scopes_trie",
]


def parse_profile_from_bytes(profile_bytes: bytes) -> xplane_pb2.XSpace:
  p = xplane_pb2.XSpace()
  p.ParseFromString(profile_bytes)
  return p


def find_device_plane_ids(p: xplane_pb2.XSpace, device_str: str) -> list[int]:
  return [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]


@dataclasses.dataclass(frozen=True)
class Event:
  name: str
  scopes: str
  hlo_op: str
  hlo_module: str


@dataclasses.dataclass(frozen=True)
class EventData:
  duration_s: float
  t_start_s: float


def get_events_from_plane(p: xplane_pb2.XSpace, plane_idx: int):
  plane = p.planes[plane_idx]
  all_events = {}
  for line in plane.lines:
    for event in line.events:
      event_metadata = plane.event_metadata[event.metadata_id]
      ref_stats = {
        plane.stat_metadata[stat.metadata_id].name: plane.stat_metadata[stat.ref_value].name
        for stat in event.stats
        if stat.ref_value
      }
      parsed_event = Event(
        name=event_metadata.name,
        scopes=ref_stats.get("name", None),
        hlo_op=ref_stats.get("hlo_op", None),
        hlo_module=ref_stats.get("hlo_module", None),
      )
      all_events.setdefault(parsed_event, []).append(EventData(event.duration_ps / 1e12, event.offset_ps / 1e12))
  all_events = dict(sorted(all_events.items(), key=lambda x: -max([z.duration_s for z in x[1]], default=0)))
  return all_events


# building the scope trie
@dataclasses.dataclass
class TrieNode:
  name: str | None = None
  event: Event | None = None
  starts: tuple[float] | None = None
  ends: tuple[float] | None = None
  durations: tuple[float] | None = None
  children: dict[str, "TrieNode"] = dataclasses.field(default_factory=lambda: {})


def _insert_into_trie(root: TrieNode, event: Event, times: list[EventData], keys: list[str]):
  if len(keys) == 0:
    return
  child_name = keys[0] if len(keys) > 1 else f"{keys[0]}-{event.hlo_op}"
  node = root.children.setdefault(child_name, TrieNode(name=keys[0]))
  _insert_into_trie(node, event, times, keys[1:])
  if len(keys) == 1:
    starts = tuple(t.t_start_s for t in times)
    durations = tuple(t.duration_s for t in times)
    ends = tuple(s + t for s, t in zip(starts, durations))
    root.children[child_name] = dataclasses.replace(
      node, name=child_name, event=event, durations=durations, starts=starts, ends=ends
    )


def _cache_trie_times(node: TrieNode) -> tuple[list[int], list[int]]:
  """Insert start times, end times and durations for scopes consisting of other scopes."""
  if node.starts is None:
    assert node.ends is None and node.durations is None
    children_starts_ends = [_cache_trie_times(child) for child in node.children.values()]
    if len(children_starts_ends) == 0:
      return [], []
    n_occurrences = len(children_starts_ends[0][0])
    if not (
      all(len(child[0]) == n_occurrences for child in children_starts_ends)
      and all(len(child[1]) == n_occurrences for child in children_starts_ends)
    ):
      if node.name is None:  # this is the root node
        return [0], [0]
      raise ValueError(
        "Node children have mismatched run counts, likely because event disamgiguation is imperfect. Create an issue please."
      )
    node.starts, node.ends, node.durations = [], [], []
    for i in range(n_occurrences):
      start = min([child[0][i] for child in children_starts_ends], default=0)
      end = max([child[1][i] for child in children_starts_ends], default=0)
      node.starts.append(start)
      node.ends.append(end)
      node.durations.append(end - start)
  return node.starts, node.ends


def get_scopes_trie(events: dict[Event, list[float]]) -> dict[str, TrieNode]:
  """Parse the flat list of events into a trie of scopes, jit functions and named_scopes live as scopes."""
  root = TrieNode(None, None)
  for event, times in events.items():
    keys = event.scopes.split("/")
    _insert_into_trie(root, event, times, keys)
  _cache_trie_times(root)
  return root.children  # root is a stand-in it's unnecessary


if __name__ == "__main__":
  all_profiles = list(Path("~/profiles").expanduser().glob("**/*.xplane.pb"))
  latest_profile = sorted(all_profiles, key=lambda x: x.stat().st_mtime)[-1]
  print(f"latest profile = {latest_profile}")
  p = parse_profile_from_bytes(latest_profile.read_bytes())
  gpu_plane_ids = find_device_plane_ids(p, "gpu")
  print([p.planes[i].name for i in gpu_plane_ids])

  events = get_events_from_plane(p, gpu_plane_ids[0])
  times = get_scopes_trie(events)
  pprint({k: v.durations for k, v in times.items()})
