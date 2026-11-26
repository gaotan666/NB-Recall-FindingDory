"""Typed, append-only memory with provenance-preserving temporal lookup.

Memory observations are immutable evidence.  A later observation may point to
an earlier one with ``version-of`` or ``conflicts-with``, but it never edits,
merges, deletes, or overwrites the earlier node.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

NODE_TYPES = {"Entity", "Fact", "Event", "Procedure"}
VALIDITY_STATES = {"valid", "stale", "disputed"}
WRITE_OPERATIONS = {"DROP", "WRITE", "MERGE", "UPDATE"}
EDGE_TYPES = {"generated-by", "before", "after", "supports", "occludes",
              "same-entity", "version-of", "conflicts-with"}


@dataclass
class MemoryNode:
    id: str
    type: str
    content: Any
    provenance: List[Dict[str, Any]]
    validity: Dict[str, Any]
    support: Dict[str, Any]
    trust: Optional[float] = None
    entity_id: Optional[str] = None
    attribute: Optional[str] = None
    acquisition_group: Optional[str] = None
    evidence_values: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.type not in NODE_TYPES:
            raise ValueError(f"Unsupported node type: {self.type}")
        if self.validity.get("state") not in VALIDITY_STATES:
            raise ValueError("Invalid validity state")
        if not self.provenance or any(not all(k in p for k in
            ("source", "time", "episode", "observation_ref")) for p in self.provenance):
            raise ValueError("Every node needs a traceable observation")
        if self.trust is not None and not 0 <= self.trust <= 1:
            raise ValueError("Trust must be in [0,1]")
        scope = self.support.get("scope", {})
        if not all(k in scope for k in ("entity", "scene", "task_condition")):
            raise ValueError("Support must carry entity, scene and task_condition")


@dataclass
class MemoryEdge:
    source: str
    target: str
    type: str

    def __post_init__(self):
        if self.type not in EDGE_TYPES:
            raise ValueError(f"Unsupported edge type: {self.type}")


@dataclass
class Lookup:
    node: Optional[MemoryNode]
    disputed: bool
    alternatives: List[MemoryNode] = field(default_factory=list)


class MemoryGraph:
    def __init__(self, merge_position_radius_m: float):
        if merge_position_radius_m <= 0:
            raise ValueError("merge_position_radius_m must be positive")
        self.merge_position_radius_m = merge_position_radius_m
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []

    def add_edge(self, source: str, target: str, kind: str):
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both edge endpoints must exist")
        edge = MemoryEdge(source, target, kind)
        if edge not in self.edges:
            self.edges.append(edge)

    def apply(self, operation: str, candidate: MemoryNode, existing_id: Optional[str] = None):
        if operation not in WRITE_OPERATIONS:
            raise ValueError(f"Unknown write operation: {operation}")
        if operation in {"DROP", "WRITE"}:
            # DROP is retained only for compatibility with old callers.  The
            # append-only contract stores the evidence and lets Trust decide
            # whether it should be used.
            if candidate.id in self.nodes:
                raise ValueError(f"Duplicate node id: {candidate.id}")
            self.nodes[candidate.id] = candidate
            return candidate
        if not existing_id or existing_id not in self.nodes:
            raise KeyError("MERGE/UPDATE require an existing node")
        old = self.nodes[existing_id]
        if candidate.id in self.nodes:
            raise ValueError(f"Duplicate node id: {candidate.id}")
        if operation == "MERGE":
            if (old.type != candidate.type or old.entity_id != candidate.entity_id or
                old.attribute != candidate.attribute or
                old.acquisition_group != candidate.acquisition_group):
                raise ValueError("MERGE requires compatible nodes in one acquisition group")
            compatible = old.content == candidate.content
            if (not compatible and isinstance(old.content, list) and isinstance(candidate.content, list)
                and len(old.content) == len(candidate.content) == 3
                and all(isinstance(x, (int, float)) for x in old.content + candidate.content)):
                compatible = math.dist(old.content, candidate.content) <= self.merge_position_radius_m
            if not compatible:
                raise ValueError("Conflicting values cannot be merged")
            # A historical MERGE request becomes a new immutable observation.
            self.nodes[candidate.id] = candidate
            self.add_edge(candidate.id, old.id, "same-entity")
            return candidate
        if old.type != "Fact" or candidate.type != "Fact":
            raise ValueError("Only Fact has a version chain")
        if old.entity_id != candidate.entity_id or old.attribute != candidate.attribute:
            raise ValueError("UPDATE must concern the same entity and attribute")
        old_time = old.validity["t_valid_from"]
        new_time = candidate.validity["t_valid_from"]
        self.nodes[candidate.id] = candidate
        if new_time <= old_time:
            # No reliable temporal ordering: preserve both and record conflict
            # without mutating either node.
            self.add_edge(candidate.id, old.id, "conflicts-with")
            return candidate
        self.add_edge(candidate.id, old.id, "version-of")
        return candidate

    def append_observation(self, candidate: MemoryNode,
                           entity_node_id: Optional[str] = None):
        """Append one observation and derive links without editing old nodes."""
        if candidate.id in self.nodes:
            raise ValueError(f"Duplicate node id: {candidate.id}")
        prior = None
        if candidate.type == "Fact":
            versions = [node for node in self.nodes.values()
                        if node.type == "Fact" and
                        node.entity_id == candidate.entity_id and
                        node.attribute == candidate.attribute and
                        node.support["scope"]["scene"] ==
                        candidate.support["scope"]["scene"]]
            prior = max(versions, key=lambda n: n.validity["t_valid_from"],
                        default=None)
        self.nodes[candidate.id] = candidate
        if entity_node_id is not None:
            self.add_edge(candidate.id, entity_node_id, "supports")
        if prior is None:
            return {"node": candidate, "relation": "new"}
        self.add_edge(candidate.id, prior.id, "same-entity")
        compatible = prior.content == candidate.content
        if (not compatible and isinstance(prior.content, list) and
                isinstance(candidate.content, list) and
                len(prior.content) == len(candidate.content) == 3 and
                all(isinstance(x, (int, float))
                    for x in prior.content + candidate.content)):
            compatible = math.dist(prior.content, candidate.content) <= self.merge_position_radius_m
        if compatible:
            return {"node": candidate, "relation": "same-entity"}
        old_time = prior.validity["t_valid_from"]
        new_time = candidate.validity["t_valid_from"]
        relation = "version-of" if new_time > old_time else "conflicts-with"
        self.add_edge(candidate.id, prior.id, relation)
        return {"node": candidate, "relation": relation}

    def _effective_end(self, node: MemoryNode):
        """Infer a version end from append-only edges; never edit the node."""
        ends = []
        explicit = node.validity.get("t_valid_to")
        if explicit is not None:
            ends.append(explicit)
        for edge in self.edges:
            if edge.type != "version-of" or edge.target != node.id:
                continue
            newer = self.nodes.get(edge.source)
            if newer is not None and newer.validity["t_valid_from"] > node.validity["t_valid_from"]:
                ends.append(newer.validity["t_valid_from"])
        return min(ends) if ends else None

    def query_fact(self, entity_id: str, attribute: str, reference_time: float,
                   scene: str) -> Lookup:
        versions = [n for n in self.nodes.values() if n.type == "Fact" and
                    n.entity_id == entity_id and n.attribute == attribute and
                    n.support["scope"]["scene"] == scene]
        active = []
        for node in versions:
            end = self._effective_end(node)
            if node.validity["t_valid_from"] <= reference_time and (end is None or reference_time < end):
                active.append(node)
        if not active:
            return Lookup(None, False, versions)
        if len(active) > 1 or any(n.validity["state"] == "disputed" for n in active):
            return Lookup(None, True, active)
        return Lookup(active[0], False, [])

    def to_dict(self):
        return {"merge_position_radius_m": self.merge_position_radius_m,
                "nodes": [vars(n) for n in self.nodes.values()],
                "edges": [vars(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, data):
        graph = cls(data["merge_position_radius_m"])
        for row in data["nodes"]:
            node = MemoryNode(**row)
            graph.nodes[node.id] = node
        graph.edges = [MemoryEdge(**row) for row in data["edges"]]
        return graph
