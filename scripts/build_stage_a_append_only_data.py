"""Build Stage-A supervision for one append-only NB-Recall LoRA.

Only train-split observations are used.  Every grounded sighting becomes a new
immutable Fact; repeated and changed sightings are linked, never merged,
deleted, or overwritten.  Simulator instance ids are used only to supervise
same-entity/version links and are replaced by episode-local public ids.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/stage_a_append_only")
parser.add_argument("--max-objects", type=int, default=8)
parser.add_argument("--seed", type=int, default=20261119)
args = parser.parse_args()

source = ROOT / "data/sim_replay/train"
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
with gzip.open(ROOT / "data/findingdory-habitat/findingdory/train/episodes.json.gz", "rt") as f:
    metadata = {int(row["episode_id"]): row for row in json.load(f)["episodes"]}


def relation_changed(old, new, radius=0.7):
    if old is None or new is None or len(old) != 3 or len(new) != 3:
        return False
    return math.dist(old, new) > radius


rows = []
counts = collections.Counter()
scenes_seen = set()
paths = sorted(source.glob("ep_*/summary.json"), key=lambda p: int(p.parent.name[3:]))
for summary_path in paths:
    episode = int(summary_path.parent.name[3:])
    summary = json.loads(summary_path.read_text())
    if not summary.get("collection_complete") or episode not in metadata:
        continue
    scene = metadata[episode]["scene_id"]
    scenes_seen.add(scene)
    public_ids = {}
    public_counter = 0
    latest_fact = {}
    latest_value = {}
    previous_event = None
    recent = collections.deque(maxlen=12)
    for line in (summary_path.parent / "observed.jsonl").open():
        obs = json.loads(line)
        if not obs.get("frame_ref"):
            continue
        candidates = []
        for entity in sorted(obs.get("visible_entities", []),
                             key=lambda item: item.get("pixel_count", 0), reverse=True):
            bbox = entity.get("bbox_xyxy_1000")
            category = entity.get("class")
            if (bbox is None or entity.get("kind") not in {"object", "receptacle"} or
                    not category or category.startswith("unknown_") or
                    entity.get("pixel_count", 0) < 64):
                continue
            candidates.append(entity)
            if len(candidates) >= args.max_objects:
                break
        if not candidates:
            continue
        step = int(obs["step"])
        image = str((summary_path.parent / obs["frame_ref"]).relative_to(ROOT))
        previous_snapshot = list(recent)
        provenance = {
            "source": "findingdory_train_sensor",
            "time": step,
            "episode": str(episode),
            "action": obs.get("oracle_action_name") or None,
            "observation_ref": image,
            "sha256": obs.get("rgb_sha256"),
        }
        nodes, edges = [], []
        event_id = f"event:ep{episode}:step{step}"
        event_node = {
            "id": event_id,
            "type": "Event",
            "content": {"action": obs.get("oracle_action_name") or None,
                        "time_of_day": obs.get("time_of_day"),
                        "visible_categories": [item["class"] for item in candidates]},
            "provenance": [provenance],
            "validity": {"state": "valid", "t_valid_from": step, "t_valid_to": None},
            "support": {"scope": {"entity": None, "scene": scene,
                                    "task_condition": "observation_order"}},
            "trust": None,
        }
        nodes.append(event_node)
        if previous_event is not None:
            edges.extend([{"source": previous_event, "target": event_id, "type": "before"},
                          {"source": event_id, "target": previous_event, "type": "after"}])
        previous_event = event_id
        for index, entity in enumerate(candidates):
            hidden_key = (entity["kind"], int(entity["instance_id"]))
            if hidden_key not in public_ids:
                public_counter += 1
                public_ids[hidden_key] = f"ep{episode}:{entity['kind']}:{entity['class']}:{public_counter}"
                eid = public_ids[hidden_key]
                nodes.append({
                    "id": f"entity:{eid}", "type": "Entity",
                    "content": {"kind": entity["kind"], "category": entity["class"]},
                    "provenance": [provenance],
                    "validity": {"state": "valid", "t_valid_from": step, "t_valid_to": None},
                    "support": {"scope": {"entity": eid, "scene": scene,
                                            "task_condition": entity["kind"]}},
                    "trust": None,
                })
            eid = public_ids[hidden_key]
            fact_id = f"fact:{eid}:step{step}:obs{index}"
            value = entity.get("estimated_position_world")
            nodes.append({
                "id": fact_id, "type": "Fact",
                "content": {"attribute": "observed_location",
                            "category": entity["class"],
                            "bbox_xyxy_1000": entity["bbox_xyxy_1000"],
                            "estimated_position": value,
                            "depth_m": (round(float(entity["depth_m"]), 4)
                                        if entity.get("depth_m") is not None else None)},
                "provenance": [provenance],
                "validity": {"state": "valid", "t_valid_from": step, "t_valid_to": None},
                "support": {"scope": {"entity": eid, "scene": scene,
                                        "task_condition": "object_location"}},
                "trust": None,
            })
            edges.append({"source": fact_id, "target": f"entity:{eid}", "type": "supports"})
            prior = latest_fact.get(hidden_key)
            if prior is not None:
                edges.append({"source": fact_id, "target": prior, "type": "same-entity"})
                if relation_changed(latest_value.get(hidden_key), value):
                    edges.append({"source": fact_id, "target": prior, "type": "version-of"})
                    counts["version_links"] += 1
            latest_fact[hidden_key] = fact_id
            latest_value[hidden_key] = value
            recent.append({"id": fact_id, "entity": eid, "category": entity["class"],
                           "observed_at": step, "position": value})
            counts["facts"] += 1
        target = {"operation": "APPEND", "nodes": nodes, "edges": edges}
        rows.append({
            "episode": episode, "scene": scene, "step": step, "image": image,
            "previous_memory": previous_snapshot,
            "target": target,
            "invariants": {"append_only": True, "old_nodes_modified": False,
                           "nodes_dropped": False, "nodes_merged": False},
        })
        counts["frames"] += 1
        counts["events"] += 1
    counts["episodes"] += 1

scenes = sorted(scenes_seen)
rng = np.random.default_rng(args.seed)
rng.shuffle(scenes)
n_fit = max(1, int(len(scenes) * 0.8))
n_cal = max(1, int(len(scenes) * 0.1))
groups = {"train": set(scenes[:n_fit]),
          "calibration": set(scenes[n_fit:n_fit+n_cal]),
          "audit": set(scenes[n_fit+n_cal:])}
if not groups["audit"]:
    raise RuntimeError("No audit scenes after split")

split_counts = {}
for name, group in groups.items():
    selected = [row for row in rows if row["scene"] in group]
    with (out / f"{name}.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    split_counts[name] = len(selected)

report = {
    "method": "NB-Recall Stage A memory-builder LoRA append-only supervision",
    "counts": dict(counts), "split_counts": split_counts,
    "scene_split": {name: sorted(group) for name, group in groups.items()},
    "source": "FindingDory train replay observable RGB/panoptic/depth metadata",
    "privileged_instance_use": "same-entity/version link supervision only; hidden ids absent from output",
    "hard_invariants": ["append every valid sighting", "never delete", "never overwrite",
                        "never merge", "old nodes remain byte-identical"],
}
(out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
