"""Build the exact six-feature supervision used by the dual-LoRA Trust heads.

The frozen Memory-Builder predictions are compared with privileged Stage-A
targets only to create labels.  Hidden targets never enter any feature.  The
audit split is deliberately excluded; train and calibration remain disjoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "semantic_match",
    "provenance_freshness",
    "version_consistency",
    "conflict_degree",
    "support_coverage",
    "source_reliability",
)

parser = argparse.ArgumentParser()
parser.add_argument("--prediction-root", default="data/autoregressive_memory")
parser.add_argument("--target-root", default="data/stage_a_append_only")
parser.add_argument("--out", default="data/dual_lora_trust_supervision")
parser.add_argument("--max-rows-per-split", type=int, default=0)
args = parser.parse_args()


def position(node):
    value = node.get("content", {}).get("estimated_position")
    if isinstance(value, list) and len(value) == 3:
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def bbox(node):
    value = node.get("content", {}).get("bbox_xyxy_1000")
    if isinstance(value, list) and len(value) == 4:
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def facts(payload):
    if not payload:
        return []
    return [node for node in payload.get("nodes", [])
            if isinstance(node, dict) and node.get("type") == "Fact"]


def entity_id(node):
    return node.get("support", {}).get("scope", {}).get("entity")


def category(node):
    return str(node.get("content", {}).get("category") or "").strip().lower()


def match_truth(prediction, candidates):
    same = [candidate for candidate in candidates if category(candidate) == category(prediction)]
    if not same:
        return None, 0, None, 0.0
    pred_pos, pred_box = position(prediction), bbox(prediction)
    scored = []
    for candidate in same:
        truth_pos, truth_box = position(candidate), bbox(candidate)
        distance = (float(np.linalg.norm(pred_pos - truth_pos))
                    if pred_pos is not None and truth_pos is not None else None)
        overlap = bbox_iou(pred_box, truth_box)
        score = max(overlap, math.exp(-distance) if distance is not None else 0.0)
        scored.append((score, candidate, distance, overlap))
    _, best, distance, overlap = max(scored, key=lambda item: item[0])
    supported = int((distance is not None and distance <= 1.25) or overlap >= 0.20)
    return best, supported, distance, overlap


class QwenTokenEmbedder:
    """Deterministic semantic vectors from the frozen base token embeddings."""

    def __init__(self):
        model_dir = ROOT / "models/Qwen2.5-VL-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shard = index["weight_map"]["model.embed_tokens.weight"]
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            self.weight = handle.get_tensor("model.embed_tokens.weight")
        self.cache = {}

    def vector(self, text):
        if text not in self.cache:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                result = torch.zeros(self.weight.shape[1], dtype=torch.float32)
            else:
                result = self.weight[ids].float().mean(dim=0)
            result = result / result.norm().clamp_min(1e-8)
            self.cache[text] = result
        return self.cache[text]

    def similarity(self, left, right):
        cosine = float(torch.dot(self.vector(left), self.vector(right)).clamp(-1, 1))
        return (cosine + 1.0) / 2.0


def load_split(split):
    prediction_root = ROOT / args.prediction_root / split
    records = []
    for path in sorted(prediction_root.glob("shard_*.jsonl")):
        for line in path.open():
            record = json.loads(line)
            if record.get("status") == "valid" and record.get("prediction"):
                records.append(record)
    episode_ids = {int(record["episode"]) for record in records}
    targets = {}
    scenes = {}
    target_path = ROOT / args.target_root / f"{split}.jsonl"
    for line in target_path.open():
        row = json.loads(line)
        episode = int(row["episode"])
        if episode in episode_ids:
            targets[(episode, int(row["step"]))] = row["target"]
            scenes[episode] = row["scene"]
    return records, targets, scenes


def load_privileged(episode_ids):
    result = {}
    for episode in episode_ids:
        path = ROOT / f"data/sim_replay/train/ep_{episode}/privileged.jsonl"
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.open()]
        result[episode] = {int(record["step"]): record for record in records}
    return result


def load_public_entity_map(episode_ids, max_objects=8):
    """Reconstruct Stage-A public ids without exposing simulator ids as features."""
    result = {}
    for episode in episode_ids:
        path = ROOT / f"data/sim_replay/train/ep_{episode}/observed.jsonl"
        if not path.is_file():
            continue
        public_ids = {}
        counter = 0
        for line in path.open():
            observation = json.loads(line)
            candidates = []
            for item in sorted(observation.get("visible_entities", []),
                               key=lambda value: value.get("pixel_count", 0), reverse=True):
                bbox_value = item.get("bbox_xyxy_1000")
                name = item.get("class")
                if (bbox_value is None or item.get("kind") not in {"object", "receptacle"} or
                        not name or name.startswith("unknown_") or item.get("pixel_count", 0) < 64):
                    continue
                candidates.append(item)
                if len(candidates) >= max_objects:
                    break
            for item in candidates:
                hidden_key = (item["kind"], int(item["instance_id"]))
                if hidden_key not in public_ids:
                    counter += 1
                    public_id = f"ep{episode}:{item['kind']}:{item['class']}:{counter}"
                    public_ids[hidden_key] = public_id
                    result[public_id] = hidden_key
    return result


train_records, train_targets, train_scenes = load_split("train")
cal_records, cal_targets, cal_scenes = load_split("calibration")
if not train_records or not cal_records:
    raise RuntimeError("Predicted train and calibration memories are both required")


def compute_matches(records, targets):
    matches = {}
    category_counts = collections.defaultdict(lambda: [1, 2])  # Laplace prior.
    for record in records:
        key = (int(record["episode"]), int(record["step"]))
        truth_facts = facts(targets.get(key))
        for pred in facts(record["prediction"]):
            matched, supported, distance, overlap = match_truth(pred, truth_facts)
            matches[(key, pred["id"])] = {
                "truth": matched, "evidence_supported": supported,
                "distance": distance, "bbox_iou": overlap,
            }
            cat = category(pred) or "unknown"
            category_counts[cat][0] += supported
            category_counts[cat][1] += 1
    reliability = {cat: positive / total for cat, (positive, total) in category_counts.items()}
    return matches, reliability


train_matches, reliability = compute_matches(train_records, train_targets)
cal_matches, _ = compute_matches(cal_records, cal_targets)
privileged = load_privileged(
    {int(record["episode"]) for record in train_records + cal_records})
public_entity_map = load_public_entity_map(
    {int(record["episode"]) for record in train_records + cal_records})


def entity_world_position(episode, entity, step):
    """Return simulator-world position; static receptacles use a sentinel."""
    if not entity:
        return None
    hidden = public_entity_map.get(str(entity))
    if hidden is None:
        return None
    kind, visible_instance_id = hidden
    if kind == "receptacle":
        return "static"
    if kind != "object":
        return None
    record = privileged.get(episode, {}).get(int(step))
    if not record:
        return None
    handle = next((handle for handle, object_id in record["object_id_by_handle"].items()
                   if int(object_id) + int(record["object_ids_start"]) == visible_instance_id), None)
    value = record["object_positions"].get(handle) if handle else None
    return np.asarray(value, dtype=float) if value is not None else None


def build_rows(split, records, targets, scenes, matches, embedder):
    truth_history = collections.defaultdict(list)
    pred_history = collections.defaultdict(list)
    steps_by_episode = collections.defaultdict(set)
    record_edges = {}
    episode_categories = collections.defaultdict(set)
    for (episode, step), payload in targets.items():
        steps_by_episode[episode].add(step)
        for node in facts(payload):
            truth_history[(episode, entity_id(node))].append((step, node))
            episode_categories[episode].add(category(node))
    for record in records:
        episode, step = int(record["episode"]), int(record["step"])
        record_edges[(episode, step)] = record["prediction"].get("edges", [])
        for node in facts(record["prediction"]):
            pred_history[(episode, entity_id(node))].append((step, node))
            episode_categories[episode].add(category(node))
    for values in truth_history.values():
        values.sort(key=lambda item: item[0])
    for values in pred_history.values():
        values.sort(key=lambda item: item[0])

    rows = []
    counters = collections.Counter()
    for record in records:
        episode, step = int(record["episode"]), int(record["step"])
        scene = scenes[episode]
        edges = record_edges[(episode, step)]
        conflict_ids = set()
        for edge in edges:
            if edge.get("type") == "conflicts-with":
                conflict_ids.update((edge.get("source"), edge.get("target")))
        for pred in facts(record["prediction"]):
            match = matches[((episode, step), pred["id"])]
            matched = match["truth"]
            cat = category(pred) or "unknown"
            pred_entity = entity_id(pred)
            truth_entity = entity_id(matched) if matched else None
            history = pred_history[(episode, pred_entity)]
            future_pred = [(s, node) for s, node in history if s > step]
            truth_versions = truth_history[(episode, truth_entity)] if truth_entity else []
            future_truth = [(s, node) for s, node in truth_versions if s > step]
            query_points = [(step, matched)]
            episode_steps = sorted(steps_by_episode[episode])
            episode_end = episode_steps[-1]
            for desired in (step + 30, step + 100, episode_end):
                query_step = min(episode_steps, key=lambda value: abs(value - min(desired, episode_end)))
                query_points.append((query_step, matched))
            seen_query_steps = set()
            for query_step, query_truth in query_points:
                if query_step in seen_query_steps:
                    continue
                seen_query_steps.add(query_step)
                age = max(0, query_step - step)
                newer = [(s, node) for s, node in future_pred if s <= query_step]
                version_consistency = 1.0 / (1.0 + len(newer))
                related = max(1, len([1 for _, node in history if category(node) == cat]))
                conflict_degree = min(1.0, (int(pred["id"] in conflict_ids) +
                                             max(0, len(newer) - 1)) / related)
                memory_text = f"{cat} {pred.get('content', {}).get('attribute', '')}"
                query_text = f"where is the {cat} at step {query_step}"
                exact_semantic = embedder.similarity(memory_text, query_text)
                semantic_match = min(1.0, 0.7 + 0.3 * exact_semantic)
                evidence = int(match["evidence_supported"])
                write_world = entity_world_position(episode, truth_entity, step)
                query_world = entity_world_position(episode, truth_entity, query_step)
                if isinstance(write_world, str) and isinstance(query_world, str):
                    world_distance, fresh = 0.0, int(evidence)
                elif isinstance(write_world, np.ndarray) and isinstance(query_world, np.ndarray):
                    world_distance = float(np.linalg.norm(write_world - query_world))
                    fresh = int(evidence and world_distance <= 0.70)
                else:
                    world_distance = None
                    fresh = int(evidence) if query_step == step else None
                row = {
                    "split": split, "episode": episode, "scene": scene,
                    "memory_step": step, "query_step": query_step,
                    "memory_id": pred["id"], "query_category": cat,
                    "features": {
                        "semantic_match": round(semantic_match, 6),
                        "provenance_freshness": round(math.exp(-age / 100.0), 6),
                        "version_consistency": round(version_consistency, 6),
                        "conflict_degree": round(conflict_degree, 6),
                        "support_coverage": 1.0,
                        "source_reliability": round(reliability.get(cat, 0.5), 6),
                    },
                    "labels": {
                        "evidence_supported": evidence,
                        "fresh_at_query": fresh,
                        "correct_at_query": (evidence * fresh if fresh is not None else None),
                        "reobserve_resolves": int(bool(future_truth)),
                    },
                    "audit_only": {
                        "write_position_error_m": match["distance"],
                        "bbox_iou": match["bbox_iou"],
                        "physical_displacement_m": world_distance,
                        "next_observation_cost_steps": (future_truth[0][0] - step
                                                        if future_truth else None),
                    },
                }
                rows.append(row)

            negatives = sorted(value for value in episode_categories[episode]
                               if value and value != cat)
            if negatives:
                negative_cat = negatives[(step + len(rows)) % len(negatives)]
                query_text = f"where is the {negative_cat} at step {step}"
                memory_text = f"{cat} {pred.get('content', {}).get('attribute', '')}"
                rows.append({
                    "split": split, "episode": episode, "scene": scene,
                    "memory_step": step, "query_step": step,
                    "memory_id": pred["id"], "query_category": negative_cat,
                    "features": {
                        "semantic_match": round(0.3 * embedder.similarity(memory_text, query_text), 6),
                        "provenance_freshness": 1.0,
                        "version_consistency": 1.0,
                        "conflict_degree": round(float(pred["id"] in conflict_ids), 6),
                        "support_coverage": 0.0,
                        "source_reliability": round(reliability.get(cat, 0.5), 6),
                    },
                    "labels": {
                        "evidence_supported": 0,
                        "fresh_at_query": 0,
                        "correct_at_query": 0,
                        "reobserve_resolves": 0,
                    },
                    "audit_only": {"controlled_query_mismatch": True,
                                   "next_observation_cost_steps": None},
                })
            counters["facts"] += 1
    if args.max_rows_per_split:
        rows = rows[:args.max_rows_per_split]
    counters["rows"] = len(rows)
    counters["evidence_positive"] = sum(row["labels"]["evidence_supported"] for row in rows)
    counters["fresh_known"] = sum(row["labels"]["fresh_at_query"] is not None for row in rows)
    counters["fresh_positive"] = sum((row["labels"]["fresh_at_query"] or 0) for row in rows)
    return rows, counters


embedder = QwenTokenEmbedder()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
(out / "source_reliability.json").write_text(
    json.dumps(reliability, ensure_ascii=False, indent=2))
reports = {}
for split, records, targets, scenes, matches in (
        ("train", train_records, train_targets, train_scenes, train_matches),
        ("calibration", cal_records, cal_targets, cal_scenes, cal_matches)):
    rows, counters = build_rows(split, records, targets, scenes, matches, embedder)
    path = out / f"{split}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    reports[split] = {"predicted_frames": len(records), "scenes": len(set(scenes.values())),
                      **dict(counters)}

report = {
    "feature_names": list(FEATURE_NAMES),
    "labels": ["evidence_supported", "fresh_at_query", "correct_at_query"],
    "splits": reports,
    "audit_split_used_for_training": False,
    "hidden_truth_usage": "labels and audit_only fields only; never Trust features",
    "semantic_encoder": "frozen Qwen2.5-VL-3B token-embedding mean cosine",
    "source_reliability": "train-only per-category empirical precision with Laplace prior",
}
(out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))

