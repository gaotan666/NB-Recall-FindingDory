"""Join official questions to frozen question-blind memories of the same videos."""
import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb_recall.runtime import TrustScorer
from nb_recall.stage_c import retrieval_features, validate_answer
from nb_recall.retrieval_action8 import select_action8

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source-split", choices=["train", "validation", "sim_validation"], default="train")
parser.add_argument("--episode-id", default=None, help="Process exactly one ep_#")
parser.add_argument("--episodes", type=int, default=12, help="0 means all")
parser.add_argument("--smoke-three-videos", action="store_true")
parser.add_argument("--out", default="data/stage_c_answer_pilot")
parser.add_argument("--separate-reader-gate", action="store_true",
                    help="Supervise READ on official answers and GATE separately")
parser.add_argument("--allow-partial-for-preflight", action="store_true",
                    help="I/O and backward smoke only; never a full training corpus")
args = parser.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
question_split = "validation" if args.source_split == "sim_validation" else args.source_split
rows = pq.read_table(ROOT / f"data/findingdory-subsampled-96/data/{question_split}-00000-of-00001.parquet").to_pylist()
episodes = sorted({r["ep_id"] for r in rows}, key=lambda key: int(key.split("_")[-1]))
pilot_holdout = set(episodes[9:12])
pilot_holdout.add(episodes[2])
if args.episode_id:
    if args.episode_id not in episodes:
        raise ValueError("Unknown episode id")
    episodes = [args.episode_id]
elif args.smoke_three_videos:
    args.episodes = 3
if args.episodes:
    episodes = episodes[:args.episodes]
allowed = set(episodes)


class Embedder:
    def __init__(self):
        base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        index = json.loads((base / "model.safetensors.index.json").read_text())
        with safe_open(base / index["weight_map"]["model.embed_tokens.weight"],
                       framework="pt", device="cpu") as handle:
            self.weight = handle.get_tensor("model.embed_tokens.weight")
        self.cache = {}

    def vector(self, text):
        if text not in self.cache:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            value = self.weight[ids].float().mean(0) if ids else torch.zeros(self.weight.shape[1])
            self.cache[text] = value / value.norm().clamp_min(1e-8)
        return self.cache[text]

    def similarity(self, left, right):
        return (float(torch.dot(self.vector(left), self.vector(right))) + 1) / 2


embedder = Embedder()
trust = TrustScorer(ROOT / "models/nb_recall_dual_lora_trust_heads")
reliability = json.loads((ROOT / "data/dual_lora_trust_supervision/source_reliability.json").read_text())
counts = collections.Counter()
handles = {name: (out / f"{name}.jsonl").open("w") for name in ("train", "audit")}
try:
    for ep in episodes:
        folder = ROOT / "data/stage_c_video_memory" / args.source_split / ep
        if not (folder / "complete.json").is_file() and not args.allow_partial_for_preflight:
            raise RuntimeError(f"Question-blind full-video memory not ready for {ep}")
        source = json.loads((folder / "source.json").read_text())
        if source.get("split") != args.source_split:
            raise RuntimeError(f"Memory split mismatch for {ep}")
        observations = [json.loads(line) for line in (folder / "observations.jsonl").open()]
        if [record["clip_index"] for record in observations] != list(range(len(observations))):
            raise RuntimeError(f"Invalid official frame alignment for {ep}")
        candidates = []
        for observation in observations:
            payload = observation.get("prediction") or {}
            for node in payload.get("nodes", []):
                if node.get("type") in {"Fact", "Event"}:
                    candidates.append({"node": node, "clip_index": observation["clip_index"],
                                       "image": observation["image"], "edges": payload.get("edges", [])})
        by_entity = collections.defaultdict(list)
        for candidate in candidates:
            entity = candidate["node"].get("support", {}).get("scope", {}).get("entity")
            by_entity[entity].append(candidate)
        timeline=[]
        for observation in observations:
            nodes=(observation.get('prediction') or {}).get('nodes',[])
            categories=sorted({str(n.get('content',{}).get('category')) for n in nodes
                               if n.get('type')=='Fact' and n.get('content',{}).get('category')})
            events=[n.get('content',{}).get('action') for n in nodes
                    if n.get('type')=='Event' and n.get('content',{}).get('action')]
            timeline.append({'clip':observation['clip_index'],'objects':categories,'actions':events})
        split = "audit" if args.source_split != "train" or ep in pilot_holdout else "train"
        for index, question in enumerate(r for r in rows if r["ep_id"] == ep):
            if args.source_split != "sim_validation" and source["video"] != question["video"]:
                raise RuntimeError(f"Video mismatch for {ep}")
            ranked = []
            features_batch = []
            probability_e, probability_f = [], []
            allowed_frames=set(select_action8(question["question"],observations))
            query_candidates=[c for c in candidates if c["clip_index"] in allowed_frames]
            for candidate in query_candidates:
                entity = candidate["node"].get("support", {}).get("scope", {}).get("entity")
                features = retrieval_features(question["question"], candidate, by_entity[entity], reliability, embedder)
                features_batch.append(features)
            from nb_recall.runtime import TRUST_FEATURES
            if features_batch:
                matrix = np.asarray([[features[name] for name in TRUST_FEATURES] for features in features_batch])
                probability_e = 1/(1+np.exp(-np.clip(trust.evidence.decision_function(matrix)/trust.temp_e,-50,50)))
                probability_f = 1/(1+np.exp(-np.clip(trust.fresh.decision_function(matrix)/trust.temp_f,-50,50)))
            for candidate, features, pe, pf in zip(query_candidates, features_batch, probability_e, probability_f):
                scores = {"tau_evidence": float(pe), "tau_fresh": float(pf), "tau": float(pe*pf)}
                ranked.append((features["semantic_match"], scores["tau"], candidate, features, scores))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected, seen_frames = [], set()
            for _, _, candidate, features, scores in ranked:
                if candidate["clip_index"] not in seen_frames:
                    selected.append((candidate, features, scores))
                    seen_frames.add(candidate["clip_index"])
                if len(selected) == 8:
                    break
            if seen_frames != allowed_frames:
                raise RuntimeError(f"Missing memory nodes for selected frames: {ep} {allowed_frames-seen_frames}")
            memories = []
            for candidate, features, scores in selected:
                node = candidate["node"]
                memories.append({"clip_index": candidate["clip_index"], "type": node.get("type"),
                                 "content": node.get("content", {}),
                                 "features": features, "trust": scores,
                                 "entity": node.get("support", {}).get("scope", {}).get("entity"),
                                 "relations": candidate.get("edges", [])[-8:]})
            tau = max((scores["tau"] for _, _, scores in selected), default=0.0)
            decision = "ANSWER" if tau >= trust.tau_hi else "REOBSERVE" if tau >= trust.tau_lo else "ABSTAIN"
            target_answer = validate_answer(question["answer"])
            if any(group == [-1] for group in target_answer):
                counts[f"{split}_no_successful_history_frame"] += 1
            evidence = [{"clip_index": candidate["clip_index"], "image": candidate["image"]}
                        for candidate, _, _ in selected[:4]]
            context = {"question": question["question"], "memory": memories,
                       "tau_lo": trust.tau_lo, "tau_hi": trust.tau_hi,
                       "phase": "initial", "evidence_revisited": False}
            if args.separate_reader_gate:
                context['task']='GATE'
                context['timeline']=timeline
            common = {"episode": ep, "question_id": f"{ep}:{index}", "split": split,
                      "official_answer": target_answer, "video": question["video"],
                      "category": question["high_level_category"]}
            common['low_level_category']=question['low_level_category']
            policy_target = {"decision": decision,
                             "frames": target_answer if decision == "ANSWER" else [],
                             "observe_frames": [item["clip_index"] for item in evidence] if decision == "REOBSERVE" else []}
            if args.separate_reader_gate:
                # Gold is used only as supervision. Evidence/retrieval was already fixed above.
                canonical=[]
                for group in target_answer:
                    if group==[-1] or not group:
                        canonical.append(group)
                    else:
                        supported=sorted(set(group)&seen_frames)
                        canonical.append([supported[len(supported)//2] if supported else group[len(group)//2]])
                reader={**common,'context':{**context,'task':'READ'},'evidence':[],
                        'training_mode':'READ',
                        'target':{'decision':'ANSWER','frames':canonical,'observe_frames':[]}}
                handles[split].write(json.dumps(reader,ensure_ascii=False,separators=(',',':'))+'\n')
                counts[f'{split}_reader_rows']+=1
            row = {**common, "context": context, "evidence": [], "target": policy_target,
                   "training_mode":"GATE" if args.separate_reader_gate else "legacy"}
            handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[f"{split}_{decision}"] += 1
            if decision == "REOBSERVE" or args.allow_partial_for_preflight:
                # Evidence is selected without answer labels. Labels only supervise the final answer.
                row = {**common, "context": {**context, "phase": "after_reobserve",
                                             "evidence_revisited": True},
                       "evidence": evidence,
                       "target": {"decision": "ANSWER", "frames": target_answer, "observe_frames": []}}
                if args.separate_reader_gate:
                    row['context']['task']='READ'
                    row['target']['frames']=canonical
                    row['training_mode']='READ'
                handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[f"{split}_after_reobserve"] += 1
            counts[f"{split}_questions"] += 1
            if args.allow_partial_for_preflight and index >= 2:
                break
        print(json.dumps({"episode": ep, "counts": dict(counts)}), flush=True)
        # Bound CPU memory while processing all 1,478 videos.
        embedder.cache.clear()
finally:
    for handle in handles.values():
        handle.close()
report = {"scope": "pilot" if args.episodes else f"full official {args.source_split} videos",
          "source_split": args.source_split,
          "source_official_questions": len(rows), "episodes": episodes,
          "held_out_train_video_episodes": sorted(pilot_holdout) if args.source_split == "train" else [],
          "counts": dict(counts), "memory_question_blind": True,
          "retrieval_uses_gold_answers": False, "top_k": 8, "retrieval_method": "action_first_stratified8",
          "evidence_budget": 4, "official_validation_used": question_split == "validation",
          "official_validation_used_for_training": False,
          "reobserve_scope": "historical evidence-frame revisit; physical-navigation training not yet provided",
          "tau_hi_risk_guarantee": False}
report["preflight_only"] = args.allow_partial_for_preflight
report['reader_and_gate_separated']=args.separate_reader_gate
(out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False), flush=True)
