"""Generate append-only Stage-A memories in parallel episode shards.

One pass from the frozen memory-builder adapter serves three downstream uses:
Stage-A autoregressive audit, Stage-B Trust supervision, and Stage-C answer-
adapter memory context construction.  Each process owns
whole episodes, so a frame always consumes only that episode's earlier model
outputs.  Independent episode streams are batched on one GPU.
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "calibration", "audit"], required=True)
parser.add_argument("--shard-index", type=int, required=True)
parser.add_argument("--num-shards", type=int, default=3)
parser.add_argument("--adapter", default="models/nb_recall_unified_lora_stage_a/adapter")
parser.add_argument("--out", default="data/autoregressive_memory")
parser.add_argument("--batch-episodes", type=int, default=4)
parser.add_argument("--max-new-tokens", type=int, default=4096)
parser.add_argument("--max-episodes", type=int, default=0)
parser.add_argument("--resume", action="store_true",
                    help="Resume an interrupted shard from its existing JSONL")
args = parser.parse_args()
if not 0 <= args.shard_index < args.num_shards:
    raise ValueError("Invalid shard index")

data = ROOT / "data/stage_a_append_only" / f"{args.split}.jsonl"
by_episode = collections.defaultdict(list)
for line in data.open():
    row = json.loads(line)
    if int(row["episode"]) % args.num_shards == args.shard_index:
        by_episode[int(row["episode"])].append(row)
episodes = sorted(by_episode)
if args.max_episodes:
    episodes = episodes[:args.max_episodes]
for episode in episodes:
    by_episode[episode].sort(key=lambda row: int(row["step"]))
if not episodes:
    raise RuntimeError("Shard contains no episodes")

base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(base, use_fast=False)
processor.tokenizer.padding_side = "left"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
model = PeftModel.from_pretrained(model, ROOT / args.adapter).eval()
system = (
    "你是NB-Recall追加式记忆建库器。根据当前机器人图像、时间和已有记忆，输出紧凑JSON。"
    "operation只能是APPEND。每个有效观察都新建不可变节点；禁止DROP、MERGE、覆盖或修改旧节点。"
    "重复观察用same-entity连接，状态变化用version-of连接，冲突用conflicts-with连接。"
    "只输出{\"operation\":\"APPEND\",\"nodes\":[...],\"edges\":[...]}。")


def parse_answer(answer):
    try:
        payload = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None, "invalid_json"
    if payload.get("operation") != "APPEND":
        return None, "not_append"
    nodes, edges = payload.get("nodes"), payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None, "invalid_schema"
    ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if None in ids or len(ids) != len(nodes):
        return None, "invalid_node_ids"
    for node in nodes:
        validity = node.get("validity", {})
        if validity.get("t_valid_to") is not None:
            return None, "old_state_mutation"
    return payload, "valid"


def summary_from_payload(payload, previous):
    updated = list(previous)
    for node in payload.get("nodes", []):
        if node.get("type") != "Fact":
            continue
        content = node.get("content", {})
        scope = node.get("support", {}).get("scope", {})
        provenance = node.get("provenance") or [{}]
        updated.append({"id": node.get("id"), "entity": scope.get("entity"),
                        "category": content.get("category"),
                        "observed_at": provenance[0].get("time"),
                        "position": content.get("estimated_position")})
    return updated[-12:]


def make_message(row, previous):
    previous_text = json.dumps(previous, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(ROOT / row["image"]),
             "max_pixels": 384 * 384},
            {"type": "text", "text":
             f"episode={row['episode']}; step={row['step']}; "
             f"previous_memory={previous_text}. 追加当前观察。"},
        ]},
    ]


out_dir = ROOT / args.out / args.split
out_dir.mkdir(parents=True, exist_ok=True)
output_path = out_dir / f"shard_{args.shard_index:02d}.jsonl"
report_path = out_dir / f"shard_{args.shard_index:02d}_report.json"
states = {episode: {"cursor": 0, "memory": []} for episode in episodes}
counts = collections.Counter()
if args.resume and output_path.is_file():
    with output_path.open() as existing:
        for line_number, line in enumerate(existing, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt resume file {output_path}:{line_number}") from exc
            episode = int(record["episode"])
            if episode not in states:
                raise RuntimeError(f"Resume record belongs to unexpected episode {episode}")
            state = states[episode]
            expected = by_episode[episode][state["cursor"]]
            if int(record["step"]) != int(expected["step"]):
                raise RuntimeError(
                    f"Resume order mismatch for episode {episode}: "
                    f"got {record['step']}, expected {expected['step']}")
            payload = record.get("prediction")
            if payload is not None:
                state["memory"] = summary_from_payload(payload, state["memory"])
            state["cursor"] += 1
            counts[record["status"]] += 1
            counts["frames"] += 1
    counts["episodes"] = sum(
        state["cursor"] >= len(by_episode[episode])
        for episode, state in states.items())
    print(json.dumps({"event": "resumed", "shard": args.shard_index,
                      **counts}), flush=True)

pending = collections.deque(
    episode for episode in episodes
    if states[episode]["cursor"] < len(by_episode[episode]))
active = []

with output_path.open("a" if args.resume else "w") as output:
    while pending or active:
        while pending and len(active) < args.batch_episodes:
            active.append(pending.popleft())
        batch_episodes = [episode for episode in active
                          if states[episode]["cursor"] < len(by_episode[episode])]
        if not batch_episodes:
            break
        rows = [by_episode[episode][states[episode]["cursor"]] for episode in batch_episodes]
        messages = [make_message(row, states[episode]["memory"])
                    for episode, row in zip(batch_episodes, rows)]
        prompts, images = [], []
        for message in messages:
            image_inputs, _ = process_vision_info(message)
            images.extend(image_inputs)
            prompts.append(processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=prompts, images=images, padding=True,
                           return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
        suffix = generated[:, inputs["input_ids"].shape[1]:]
        answers = processor.batch_decode(suffix, skip_special_tokens=True)
        for episode, row, answer in zip(batch_episodes, rows, answers):
            payload, status = parse_answer(answer)
            counts[status] += 1
            record = {"episode": episode, "scene": row["scene"], "step": row["step"],
                      "image": row["image"], "status": status,
                      "prediction": payload, "raw_answer": answer if payload is None else None}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            if payload is not None:
                states[episode]["memory"] = summary_from_payload(
                    payload, states[episode]["memory"])
            states[episode]["cursor"] += 1
            counts["frames"] += 1
        finished = [episode for episode in active
                    if states[episode]["cursor"] >= len(by_episode[episode])]
        for episode in finished:
            active.remove(episode)
            counts["episodes"] += 1
        if counts["frames"] % 20 == 0:
            print(json.dumps({"shard": args.shard_index, **counts}), flush=True)

report = {"split": args.split, "shard_index": args.shard_index,
          "num_shards": args.num_shards, "batch_episodes": args.batch_episodes,
          "adapter": args.adapter, "counts": dict(counts),
          "append_only_valid_rate": (counts["valid"] / counts["frames"]
                                     if counts["frames"] else 0.0),
          "adapter_role": "memory_builder",
          "reused_for": ["Stage A autoregressive audit", "Stage B Trust supervision",
                         "Stage C answer-LoRA memory context"]}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False), flush=True)
