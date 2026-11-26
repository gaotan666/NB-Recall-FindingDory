"""Build question-blind append-only memories on the exact official video frames.

Each episode is an independent append-only JSONL with a completion manifest.
Partial files resume from the last complete frame; no generated past record is
rewritten. Original simulator replays are never substituted for official video.
"""
import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import av
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--shards", type=int, default=3)
parser.add_argument("--episodes", type=int, default=12, help="0 for all train videos")
parser.add_argument("--batch-episodes", type=int, default=4)
parser.add_argument("--max-new-tokens", type=int, default=6144)
parser.add_argument("--merge-for-inference", action="store_true",
                    help="Merge this frozen builder adapter into an in-memory base copy")
parser.add_argument("--cpu-threads", type=int, default=4)
parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
args = parser.parse_args()
torch.set_num_threads(args.cpu_threads)
if not 0 <= args.shard < args.shards:
    raise ValueError("Invalid GPU shard")
data = ROOT / "data/findingdory-subsampled-96"
out = ROOT / "data/stage_c_video_memory/train"
out.mkdir(parents=True, exist_ok=True)
rows = pq.read_table(data / "data/train-00000-of-00001.parquet",
                     columns=["ep_id", "video"]).to_pylist()
videos = {row["ep_id"]: row["video"] for row in rows}
episodes = sorted(videos, key=lambda key: int(key.split("_")[-1]))
if args.episodes:
    episodes = episodes[:args.episodes]
episodes = [ep for i, ep in enumerate(episodes) if i % args.shards == args.shard]
pending = collections.deque(ep for ep in episodes if not (out / ep / "complete.json").exists())
adapter = ROOT / "models/nb_recall_unified_lora_stage_a/adapter"
fingerprint = hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest()
base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(base, use_fast=False)
processor.tokenizer.padding_side = "left"
if args.backend == "vllm":
    from vllm import LLM, SamplingParams
    merged_copy = ROOT / "models/nb_recall_memory_builder_inference_merged"
    if not (merged_copy / "export.json").exists():
        raise RuntimeError("Export the full frozen builder, including its vision LoRA")
    model = LLM(model=str(merged_copy), dtype="bfloat16",
                max_model_len=12288, max_num_seqs=max(32,args.batch_episodes),
                gpu_memory_utilization=.80, enforce_eager=True,
                limit_mm_per_prompt={"image":1}, disable_log_stats=True,
                mm_processor_kwargs={"max_pixels":384*384})
else:
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    if args.merge_for_inference:
        model = model.merge_and_unload().eval()
print(json.dumps({"event":"inference_backend", "merged_frozen_builder":args.merge_for_inference,
                  "backend":args.backend,
                  "batch_episodes":args.batch_episodes,"cpu_threads":args.cpu_threads}),flush=True)
system = (
    "你是NB-Recall追加式记忆建库器。根据当前机器人图像、时间和已有记忆，输出紧凑JSON。"
    "operation只能是APPEND。每个有效观察都新建不可变节点；禁止DROP、MERGE、覆盖或修改旧节点。"
    "重复观察用same-entity连接，状态变化用version-of连接，冲突用conflicts-with连接。"
    "只输出{\"operation\":\"APPEND\",\"nodes\":[...],\"edges\":[...]}。")


def recent(records):
    result = []
    for record in records:
        for node in (record.get("prediction") or {}).get("nodes", []):
            if node.get("type") == "Fact":
                result.append({"id": node.get("id"),
                    "entity": node.get("support", {}).get("scope", {}).get("entity"),
                    "category": node.get("content", {}).get("category"),
                    "observed_at": record["clip_index"],
                    "position": node.get("content", {}).get("estimated_position")})
    return result[-12:]


def parse(raw):
    try:
        payload = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
        if payload.get("operation") != "APPEND":
            return None, "invalid_operation"
        if not isinstance(payload.get("nodes"), list) or not isinstance(payload.get("edges"), list):
            return None, "invalid_schema"
        if any(not isinstance(n, dict) or not n.get("id") or
               n.get("validity", {}).get("t_valid_to") is not None for n in payload["nodes"]):
            return None, "invalid_node"
        return payload, "valid"
    except (ValueError, TypeError, KeyError):
        return None, "invalid_json"


def open_episode(ep):
    folder = out / ep
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "source.json"
    source = {"episode": ep, "video": videos[ep], "split": "train",
              "adapter": str(adapter.relative_to(ROOT)), "adapter_config_sha256": fingerprint,
              "answers_used_for_memory": False, "expected_frames": 96}
    if metadata.exists() and json.loads(metadata.read_text()) != source:
        raise RuntimeError(f"Incompatible memory cache: {folder}")
    if not metadata.exists():
        metadata.write_text(json.dumps(source, indent=2))
    records_path = folder / "observations.jsonl"
    records = [json.loads(line) for line in records_path.open()] if records_path.exists() else []
    if [r["clip_index"] for r in records] != list(range(len(records))):
        raise RuntimeError(f"Non-contiguous memory: {records_path}")
    with av.open(str(data / videos[ep])) as video:
        frames = [frame.to_image() for frame in video.decode(video=0)]
    if len(frames) != 96:
        raise RuntimeError(f"Expected exactly 96 video frames: {ep}, got {len(frames)}")
    return {"ep": ep, "folder": folder, "records": records, "frames": frames,
            "memory": recent(records), "cursor": len(records)}


active = []
started = time.monotonic()
generated_frames = 0
while pending or active:
    while pending and len(active) < args.batch_episodes:
        active.append(open_episode(pending.popleft()))
    # A completed file may have missed only its manifest during an interruption.
    finished = [state for state in active if state["cursor"] == 96]
    for state in finished:
        counts = collections.Counter(r["status"] for r in state["records"])
        (state["folder"] / "complete.json").write_text(json.dumps({
            "frames": 96, "status_counts": dict(counts), "episode": state["ep"],
            "question_blind": True, "memory_append_only": True}, indent=2))
        active.remove(state)
        print(json.dumps({"event": "episode_complete", "episode": state["ep"],
                          "status_counts": dict(counts)}), flush=True)
    if not active:
        continue
    prompts, images = [], []
    from qwen_vl_utils import process_vision_info
    for state in active:
        index = state["cursor"]
        image_path = state["folder"] / f"clip_{index:02d}.jpg"
        if not image_path.exists():
            state["frames"][index].save(image_path, quality=95)
        text = (f"episode={state['ep'].split('_')[-1]}; step={index}; previous_memory="
                + json.dumps(state["memory"], ensure_ascii=False, separators=(",", ":"))
                + ". 追加当前观察。")
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(image_path), "max_pixels": 384*384},
                        {"type": "text", "text": text}]}]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.extend(process_vision_info(messages)[0])
    if args.backend == "vllm":
        requests = [{"prompt":prompt,"multi_modal_data":{"image":image}}
                    for prompt,image in zip(prompts,images)]
        results = model.generate(requests, SamplingParams(temperature=0.0,max_tokens=args.max_new_tokens),
                                 use_tqdm=False)
        texts = [result.outputs[0].text for result in results]
    else:
        batch = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            outputs = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        suffix = outputs[:, batch["input_ids"].shape[1]:]
        texts = processor.batch_decode(suffix, skip_special_tokens=True)
    for state, text in zip(active, texts):
        index = state["cursor"]
        payload, status = parse(text)
        record = {"episode": state["ep"], "clip_index": index,
                  "image": str((state["folder"] / f"clip_{index:02d}.jpg").relative_to(ROOT)),
                  "status": status, "prediction": payload,
                  "raw_answer": text if payload is None else None}
        # Authoritative provenance is this wrapper, never a model-generated hash/scene/time.
        with (state["folder"] / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        state["records"].append(record)
        state["memory"] = recent(state["records"])
        state["cursor"] += 1
    generated_frames += len(active)
    print(json.dumps({"event": "batch", "shard": args.shard,
                      "generated_frames": generated_frames,
                      "seconds": round(time.monotonic()-started, 1),
                      "cursors": {s["ep"]: s["cursor"] for s in active}}), flush=True)
    if args.backend == "transformers":
        del batch, outputs
