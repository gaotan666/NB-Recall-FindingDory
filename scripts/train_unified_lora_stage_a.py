"""Three-GPU Stage-A training for the append-only memory-builder LoRA.

This adapter has one role only: converting observations into immutable memory
nodes and relations.  It is frozen after Stage A and is never resumed by the
Stage-C answer-policy trainer.
"""
import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/stage_a_append_only/train.jsonl")
parser.add_argument("--run-name", default="nb_recall_memory_builder_lora_stage_a")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--max-steps", type=int, default=0,
                    help="Optimizer steps; 0 means full requested epochs")
parser.add_argument("--gradient-accumulation", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--max-pixels", type=int, default=384 * 384)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--seed", type=int, default=20261119)
args = parser.parse_args()

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if world_size > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.manual_seed(args.seed + rank)
random.seed(args.seed + rank)

data_path = ROOT / args.data
rows = [json.loads(line) for line in data_path.open()]
if not rows:
    raise RuntimeError(f"No Stage A rows in {data_path}")
for row in rows:
    target = row.get("target", {})
    if target.get("operation") != "APPEND":
        raise ValueError("Stage A data violated append-only operation")
    if not row.get("invariants", {}).get("append_only"):
        raise ValueError("Stage A row lacks append-only invariant")
order = list(range(len(rows)))
random.Random(args.seed).shuffle(order)

base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
out = ROOT / "models" / args.run_name
if rank == 0:
    out.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(base, use_fast=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))
model.to(device)
model.train()
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=args.learning_rate, weight_decay=0.01, fused=True)
steps_per_epoch = math.ceil(len(rows) / (world_size * args.gradient_accumulation))
max_steps = args.max_steps or steps_per_epoch * args.epochs

system = (
    "你是NB-Recall追加式记忆建库器。根据当前机器人图像、时间和已有记忆，输出紧凑JSON。"
    "operation只能是APPEND。每个有效观察都新建不可变节点；禁止DROP、MERGE、覆盖或修改旧节点。"
    "重复观察用same-entity连接，状态变化用version-of连接，冲突用conflicts-with连接。"
    "只输出{\"operation\":\"APPEND\",\"nodes\":[...],\"edges\":[...]}。")


def make_batch(row):
    image_path = ROOT / row["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    previous = json.dumps(row["previous_memory"], ensure_ascii=False,
                          separators=(",", ":"))
    user_text = (f"episode={row['episode']}; step={row['step']}; "
                 f"previous_memory={previous}. 追加当前观察。")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path), "max_pixels": args.max_pixels},
            {"type": "text", "text": user_text},
        ]},
    ]
    images, _ = process_vision_info(messages)
    prefix = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
    full = prefix + target + "<|im_end|>"
    batch = processor(text=[full], images=images, padding=True,
                      return_tensors="pt").to(device)
    prefix_batch = processor(text=[prefix], images=images, padding=True,
                             return_tensors="pt")
    prefix_len = int(prefix_batch["attention_mask"].sum())
    labels = batch["input_ids"].clone()
    labels[:, :prefix_len] = -100
    if not (labels != -100).any():
        raise RuntimeError("Assistant target was fully masked")
    return batch, labels


if rank == 0:
    print(json.dumps({"event": "stage_a_start", "base": str(base),
                      "official_sft_used": False, "rows": len(rows),
                      "world_size": world_size, "trainable_parameters": trainable,
                      "total_parameters": total, "optimizer_steps": max_steps,
                      "effective_batch": world_size * args.gradient_accumulation,
                      "append_only": True}, ensure_ascii=False), flush=True)

history = []
optimizer.zero_grad(set_to_none=True)
for step in range(1, max_steps + 1):
    started = time.monotonic()
    loss_sum = torch.zeros((), device=device)
    for micro in range(args.gradient_accumulation):
        logical = ((step - 1) * args.gradient_accumulation + micro) * world_size + rank
        row = rows[order[logical % len(order)]]
        batch, labels = make_batch(row)
        sync = micro == args.gradient_accumulation - 1
        sync_context = contextlib.nullcontext() if sync or world_size == 1 else model.no_sync()
        with sync_context:
            loss = model(**batch, labels=labels).loss
            (loss / args.gradient_accumulation).backward()
        loss_sum += loss.detach()
        del batch, labels, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mean_loss = loss_sum / args.gradient_accumulation
    if world_size > 1:
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= world_size
    record = {"step": step, "loss": round(float(mean_loss.item()), 6),
              "seconds": round(time.monotonic() - started, 2),
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)}
    if rank == 0:
        history.append(record)
        print(json.dumps(record), flush=True)
    if args.save_every and step % args.save_every == 0:
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            unwrapped = model.module if isinstance(model, DDP) else model
            unwrapped.save_pretrained(out / f"checkpoint-{step}")
            (out / "run.json").write_text(json.dumps({
                "method": "NB-Recall Stage A append-only memory-builder LoRA",
                "adapter_role": "memory_builder",
                "frozen_after_stage_a": True,
                "base": str(base), "official_sft_used": False,
                "append_only": True, "steps_completed": step,
                "train_examples": len(rows), "world_size": world_size,
                "gradient_accumulation": args.gradient_accumulation,
                "history": history}, ensure_ascii=False, indent=2))
        if world_size > 1:
            dist.barrier()

if world_size > 1:
    dist.barrier()
if rank == 0:
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "processor")
    (out / "run.json").write_text(json.dumps({
        "method": "NB-Recall Stage A append-only memory-builder LoRA",
        "adapter_role": "memory_builder",
        "frozen_after_stage_a": True,
        "base": str(base), "official_sft_used": False,
        "append_only": True, "steps_completed": max_steps,
        "train_examples": len(rows), "world_size": world_size,
        "gradient_accumulation": args.gradient_accumulation,
        "history": history}, ensure_ascii=False, indent=2))
    print(json.dumps({"event": "stage_a_complete", "adapter": str(out / "adapter")}),
          flush=True)
if world_size > 1:
    dist.destroy_process_group()
