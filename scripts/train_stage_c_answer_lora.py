"""DDP training of a NEW answer LoRA on the frozen original Qwen base."""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nb_recall.adapter_roles import load_and_validate_config
from nb_recall.stage_c import make_messages, validate_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/stage_c_answer_pilot/train.jsonl")
parser.add_argument("--run-name", default="nb_recall_answer_lora_stage_c_pilot")
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--gradient-accumulation", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--reader-fraction",type=float,default=0.7)
args = parser.parse_args()
config = load_and_validate_config(ROOT / "configs/nb_recall_fast_pipeline.json")
out = ROOT / "models" / args.run_name
if out.resolve().is_relative_to((ROOT / config["stage_a"]["adapter"]).parent.resolve()):
    raise ValueError("Cannot overwrite the memory-builder adapter")
local = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
if world > 1:
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
torch.cuda.set_device(local)
device = torch.device("cuda", local)
torch.manual_seed(20261122)
rows = [json.loads(line) for line in (ROOT / args.data).open()]
if not rows:
    raise RuntimeError("No answer-policy training rows")
for row in rows:
    validate_prediction(row["target"])
    if row["split"] != "train":
        raise ValueError("Held-out row entered training")
order = list(range(len(rows)))
random.Random(20261122).shuffle(order)
mode_indices={name:[i for i,row in enumerate(rows) if row.get('training_mode')==name]
              for name in ['READ','GATE']}
gate_classes={name:[i for i in mode_indices['GATE'] if rows[i]['target']['decision']==name]
              for name in ['ANSWER','REOBSERVE','ABSTAIN']}
gate_classes={name:indices for name,indices in gate_classes.items() if indices}
sampling_rng=random.Random(20261122)
if mode_indices['READ'] and mode_indices['GATE']:
    # A bounded deterministic schedule avoids drowning frame supervision in short ABSTAIN labels.
    schedule=[]
    for index in range(max(len(rows), (args.max_steps or 1)*world*args.gradient_accumulation)):
        if index%10 < round(args.reader_fraction*10):
            schedule.append(sampling_rng.choice(mode_indices['READ']))
        else:
            decision=list(gate_classes)[(index//10)%len(gate_classes)]
            schedule.append(sampling_rng.choice(gate_classes[decision]))
    order=schedule
base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(base, use_fast=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
# No PeftModel.from_pretrained: this role always starts with fresh LoRA matrices.
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))
model.to(device).train()
if world > 1:
    print(json.dumps({"event":"ddp_initialization", "rank":rank}),flush=True)
    model = DDP(model, device_ids=[local], find_unused_parameters=True)
    print(json.dumps({"event":"ddp_ready", "rank":rank}),flush=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=args.learning_rate, weight_decay=0.01, fused=True)
steps = args.max_steps or math.ceil(len(rows)/(world*args.gradient_accumulation))
history = []
if rank == 0:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter").exists():
        raise RuntimeError(f"Refusing to overwrite a completed answer adapter: {out}")
    print(json.dumps({"event": "stage_c_start", "rows": len(rows), "steps": steps,
                      "world_size": world, "init": "original_base_fresh_answer_lora"}), flush=True)


def save(step, final=False):
    if world > 1:
        dist.barrier()
    if rank == 0:
        core = model.module if world > 1 else model
        core.save_pretrained(out / ("adapter" if final else f"checkpoint-{step}"))
        if final:
            processor.save_pretrained(out / "processor")
        (out / "run.json").write_text(json.dumps({
            "adapter_role": "answer", "initialized_from": str(base),
            "memory_builder_weights_loaded": False, "official_sft_used": False,
            "data": args.data, "train_rows": len(rows), "steps_completed": step,
            "planned_steps": steps, "complete": final, "world_size": world,
            "effective_batch": world*args.gradient_accumulation,
            "learning_rate": args.learning_rate, "lora_r": 64,
            "reader_fraction":args.reader_fraction,
            "reader_gate_mode_counts":{name:len(indices) for name,indices in mode_indices.items()},
            "history": history}, ensure_ascii=False, indent=2))
    if world > 1:
        dist.barrier()


optimizer.zero_grad(set_to_none=True)
for step in range(1, steps+1):
    start = time.monotonic()
    loss_sum = torch.zeros((), device=device)
    for micro in range(args.gradient_accumulation):
        index = ((step-1)*args.gradient_accumulation+micro)*world+rank
        row = rows[order[index % len(order)]]
        messages = make_messages(row)
        images, _ = process_vision_info(messages)
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = prefix + json.dumps(row["target"], separators=(",", ":")) + "<|im_end|>"
        kwargs = {"images": images} if images else {}
        batch = processor(text=[full], return_tensors="pt", **kwargs).to(device)
        prefix_len = processor(text=[prefix], return_tensors="pt", **kwargs)["input_ids"].shape[1]
        labels = batch["input_ids"].clone()
        labels[:, :prefix_len] = -100
        if not (labels != -100).any():
            raise RuntimeError("Answer targets are fully masked")
        sync = micro == args.gradient_accumulation-1
        sync_context = contextlib.nullcontext() if world == 1 or sync else model.no_sync()
        with sync_context:
            loss = model(**batch, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Stage-C loss")
            (loss/args.gradient_accumulation).backward()
        loss_sum += loss.detach()
        del batch, labels, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss_sum /= args.gradient_accumulation
    if world > 1:
        dist.all_reduce(loss_sum)
        loss_sum /= world
    if rank == 0:
        record = {"step": step, "loss": float(loss_sum.item()),
                  "seconds": round(time.monotonic()-start, 2),
                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device)/1e9, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
    if step % args.save_every == 0:
        save(step)
save(steps, final=True)
if world > 1:
    dist.destroy_process_group()
