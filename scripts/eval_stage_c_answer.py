"""Small held-out-train-video policy audit; no simulator SR is claimed here."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nb_recall.stage_c import make_messages, validate_prediction, all_destination_groups_hit

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/stage_c_answer_pilot/audit.jsonl")
parser.add_argument("--adapter", default="models/nb_recall_answer_lora_stage_c_pilot/adapter")
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--out", default="reports/stage_c_pilot_audit")
parser.add_argument("--questions-per-shard", type=int, default=8)
parser.add_argument('--execution',choices=['learned_policy','controller','reader_only'],default='learned_policy')
parser.add_argument('--source-split',choices=['train','validation','sim_validation'],default='train')
parser.add_argument('--resume',action='store_true',help='Append completed predictions and resume after interruption')
args = parser.parse_args()
rows = [json.loads(line) for line in (ROOT / args.data).open()]
rows = [r for r in rows if r["context"]["phase"] == "initial"]
if any(r.get('training_mode')=='GATE' for r in rows):
    rows=[r for r in rows if r.get('training_mode')=='GATE']
if args.source_split == 'train':
    random.Random(20261122).shuffle(rows)
rows = rows if args.shard == -1 else rows[args.shard::3]
if args.questions_per_shard:
    rows = rows[:args.questions_per_shard]
base = ROOT / "models/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(base, use_fast=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
model = PeftModel.from_pretrained(model, ROOT / args.adapter, is_trainable=False).eval()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
progress_path=out / f'shard_{args.shard}.jsonl'
if progress_path.exists() and not args.resume:
    raise RuntimeError(f'Existing prediction file requires --resume: {progress_path}')
records=[json.loads(line) for line in progress_path.open()] if progress_path.exists() else []
completed={record['question_id'] for record in records}
if not completed.issubset({row['question_id'] for row in rows}):
    raise RuntimeError('Resume file contains questions outside this evaluation set')


def relaxed_match(predicted,gold):
    if len(predicted)!=len(gold):return 0.0
    scores=[]
    for selected,valid in zip(predicted,gold):
        if not selected and not valid:scores.append(1.0)
        elif not selected or not valid:scores.append(0.0)
        else:scores.append(sum(frame in valid for frame in selected)/len(selected))
    return math.prod(scores)


def generate(row):
    messages = make_messages(row)
    images, _ = process_vision_info(messages)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt", **({"images": images} if images else {})).to("cuda:0")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        value = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
        return validate_prediction(value), raw
    except (ValueError, KeyError, TypeError):
        return None, raw


for row in rows:
    if row['question_id'] in completed:continue
    tau=max((m['trust']['tau'] for m in row['context']['memory']),default=0)
    gate=('ANSWER' if tau>=row['context']['tau_hi'] else
          'REOBSERVE' if tau>=row['context']['tau_lo'] else 'ABSTAIN')
    controller_observed=False
    model_row=row
    if args.execution in {'controller','reader_only'}:
        model_row={**row,'context':{**row['context'],'task':'READ'}}
    if args.execution=='controller' and gate=='ABSTAIN':
        value={'decision':'ABSTAIN','frames':[],'observe_frames':[]}
        raw=json.dumps(value)
    else:
        if args.execution=='controller' and gate=='REOBSERVE':
            evidence=[]
            for memory in row['context']['memory'][:4]:
                index=memory['clip_index']
                image=ROOT/'data/stage_c_video_memory'/args.source_split/row['episode']/f'clip_{index:02d}.jpg'
                if image.exists():evidence.append({'clip_index':index,'image':str(image.relative_to(ROOT))})
            model_row={**model_row,'evidence':evidence,
                       'context':{**model_row['context'],'phase':'after_reobserve','evidence_revisited':True}}
            controller_observed=True
        value,raw=generate(model_row)
    reread = None
    if value and value["decision"] == "REOBSERVE":
        evidence = []
        for index in list(dict.fromkeys(value["observe_frames"]))[:4]:
            image = ROOT / "data/stage_c_video_memory" / args.source_split / row["episode"] / f"clip_{index:02d}.jpg"
            if image.exists():
                evidence.append({"clip_index": index, "image": str(image.relative_to(ROOT))})
        followup = {**model_row, "evidence": evidence,
                    "context": {**model_row["context"], "phase": "after_reobserve", "evidence_revisited": True}}
        value, reread = generate(followup)
    predicted = value["frames"] if value and value["decision"] == "ANSWER" else []
    gold = row["official_answer"]
    exact = predicted == gold
    hit=all_destination_groups_hit(predicted,gold,
                                  unordered='unordered' in row.get('low_level_category','').lower())
    records.append({"question_id": row["question_id"], "raw": raw, "reread_raw": reread,
                    "valid": value is not None, "prediction": predicted, "gold": gold,
                    "exact": exact, "relaxed_match":relaxed_match(predicted,gold),
                    "all_groups_hit": hit,
                    "category":row.get('category'),"low_level_category":row.get('low_level_category'),
                    'controller_gate':gate,'controller_reobserved':controller_observed,
                    'final_decision':value['decision'] if value else None})
    with progress_path.open('a') as handle:
        handle.write(json.dumps(records[-1],ensure_ascii=False)+'\n')
        handle.flush()
    if args.source_split == 'train' or len(records)%25==0:
        print(json.dumps({'completed':len(records),'total':len(rows),
                          'all_groups_hit':sum(r['all_groups_hit'] for r in records),
                          'valid':sum(r['valid'] for r in records)}),flush=True)
report = {"scope": f"official {args.source_split} videos; offline frame metrics only",
          "source_split":args.source_split,"adapter":args.adapter,
          'execution':args.execution,
          "shard": args.shard, "questions": len(records), "records": records,
          "valid": sum(r["valid"] for r in records), "exact": sum(r["exact"] for r in records),
          "avg_relaxed_match":sum(r['relaxed_match'] for r in records)/len(records) if records else 0,
          "all_groups_hit": sum(r["all_groups_hit"] for r in records),
          "simulator_evaluation": False, "physical_reobservation": False}
if len(records)!=len(rows) or len({r['question_id'] for r in records})!=len(rows):
    raise RuntimeError('Incomplete or duplicate evaluation predictions')
report['category_breakdown']={}
for category in sorted({r['category'] for r in records}):
    subset=[r for r in records if r['category']==category]
    report['category_breakdown'][category]={
        'questions':len(subset), 'exact':sum(r['exact'] for r in subset),
        'all_groups_hit':sum(r['all_groups_hit'] for r in subset),
        'avg_relaxed_match':sum(r['relaxed_match'] for r in subset)/len(subset)}
report['low_level_breakdown']={}
for category in sorted({r['low_level_category'] for r in records}):
    subset=[r for r in records if r['low_level_category']==category]
    report['low_level_breakdown'][category]={
        'questions':len(subset), 'exact':sum(r['exact'] for r in subset),
        'all_groups_hit':sum(r['all_groups_hit'] for r in subset),
        'avg_relaxed_match':sum(r['relaxed_match'] for r in subset)/len(subset)}
(out / f"shard_{args.shard}.json").write_text(json.dumps(report, indent=2))
