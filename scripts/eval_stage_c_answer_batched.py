"""Small held-out-train-video policy audit; no simulator SR is claimed here."""
import argparse
import json
import math
import time
import gc
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
parser.add_argument('--batch-size',type=int,default=8)
parser.add_argument('--batch-token-budget',type=int,default=65536)
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
processor.tokenizer.padding_side='left'
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



def parse(raw):
    try:
        return validate_prediction(json.loads(raw[raw.index('{'):raw.rindex('}')+1]))
    except (ValueError,KeyError,TypeError):return None

def generate_batch(batch):
    inputs=outputs=None
    try:
        prompts=[];all_images=[]
        for row in batch:
            messages=make_messages(row)
            images,_=process_vision_info(messages)
            if images:all_images.extend(images)
            prompts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
        inputs=processor(text=prompts,padding=True,return_tensors='pt',**({'images':all_images} if all_images else {})).to('cuda:0')
        width=inputs['input_ids'].shape[1]
        with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=256,do_sample=False)
        raw=processor.batch_decode(outputs[:,width:],skip_special_tokens=True)
        return [(parse(x),x) for x in raw]
    except torch.cuda.OutOfMemoryError:
        inputs=outputs=None;gc.collect();torch.cuda.empty_cache()
        if len(batch)==1:raise
        print(json.dumps({'event':'oom_split','batch_size':len(batch)}),flush=True)
        middle=len(batch)//2
        return generate_batch(batch[:middle])+generate_batch(batch[middle:])

def write_result(row,gate,observed,value,raw,reread=None):
    predicted=value['frames'] if value and value['decision']=='ANSWER' else []
    gold=row['official_answer']
    record={'question_id':row['question_id'],'raw':raw,'reread_raw':reread,
            'valid':value is not None,'prediction':predicted,'gold':gold,
            'exact':predicted==gold,'relaxed_match':relaxed_match(predicted,gold),
            'all_groups_hit':all_destination_groups_hit(predicted,gold,unordered='unordered' in row.get('low_level_category','').lower()),
            'category':row.get('category'),'low_level_category':row.get('low_level_category'),
            'controller_gate':gate,'controller_reobserved':observed,
            'final_decision':value['decision'] if value else None,'inference_backend':'transformers_length_batched'}
    with progress_path.open('a') as handle:
        handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        handle.flush()
    records.append(record)

jobs=[]
for row in rows:
    if row['question_id'] in completed:continue
    tau=max((m['trust']['tau'] for m in row['context']['memory']),default=0)
    gate='ANSWER' if tau>=row['context']['tau_hi'] else 'REOBSERVE' if tau>=row['context']['tau_lo'] else 'ABSTAIN'
    model_row=row
    if args.execution in {'controller','reader_only'}:
        model_row={**row,'context':{**row['context'],'task':'READ'}}
    if args.execution=='controller' and gate=='ABSTAIN':
        value={'decision':'ABSTAIN','frames':[],'observe_frames':[]}
        write_result(row,gate,False,value,json.dumps(value));continue
    observed=args.execution=='controller' and gate=='REOBSERVE'
    if observed:
        evidence=[]
        for memory in row['context']['memory'][:4]:
            i=memory['clip_index'];image=ROOT/'data/stage_c_video_memory'/args.source_split/row['episode']/f'clip_{i:02d}.jpg'
            if image.exists():evidence.append({'clip_index':i,'image':str(image.relative_to(ROOT))})
        model_row={**model_row,'evidence':evidence,'context':{**model_row['context'],'phase':'after_reobserve','evidence_revisited':True}}
    prompt=processor.apply_chat_template(make_messages(model_row),tokenize=False,add_generation_prompt=True)
    length=len(processor.tokenizer(prompt,add_special_tokens=False)['input_ids'])+len(model_row.get('evidence',[]))*512
    jobs.append((length,row,model_row,gate,observed))
jobs.sort(key=lambda job:(bool(job[2].get('evidence')),job[0]))
started=time.monotonic();initial_completed=len(records);cursor=0
while cursor<len(jobs):
    group=[]
    while cursor<len(jobs):
        candidate=jobs[cursor]
        if group and (len(group)>=args.batch_size or candidate[0]*(len(group)+1)>args.batch_token_budget or bool(candidate[2].get('evidence'))!=bool(group[0][2].get('evidence'))):break
        group.append(candidate);cursor+=1
    begin=time.monotonic()
    answers=generate_batch([job[2] for job in group])
    followups=[]
    for job,(value,raw) in zip(group,answers):
        _,row,model_row,gate,observed=job
        if value and value['decision']=='REOBSERVE':
            evidence=[]
            for i in list(dict.fromkeys(value['observe_frames']))[:4]:
                image=ROOT/'data/stage_c_video_memory'/args.source_split/row['episode']/f'clip_{i:02d}.jpg'
                if image.exists():evidence.append({'clip_index':i,'image':str(image.relative_to(ROOT))})
            follow={**model_row,'evidence':evidence,'context':{**model_row['context'],'phase':'after_reobserve','evidence_revisited':True}}
            followups.append((job,raw,follow))
        else:write_result(row,gate,observed,value,raw)
    if followups:
        follow_answers=generate_batch([f[2] for f in followups])
        for (job,raw,follow),(value,reread) in zip(followups,follow_answers):
            write_result(job[1],job[3],job[4],value,raw,reread)
    print(json.dumps({'completed':len(records),'total':len(rows),'batch':len(group),'batch_seconds':round(time.monotonic()-begin,2),
                      'new_questions_per_min':round((len(records)-initial_completed)*60/(time.monotonic()-started),2),
                      'gpu_reserved_gb':round(torch.cuda.memory_reserved()/2**30,2)}),flush=True)

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
