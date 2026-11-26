"""Expand selected frames without changing selection, trust scores or gold labels."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for split,source,dest in [
 ('validation','data/stage_c_answer_validation_700_action8','data/stage_c_answer_validation_700_framepack'),
 ('sim_validation','data/stage_c_answer_sim_validation_ep_38_action8','data/stage_c_answer_sim_validation_ep_38_framepack')]:
 out=ROOT/dest;out.mkdir(parents=True,exist_ok=True)
 cache={};count=0
 with (out/'audit.jsonl.tmp').open('w') as target:
  for line in (ROOT/source/'audit.jsonl').open():
   row=json.loads(line)
   if row.get('training_mode')!='GATE' or row['context']['phase']!='initial':continue
   ep=row['episode']
   if ep not in cache:
    cache[ep]=list(map(json.loads,(ROOT/'data/stage_c_video_memory'/split/ep/'observations.jsonl').open()))
    assert [o['clip_index'] for o in cache[ep]]==list(range(96))
   before=[(m['clip_index'],m['trust']) for m in row['context']['memory']]
   for memory in row['context']['memory']:
    obs=cache[ep][memory['clip_index']];payload=obs.get('prediction') or {};nodes=payload.get('nodes',[])
    memory['frame_evidence']={
       'facts':[{'id':n['id'],'content':n.get('content',{}),'entity':n.get('support',{}).get('scope',{}).get('entity')} for n in nodes if n.get('type')=='Fact'],
       'events':[{'id':n['id'],'content':n.get('content',{})} for n in nodes if n.get('type')=='Event'],
       'entities':[{'id':n['id'],'content':n.get('content',{})} for n in nodes if n.get('type')=='Entity'],
       'relations':payload.get('edges',[]),
       'source_image':obs['image']}
   assert before==[(m['clip_index'],m['trust']) for m in row['context']['memory']]
   target.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');count+=1
 (out/'audit.jsonl.tmp').replace(out/'audit.jsonl')
 (out/'report.json').write_text(json.dumps({'questions':count,'source_split':split,'source':source,'frozen_selection_and_trust':True,'expansion':'all Fact, Event, Entity content and all edges from each selected observation; no ground-truth data introduced'},indent=2))
 print(split,count,flush=True)
