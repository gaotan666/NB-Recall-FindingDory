"""Continuous-batching memory building with independent sequential episode streams.

When one frame finishes, its episode immediately submits the next frame. Other
episodes do not wait for the longest JSON in a fixed batch. The frozen model,
prompt, 96-frame coverage and append-only cache contract are unchanged.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('VLLM_USE_V1','0')
import av
import pyarrow.parquet as pq
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--shard',type=int,required=True)
parser.add_argument('--shards',type=int,default=3)
parser.add_argument('--split',choices=['train','validation'],default='train')
parser.add_argument('--memory-split',default=None,
                    help='Separate output namespace, for example sim_validation')
parser.add_argument('--episode-id',default=None,help='Generate one ep_# from the source split')
parser.add_argument('--video-override-dir',default=None,
                    help='Directory of episode MP4s from an official simulator replay')
parser.add_argument('--episodes',type=int,default=3,help='0 means all videos in the selected split')
parser.add_argument('--concurrency',type=int,default=64)
parser.add_argument('--gpu-memory-utilization',type=float,default=.8)
parser.add_argument('--max-new-tokens',type=int,default=6144)
args=parser.parse_args()
if not 0<=args.shard<args.shards:raise ValueError('Invalid episode shard')
data=ROOT/'data/findingdory-subsampled-96'
memory_split=args.memory_split or args.split
out=ROOT/'data/stage_c_video_memory'/memory_split
adapter=ROOT/'models/nb_recall_unified_lora_stage_a/adapter'
model_dir=ROOT/'models/nb_recall_memory_builder_inference_merged'
if not (model_dir/'export.json').exists():raise RuntimeError('Missing complete merged builder export')
fingerprint=hashlib.sha256((adapter/'adapter_config.json').read_bytes()).hexdigest()
rows=pq.read_table(data/f'data/{args.split}-00000-of-00001.parquet',columns=['ep_id','video']).to_pylist()
videos={row['ep_id']:row['video'] for row in rows}
episodes=sorted(videos,key=lambda key:int(key.split('_')[-1]))
if args.episode_id:
    if args.episode_id not in videos:raise ValueError('Unknown episode id')
    episodes=[args.episode_id]
elif args.episodes:
    episodes=episodes[:args.episodes]
episodes=[ep for i,ep in enumerate(episodes) if i%args.shards==args.shard]
processor=AutoProcessor.from_pretrained(ROOT/'models/Qwen2.5-VL-3B-Instruct',use_fast=False)
engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=str(model_dir),dtype='bfloat16',max_model_len=12288,
    max_num_seqs=max(32,args.concurrency),gpu_memory_utilization=args.gpu_memory_utilization,
    enforce_eager=False,limit_mm_per_prompt={'image':1},disable_log_stats=True,
    disable_log_requests=True,
    mm_processor_kwargs={'max_pixels':384*384}))
sampling=SamplingParams(temperature=0,max_tokens=args.max_new_tokens)
system=('你是NB-Recall追加式记忆建库器。根据当前机器人图像、时间和已有记忆，输出紧凑JSON。'
        'operation只能是APPEND。每个有效观察都新建不可变节点；禁止DROP、MERGE、覆盖或修改旧节点。'
        '重复观察用same-entity连接，状态变化用version-of连接，冲突用conflicts-with连接。'
        '只输出{"operation":"APPEND","nodes":[...],"edges":[...]}。')
started=time.monotonic()
counters=collections.Counter()


def memory_summary(records):
    result=[]
    for record in records:
        for node in (record.get('prediction') or {}).get('nodes',[]):
            if node.get('type')=='Fact':
                result.append({'id':node.get('id'),
                    'entity':node.get('support',{}).get('scope',{}).get('entity'),
                    'category':node.get('content',{}).get('category'),
                    'observed_at':record['clip_index'],
                    'position':node.get('content',{}).get('estimated_position')})
    return result[-12:]


def decode_video(path):
    with av.open(str(path)) as container:
        frames=[frame.to_image() for frame in container.decode(video=0)]
    if len(frames)!=96:raise RuntimeError(f'Expected 96 original video frames: {path}')
    return frames


def parse(raw):
    try:
        value=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        if value.get('operation')!='APPEND':return None,'invalid_operation'
        if not isinstance(value.get('nodes'),list) or not isinstance(value.get('edges'),list):
            return None,'invalid_schema'
        if any(not isinstance(n,dict) or not n.get('id') or
               n.get('validity',{}).get('t_valid_to') is not None for n in value['nodes']):
            return None,'invalid_node'
        return value,'valid'
    except (ValueError,TypeError,KeyError):return None,'invalid_json'


async def episode_stream(ep,semaphore):
    async with semaphore:
        folder=out/ep
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():return
        video_path=(ROOT/args.video_override_dir/f'{ep}.mp4') if args.video_override_dir else data/videos[ep]
        source_video=str(video_path.relative_to(ROOT))
        source={'episode':ep,'video':source_video,'split':memory_split,
                'adapter':str(adapter.relative_to(ROOT)),'adapter_config_sha256':fingerprint,
                'answers_used_for_memory':False,'expected_frames':96}
        metadata=folder/'source.json'
        if metadata.exists() and json.loads(metadata.read_text())!=source:
            raise RuntimeError(f'Incompatible frame-memory provenance: {folder}')
        if not metadata.exists():metadata.write_text(json.dumps(source,indent=2))
        path=folder/'observations.jsonl'
        records=[json.loads(line) for line in path.open()] if path.exists() else []
        if [r['clip_index'] for r in records]!=list(range(len(records))):
            raise RuntimeError(f'Non-contiguous episode cache: {ep}')
        frames=await asyncio.to_thread(decode_video,video_path)
        for index in range(len(records),96):
            image_path=folder/f'clip_{index:02d}.jpg'
            if not image_path.exists():frames[index].save(image_path,quality=95)
            previous=json.dumps(memory_summary(records),ensure_ascii=False,separators=(',',':'))
            text=f"episode={ep.split('_')[-1]}; step={index}; previous_memory={previous}. 追加当前观察。"
            messages=[{'role':'system','content':[{'type':'text','text':system}]},
                {'role':'user','content':[{'type':'image','image':str(image_path),'max_pixels':384*384},
                                        {'type':'text','text':text}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,_=process_vision_info(messages)
            final=None
            async for result in engine.generate(
                    {'prompt':prompt,'multi_modal_data':{'image':images[0]}},sampling,
                    request_id=f'{ep}:{index}'):
                final=result
            if final is None:raise RuntimeError(f'No engine result for {ep}:{index}')
            raw=final.outputs[0].text
            payload,status=parse(raw)
            record={'episode':ep,'clip_index':index,'image':str(image_path.relative_to(ROOT)),
                    'status':status,'prediction':payload,'raw_answer':raw if payload is None else None}
            with path.open('a') as handle:
                handle.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
                handle.flush();os.fsync(handle.fileno())
            records.append(record)
            counters['new_frames']+=1;counters[status]+=1
            print(json.dumps({'event':'frame','episode':ep,'clip_index':index,'status':status,
                              'counts':dict(counters),'seconds':round(time.monotonic()-started,1)}),flush=True)
        count=collections.Counter(r['status'] for r in records)
        (folder/'complete.json').write_text(json.dumps({'frames':96,'status_counts':dict(count),
            'episode':ep,'question_blind':True,'memory_append_only':True},indent=2))
        print(json.dumps({'event':'episode_complete','episode':ep,'counts':dict(count)}),flush=True)


async def main():
    semaphore=asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(episode_stream(ep,semaphore) for ep in episodes))


try:
    asyncio.run(main())
finally:
    engine.shutdown_background_loop()
