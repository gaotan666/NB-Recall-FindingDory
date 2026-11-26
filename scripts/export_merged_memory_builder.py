"""Export a separate frozen inference copy, including vision AND language LoRA.

The older inference runtime cannot load vision-tower adapters dynamically.
All trained adapter tensors are therefore merged and exported in the original
Qwen checkpoint key layout. The source base and both training roles stay intact.
"""
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import Qwen2_5_VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'models/Qwen2.5-VL-3B-Instruct'
adapter=ROOT/'models/nb_recall_unified_lora_stage_a/adapter'
out=ROOT/'models/nb_recall_memory_builder_inference_merged'
out.mkdir(parents=True,exist_ok=True)
if (out/'export.json').exists():
    print('Merged inference copy already complete',flush=True)
    raise SystemExit(0)
torch.set_num_threads(8)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(base,torch_dtype=torch.bfloat16)
model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
model=model.merge_and_unload().eval()
weights={}
for name,tensor in model.state_dict().items():
    name=name.replace('model.language_model.','model.').replace('model.visual.','visual.')
    if name in weights: raise RuntimeError(f'Export key collision: {name}')
    weights[name]=tensor.detach().cpu().contiguous().clone()
expected=set(json.loads((base/'model.safetensors.index.json').read_text())['weight_map'])
# Transformers materializes the tied output embedding alias; the source stores
# only its input-embedding tensor. Remove that alias only after exact equality.
if ('lm_head.weight' not in expected and 'lm_head.weight' in weights
        and model.config.tie_word_embeddings
        and torch.equal(weights['lm_head.weight'], weights['model.embed_tokens.weight'])):
    del weights['lm_head.weight']
if set(weights)!=expected:
    raise RuntimeError(f'Export key mismatch: missing={expected-set(weights)}, extra={set(weights)-expected}')
save_file(weights,out/'model.safetensors',metadata={'format':'pt'})
for name in ['config.json','generation_config.json','preprocessor_config.json',
             'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.json']:
    if (base/name).exists():shutil.copy2(base/name,out/name)
(out/'export.json').write_text(json.dumps({
    'base':str(base),'builder_adapter':str(adapter),'weight_keys':len(weights),
    'includes_vision_lora':True,'includes_language_lora':True,
    'answer_adapter_loaded':False,'source_weights_modified':False},indent=2))
print('Merged inference copy exported with all vision and language adapters',flush=True)
