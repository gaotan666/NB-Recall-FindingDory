#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
envs/model/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/train_unified_lora_stage_a.py \
  --data data/stage_a_append_only/train.jsonl \
  --run-name nb_recall_unified_lora_stage_a \
  --epochs 2 --gradient-accumulation 16 \
  --learning-rate 1e-4 --save-every 300
