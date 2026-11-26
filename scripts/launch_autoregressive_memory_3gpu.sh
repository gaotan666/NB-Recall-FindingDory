#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
split="${1:-audit}"
adapter="${2:-models/nb_recall_unified_lora_stage_a/adapter}"
batch_episodes="${BATCH_EPISODES:-4}"
max_episodes="${MAX_EPISODES_PER_SHARD:-0}"
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$shard" envs/model/bin/python \
    scripts/generate_autoregressive_memory_shard.py \
    --split "$split" --shard-index "$shard" --num-shards 3 \
    --batch-episodes "$batch_episodes" --max-episodes "$max_episodes" \
    --adapter "$adapter" --resume \
    > "logs/autoregressive_${split}_shard${shard}.log" 2>&1 &
done
wait
