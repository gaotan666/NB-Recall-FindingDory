#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD/vendor/findingdory-habitat:$PWD"
run=reports/stage_c_validation_3100_framepack
mkdir -p "$run/hlsr_official" logs/hlsr_official_3100_framepack
if [[ ! -s "$run/shard_-1.json" ]]; then
 envs/model/bin/python -u scripts/eval_stage_c_answer_batched.py --source-split validation --data data/stage_c_answer_validation_700_framepack/audit.jsonl --adapter models/nb_recall_answer_lora_stage_c/checkpoint-3100 --execution controller --shard -1 --questions-per-shard 0 --resume --out "$run" >> logs/framepack3100_answer_gpu3.log 2>&1
fi
envs/model/bin/python scripts/prepare_hlsr_cached_predictions_framepack.py --step 3100
if [[ ! -s "$run/sim_replay_answer_ep_38/shard_-1.json" ]]; then
 envs/model/bin/python -u scripts/eval_stage_c_answer_batched.py --source-split sim_validation --data data/stage_c_answer_sim_validation_ep_38_framepack/audit.jsonl --adapter models/nb_recall_answer_lora_stage_c/checkpoint-3100 --execution controller --shard -1 --questions-per-shard 0 --resume --out "$run/sim_replay_answer_ep_38" >> logs/framepack3100_replay38.log 2>&1
fi
envs/model/bin/python scripts/merge_hlsr_sim_replay_predictions_framepack.py --step 3100 --episode 38
for episode in $(seq 1 100); do
 [[ -s "$run/hlsr_official/ep_${episode}.json" ]] && continue
 extra=()
 if [[ "$episode" == 38 ]]; then extra=(--alignment-video data/sim_replay/validation/ep_38.mp4); fi
 success=0
 for attempt in 1 2; do
  if envs/habitat/bin/python -u scripts/eval_findingdory_hlsr_cached_framepack.py --step 3100 --episode "$episode" --prediction-manifest "$run/hlsr_sim_replay_predictions.json" "${extra[@]}" > "logs/hlsr_official_3100_framepack/ep_${episode}.log" 2>&1; then success=1;break;fi
 done
 [[ "$success" == 1 ]] || exit 1
done
envs/model/bin/python scripts/aggregate_findingdory_hlsr_cached_framepack.py --step 3100
