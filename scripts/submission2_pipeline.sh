#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ ! -f REPRODUCTION_WORKSPACE.json ]]; then
  echo '请先用 创建复现目录.py 创建工作副本，不在归档目录内运行实验。' >&2
  exit 1
fi
export PYTHONPATH="$PWD/vendor/findingdory-habitat:$PWD"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
stage="${1:-help}"
run=reports/stage_c_validation_3100_framepack
case "$stage" in
 stage-a)
  [[ ! -e models/nb_recall_unified_lora_stage_a ]] || { echo '建库模型输出已存在，拒绝覆盖'; exit 1; }
  bash scripts/launch_unified_stage_a_3gpu.sh
  ;;
 stage-b)
  [[ ! -e models/nb_recall_dual_lora_trust_heads ]] || { echo 'Trust 输出已存在，拒绝覆盖'; exit 1; }
  BATCH_EPISODES=8 MAX_EPISODES_PER_SHARD=20 bash scripts/launch_autoregressive_memory_3gpu.sh train
  BATCH_EPISODES=8 MAX_EPISODES_PER_SHARD=8 bash scripts/launch_autoregressive_memory_3gpu.sh calibration
  envs/model/bin/python scripts/build_dual_lora_trust_supervision.py
  envs/model/bin/python scripts/train_dual_lora_trust_heads.py
  ;;
 memory)
  split="${2:-train}"
  [[ "$split" == train || "$split" == validation ]] || exit 1
  envs/model/bin/python scripts/export_merged_memory_builder.py
  pids=()
  for gpu in 0 1 2; do
   CUDA_VISIBLE_DEVICES="$gpu" envs/inference_fast/bin/python -u scripts/generate_stage_c_async_memory.py \
    --split "$split" --episodes 0 --shard "$gpu" --shards 3 --concurrency 64 \
    > "logs/full_memory_${split}_${gpu}.log" 2>&1 &
   pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]]
  ;;
 train-data)
  envs/model/bin/python scripts/build_stage_c_answer_data.py --episodes 0 --out data/stage_c_answer_full --separate-reader-gate
  ;;
 stage-c)
  [[ ! -e models/nb_recall_answer_lora_stage_c ]] || { echo '答题模型输出已存在，拒绝覆盖'; exit 1; }
  CUDA_VISIBLE_DEVICES=0,1,2 envs/model/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 \
   scripts/train_stage_c_answer_lora.py --data data/stage_c_answer_full/train.jsonl \
   --run-name nb_recall_answer_lora_stage_c --max-steps 3100 --gradient-accumulation 8 --learning-rate 5e-5 --save-every 100
  ;;
 eval-data)
  [[ -f data/sim_replay/validation/ep_38.mp4 ]] || { echo '缺少历史 episode 38 对齐视频，见运行说明'; exit 1; }
  envs/model/bin/python scripts/export_merged_memory_builder.py
  CUDA_VISIBLE_DEVICES=0 envs/inference_fast/bin/python scripts/generate_stage_c_async_memory.py \
   --split validation --memory-split sim_validation --episode-id ep_38 --video-override-dir data/sim_replay/validation \
   --episodes 0 --shard 0 --shards 1 --concurrency 1 --gpu-memory-utilization 0.5
  envs/model/bin/python scripts/build_stage_c_action8_data.py --source-split validation --episodes 0 \
   --out data/stage_c_answer_validation_700_action8 --separate-reader-gate
  envs/model/bin/python scripts/build_stage_c_action8_data.py --source-split sim_validation --episode-id ep_38 --episodes 0 \
   --out data/stage_c_answer_sim_validation_ep_38_action8 --separate-reader-gate
  envs/model/bin/python scripts/build_stage_c_framepack_data.py
  ;;
 answer)
  CUDA_VISIBLE_DEVICES="${ANSWER_GPU:-3}" envs/model/bin/python -u scripts/eval_stage_c_answer_batched.py \
   --source-split validation --data data/stage_c_answer_validation_700_framepack/audit.jsonl \
   --adapter models/nb_recall_answer_lora_stage_c/checkpoint-3100 --execution controller \
   --shard -1 --questions-per-shard 0 --resume --out "$run"
  envs/model/bin/python scripts/prepare_hlsr_cached_predictions_framepack.py --step 3100
  CUDA_VISIBLE_DEVICES="${ANSWER_GPU:-3}" envs/model/bin/python -u scripts/eval_stage_c_answer_batched.py \
   --source-split sim_validation --data data/stage_c_answer_sim_validation_ep_38_framepack/audit.jsonl \
   --adapter models/nb_recall_answer_lora_stage_c/checkpoint-3100 --execution controller \
   --shard -1 --questions-per-shard 0 --resume --out "$run/sim_replay_answer_ep_38"
  envs/model/bin/python scripts/merge_hlsr_sim_replay_predictions_framepack.py --step 3100 --episode 38
  echo '答题完成，先审阅离线结果。本命令不会启动仿真。'
  ;;
 hl)
  [[ -s "$run/hlsr_sim_replay_predictions.json" ]] || exit 1
  pids=()
  for gpu in 0 1 2 3; do
   (
    for ((episode=gpu+1;episode<=100;episode+=4)); do
     [[ -s "$run/hlsr_official/ep_${episode}.json" ]] && continue
     extra=()
     if [[ "$episode" == 38 ]]; then extra=(--alignment-video data/sim_replay/validation/ep_38.mp4); fi
     CUDA_VISIBLE_DEVICES="$gpu" envs/habitat/bin/python -u scripts/eval_findingdory_hlsr_cached_framepack.py \
      --step 3100 --episode "$episode" --prediction-manifest "$run/hlsr_sim_replay_predictions.json" "${extra[@]}" \
      > "logs/hl_ep_${episode}.log" 2>&1
    done
   ) &
   pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]]
  envs/model/bin/python scripts/aggregate_findingdory_hlsr_cached_framepack.py --step 3100
  ;;
 *) echo '阶段：stage-a / stage-b / memory train / train-data / stage-c / memory validation / eval-data / answer / hl' ;;
esac
