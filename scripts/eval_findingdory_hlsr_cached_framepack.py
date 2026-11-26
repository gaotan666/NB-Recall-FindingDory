"""Evaluate frozen frame predictions with FindingDory's simulator HL-SR verifier.

The simulator replays the official oracle experience once per validation episode.
Only the cached checkpoint decisions are supplied at question time; no model
weights or validation answers are read by this process.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import open_dict
from habitat.config.default import patch_config
from habitat.core.env import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import findingdory.config  # noqa: F401
import findingdory.dataset  # noqa: F401
import findingdory.task  # noqa: F401
from findingdory.policies.llm.vlm_agent import VLMAgent

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--step', type=int, default=700)
parser.add_argument('--task', default='', help='One task_# for alignment smoke; default evaluates every official task')
parser.add_argument('--max-steps', type=int, default=6000)
parser.add_argument('--seed', type=int, default=None,
                    help='Optional simulator replay seed; matches the official eval entrypoint when set to 100')
parser.add_argument('--prediction-manifest',default=None)
parser.add_argument('--alignment-video',default=None,
                    help='Question-blind simulator replay video when the public video follows a different trajectory')
args = parser.parse_args()
ep = f'ep_{args.episode}'
if args.prediction_manifest is None:
    args.prediction_manifest = f'reports/stage_c_validation_{args.step}_framepack/hlsr_cached_predictions.json'
cache = json.loads((ROOT / args.prediction_manifest).read_text())
predictions = cache['episodes'][ep]
if args.task and args.task not in predictions:
    raise RuntimeError(f'No frozen prediction for {ep} {args.task}')
overrides = [f'habitat.dataset.episode_ids=[{args.episode}]']
if args.task:
    overrides.append(f'habitat.task.instructions_to_evaluate=[{args.task}]')
with initialize_config_module(config_module='findingdory.config', version_base=None):
    cfg = compose(config_name='baseline/qwen_imagenav', overrides=overrides)
with open_dict(cfg):
    cfg.habitat_baselines.agent.config.prompt_file = str(
        ROOT / 'vendor/findingdory-habitat/findingdory/policies/llm/prompts/qwen_prompt.txt')
    cfg.habitat_baselines.agent.config.output_folder = str(ROOT / f'reports/stage_c_validation_{args.step}_framepack/agent/{ep}')
    cfg.habitat_baselines.agent.config.vlm_temp_folder = str(ROOT / f'reports/stage_c_validation_{args.step}_framepack/temp/{ep}')
cfg = patch_config(cfg)
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
env = Env(config=cfg)
# This run stops immediately after each high-level decision. The benchmark's
# low-level failure diagnosis assumes a navigation policy has acted and can
# crash when every candidate is already satisfied. Keep official HL-SR/PDDL
# verification intact; skip only the unused low-level stop diagnosis.
low_level_diagnosis = env._task.measurements.measures['c']
original_low_level_update = low_level_diagnosis.update_metric


def high_level_only_low_level_update(*args, task, **kwargs):
    action = kwargs.get('action', {}).get('action')
    if (not task._data_collection_phase and task._high_level_goal_assigned
            and task._high_level_goal_success
            and action in (HabitatSimActions.stop, 'pddl_intermediate_stop')):
        low_level_diagnosis._metric = {'low_level_navigation_not_run': True}
        return
    return original_low_level_update(*args, task=task, **kwargs)


low_level_diagnosis.update_metric = high_level_only_low_level_update
agent = VLMAgent(cfg.habitat_baselines.agent.config)
agent.episode_index = args.episode
results = []
alignment = {}
steps = 0
out_dir = ROOT / f'reports/stage_c_validation_{args.step}_framepack/hlsr_official'
out_dir.mkdir(parents=True, exist_ok=True)


def frame_mae(video_index, obs):
    path = ROOT / (args.alignment_video or f'data/findingdory-subsampled-96/videos/val/{ep}.mp4')
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot read {ep} frame {video_index}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    observed = np.asarray(obs['head_rgb'])[..., :3]
    if rgb.shape != observed.shape:
        raise RuntimeError(f'Video and simulator dimensions differ for {ep}')
    return float(np.abs(rgb[90:].astype(np.int16)-observed[90:].astype(np.int16)).mean())


try:
    obs = env.reset()
    if str(env.current_episode.episode_id) != str(args.episode):
        raise RuntimeError('Simulator episode id mismatch')
    agent.reset()
    while not obs['can_take_action'] and not env.episode_over:
        if steps >= args.max_steps:
            raise RuntimeError('Oracle experience exceeded max steps')
        action = agent.act(obs)
        obs = env.step(action)
        steps += 1
    if env.episode_over or not obs['can_take_action']:
        raise RuntimeError('Oracle experience did not reach question phase')
    if len(agent._observations) < 96:
        raise RuntimeError(f'Fewer than 96 simulator history frames: {len(agent._observations)}')
    # The official agent applies exactly this uniform 96-frame subsampling.
    agent.process_frames_for_vlm(save_images=False)
    if len(agent._observations) != 96:
        raise RuntimeError('Official agent frame subsampling failed')
    alignment = {
        'sampled_frames': len(agent._observations),
        'original_history_frames': steps,
        'last_original_frame_index': agent._frame_num_to_original_frame_num[-1],
        'video_first_frame_mae': frame_mae(0, agent._observations[0]),
        'video_frame_10_mae': frame_mae(10, agent._observations[10]),
        'video_frame_47_mae': frame_mae(47, agent._observations[47]),
        'video_frame_90_mae': frame_mae(90, agent._observations[90]),
        'video_last_frame_mae': frame_mae(95, agent._observations[-1]),
    }
    print('ALIGNMENT', json.dumps(alignment), flush=True)
    # Some encoded videos have a divergent initial frame even when their
    # subsequent sampled trajectory is identical to the simulator replay.
    if any(alignment[key] > 18 for key in ('video_frame_10_mae',
                                          'video_frame_47_mae',
                                          'video_frame_90_mae',
                                          'video_last_frame_mae')):
        probe = ROOT / f'reports/stage_c_validation_{args.step}_framepack/alignment' / ep
        probe.mkdir(parents=True, exist_ok=True)
        for index in (0, 10, 47, 90, 95):
            cv2.imwrite(str(probe / f'sim_{index:02d}.png'),
                        cv2.cvtColor(np.asarray(agent._observations[index]['head_rgb'])[..., :3],
                                     cv2.COLOR_RGB2BGR))
            cap = cv2.VideoCapture(str(ROOT / f'data/findingdory-subsampled-96/videos/val/{ep}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(probe / f'video_{index:02d}.png'), bgr)
        raise RuntimeError(f'Offline video and simulator trajectory do not align: {alignment}')

    expected = 1 if args.task else len(predictions)
    while not env.episode_over and len(results) < expected:
        task_id = env._task.current_task_id
        if task_id not in predictions:
            raise RuntimeError(f'No cached model prediction for simulator instruction {ep} {task_id}')
        item = predictions[task_id]
        selected = item['frame_indices']
        valid = selected != [-1]
        original_indices = [agent._frame_num_to_original_frame_num[i] for i in selected] if valid else [-1]
        goal_states = [agent._agent_states[i] for i in selected] if valid else []
        nav_modes = [agent._agent_in_nav_mode[i] for i in selected] if valid else []
        action = {'action': 'high_level_policy_action',
                  'action_args': {'nav_indices': original_indices,
                                  'nav_goal_states': goal_states,
                                  'nav_mode_flag': nav_modes}}
        obs = env.step(action)
        steps += 1
        metric = env._task.metrics_per_task.get(task_id)
        if metric is not None:
            success = bool(metric['high_level_goal_success'])
            dtg = metric.get('high_level_dtg_success')
            sem_cov = metric.get('high_level_sem_cov_success')
        else:
            success = bool(env._task._high_level_goal_success)
            dtg = bool(env._task._high_level_dtg_success)
            sem_cov = bool(env._task._high_level_sem_cov_success)
        record = {'episode': ep, 'task_id': task_id,
                  'question_id': item['question_id'], 'category': item['category'],
                  'low_level_category': item['low_level_category'],
                  'model_decision': item['model_decision'],
                  'sampled_frame_indices': selected,
                  'original_frame_indices': original_indices,
                  'high_level_goal_success': success,
                  'high_level_dtg_success': dtg,
                  'high_level_sem_cov_success': sem_cov}
        results.append(record)
        print('TASK_RESULT', json.dumps(record), flush=True)
        if not env.episode_over and env._task.current_task_id == task_id:
            obs = env.step({'action': 0})
            steps += 1
        if steps > args.max_steps:
            raise RuntimeError('High-level verification exceeded max steps')
    if len(results) != expected:
        raise RuntimeError(f'Expected {expected} task results but got {len(results)}')
    report = {'episode': ep, 'scope': 'official FindingDory simulator HL-SR',
              'checkpoint_step': args.step,
              'low_level_navigation_not_run': True,
              'replay_seed': args.seed,
              'prediction_manifest': args.prediction_manifest,
              'alignment_video': args.alignment_video or f'data/findingdory-subsampled-96/videos/val/{ep}.mp4',
              'adapter_sha256': cache['adapter_sha256'],
              'alignment': alignment, 'simulator_steps': steps,
              'questions': len(results),
              'high_level_successes': sum(r['high_level_goal_success'] for r in results),
              'results': results}
    suffix = f'_{args.task}' if args.task else ''
    (out_dir / f'{ep}{suffix}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('SUMMARY', json.dumps({k: report[k] for k in ('episode', 'simulator_steps',
                                                       'questions', 'high_level_successes')}), flush=True)
finally:
    env.close()
