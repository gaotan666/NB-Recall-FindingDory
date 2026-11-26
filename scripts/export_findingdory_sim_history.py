"""Export a question-blind 96-frame video from one official simulator replay."""
import argparse
import hashlib
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

import findingdory.config  # noqa: F401
import findingdory.dataset  # noqa: F401
import findingdory.task  # noqa: F401
from findingdory.policies.llm.vlm_agent import VLMAgent

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
with initialize_config_module(config_module='findingdory.config', version_base=None):
    cfg = compose(config_name='baseline/qwen_imagenav', overrides=[
        f'habitat.dataset.episode_ids=[{args.episode}]',
    ])
with open_dict(cfg):
    cfg.habitat_baselines.agent.config.prompt_file = str(
        ROOT / 'vendor/findingdory-habitat/findingdory/policies/llm/prompts/qwen_prompt.txt')
cfg = patch_config(cfg)
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
env = Env(config=cfg)
agent = VLMAgent(cfg.habitat_baselines.agent.config)
agent.episode_index = args.episode
try:
    obs = env.reset()
    agent.reset()
    while not obs['can_take_action'] and not env.episode_over:
        obs = env.step(agent.act(obs))
    if env.episode_over or len(agent._observations) < 96:
        raise RuntimeError('Oracle replay did not produce 96 historical frames')
    agent.process_frames_for_vlm(save_images=False)
    folder = ROOT / 'data/sim_replay/validation'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'ep_{args.episode}.mp4'
    height, width = agent._observations[0]['head_rgb'].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 1.0,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError('Could not open simulator video writer')
    try:
        for index, record in enumerate(agent._observations):
            image = cv2.cvtColor(np.asarray(record['head_rgb'])[..., :3], cv2.COLOR_RGB2BGR)
            frame_number = agent._frame_num_to_original_frame_num[index]
            for line, y in ((f'Frame: {frame_number}', 50),
                            (f'Time of Day: {record["time_of_day"]}', 70)):
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, lineType=cv2.LINE_AA)
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'episode': args.episode, 'split': 'validation',
                'source': 'official simulator oracle replay',
                'answers_used': False, 'frames': 96,
                'original_history_frames': len(agent._frame_num_to_original_frame_num)
                and agent._frame_num_to_original_frame_num[-1] + 1,
                'replay_seed': args.seed, 'video_sha256': digest}
    (folder / f'ep_{args.episode}.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
finally:
    env.close()
