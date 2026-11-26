"""Replay the official oracle's training experience; keep evidence and truth separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from hydra import compose, initialize_config_module
from habitat.config.default import patch_config
from habitat.core.env import Env
import findingdory.config  # noqa: F401
import findingdory.dataset  # noqa: F401
import findingdory.task  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--save-every", type=int, default=5)
args = parser.parse_args()
with initialize_config_module(config_module="findingdory.config", version_base=None):
    cfg = compose(config_name="baseline/qwen_imagenav", overrides=[
        "habitat.dataset.split=train",
        "habitat.dataset.data_path=data/datasets/findingdory-habitat/findingdory/train/episodes.json.gz",
        "habitat.dataset.viewpoints_matrix_path=data/datasets/findingdory-habitat/findingdory/train/viewpoints.npy",
        "habitat.dataset.transformations_matrix_path=data/datasets/findingdory-habitat/findingdory/train/transformations.npy",
        f"habitat.dataset.episode_ids=[{args.episode}]",
    ])
env = Env(config=patch_config(cfg))
out = ROOT / "data/sim_replay/train" / f"ep_{args.episode}"
out.mkdir(parents=True, exist_ok=True)
(out / "frames").mkdir(exist_ok=True)

category_maps = {
    "object": {int(v) + 1: k for k, v in env._dataset.obj_category_to_obj_category_id.items()},
    "receptacle": {int(v) + 1: k for k, v in env._dataset.recep_category_to_recep_category_id.items()},
    "other": {int(v) + 1: k for k, v in env._dataset.other_obj_category_to_other_obj_category_id.items()},
}
segment_keys = {"object": "all_object_segmentation", "receptacle": "receptacle_segmentation",
                "other": "other_object_segmentation"}

def visible_entities(obs):
    panoptic = np.asarray(obs["head_panoptic"]).squeeze()
    depth = np.asarray(obs["head_depth"]).squeeze()
    camera_pose = np.asarray(obs["camera_pose"], dtype=float)
    height, width = depth.shape
    fx = width / (2 * math.tan(math.radians(43) / 2))
    fy = fx
    entities = []
    for kind, key in segment_keys.items():
        segment = np.asarray(obs[key]).squeeze()
        for class_id in np.unique(segment):
            if int(class_id) == 0:
                continue
            ids, counts = np.unique(panoptic[segment == class_id], return_counts=True)
            for instance_id, pixels in zip(ids, counts):
                if int(instance_id) > 0 and int(pixels) >= 4:
                    yy, xx = np.where((segment == class_id) & (panoptic == instance_id))
                    valid_depth = depth[yy, xx]
                    valid_depth = valid_depth[(valid_depth > 0) & (valid_depth < 1)]
                    estimate = None
                    distance = None
                    if len(valid_depth):
                        distance = float(np.median(valid_depth) * 10.0)  # configured normalized 0-10 m depth
                        u, v = float(np.median(xx)), float(np.median(yy))
                        camera_point = np.array([(u - width / 2) / fx * distance,
                                                 -(v - height / 2) / fy * distance,
                                                 -distance, 1.0])
                        estimate = [round(float(x), 3) for x in (camera_pose @ camera_point)[:3]]
                    entities.append({"kind": kind, "instance_id": int(instance_id),
                                     "class": category_maps[kind].get(int(class_id), f"unknown_{class_id}"),
                                     "pixel_count": int(pixels), "depth_m": distance,
                                     "bbox_xyxy_1000": [int(1000 * xx.min() / width),
                                                        int(1000 * yy.min() / height),
                                                        int(1000 * (xx.max() + 1) / width),
                                                        int(1000 * (yy.max() + 1) / height)],
                                     "estimated_position_world": estimate})
    return entities

def capture(step, obs):
    task = env._task
    visible = {key: np.unique(obs[key]).astype(int).tolist() for key in
               ("all_object_segmentation", "receptacle_segmentation", "other_object_segmentation", "head_panoptic")}
    action_name = str(getattr(task.oracle_agent, "get_current_action_name", ""))
    frame_ref = None
    depth_ref = None
    if (step % args.save_every == 0 or action_name in {"snap_object", "desnap_object"}):
        frame_ref = f"frames/{step:05d}.png"
        Image.fromarray(np.asarray(obs["head_rgb"])[..., :3]).save(out / frame_ref)
        depth_ref = f"frames/{step:05d}.depth.png"
        Image.fromarray(np.clip(np.asarray(obs["head_depth"]).squeeze() * 65535, 0, 65535)
                        .astype(np.uint16)).save(out / depth_ref)
    evidence = {"step": step, "time_of_day": obs.get("time_of_day"),
                "gps": obs["gps"].tolist(),
                "manipulation_mode": bool(obs.get("manipulation_mode", False)),
                "last_action": np.asarray(obs.get("last_action", []), dtype=float).tolist(),
                "rgb_sha256": hashlib.sha256(obs["head_rgb"].tobytes()).hexdigest(),
                "frame_ref": frame_ref,
                "depth_ref": depth_ref,
                "camera_pose_start_frame": np.asarray(obs["camera_pose"]).tolist(),
                "visible_segmentation_ids": visible,
                "visible_entities": visible_entities(obs),
                "oracle_action_name": action_name,
                "collection_phase": bool(task._data_collection_phase)}
    manager = env._sim.get_rigid_object_manager()
    handles = manager.get_object_handles()
    truth = {"step": step, "object_ids_start": int(env._sim.habitat_config.object_ids_start),
             "camera_frame_start_position": [float(v) for v in task._robot_start_position],
             "camera_frame_start_rotation": [float(v) for v in task._robot_start_rotation],
             "robot_position": [float(v) for v in env._sim.get_agent_state().position],
             "object_positions": {h: [float(v) for v in manager.get_object_by_handle(h).translation]
              for h in handles},
             "object_id_by_handle": {h: int(manager.get_object_by_handle(h).object_id) for h in handles},
             "oracle_goal_idx": int(task._current_oracle_goal_idx)}
    return evidence, truth

summary = {"episode": args.episode, "steps": 0, "collection_complete": False,
           "episode_over": False, "scope": "official oracle replay, training split only"}
try:
    obs = env.reset()
    with (out / "observed.jsonl").open("w") as seen, (out / "privileged.jsonl").open("w") as hidden:
        for step in range(args.max_steps + 1):
            e, t = capture(step, obs)
            seen.write(json.dumps(e) + "\n")
            hidden.write(json.dumps(t) + "\n")
            if step % 50 == 0:
                seen.flush(); hidden.flush()
                print(json.dumps({"step": step, "collection_phase": e["collection_phase"],
                                  "oracle_action": e["oracle_action_name"]}), flush=True)
            summary["steps"] = step
            if not env._task._data_collection_phase or env.episode_over:
                break
            obs = env.step({"action": 0})  # official collector delegates to oracle in this phase
    summary["collection_complete"] = not env._task._data_collection_phase
    summary["episode_over"] = bool(env.episode_over)
finally:
    env.close()
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY", json.dumps(summary), flush=True)
