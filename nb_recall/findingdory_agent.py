"""FindingDory agent with RGB-grounded typed memory and a real re-observe branch.

This adapter keeps the official ImageNav policy for low-level navigation. Its
pilot thresholds remain provisional until re-observation benefit/cost is measured.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from findingdory.policies.end_to_end.qwen_imagenav_agent import QwenImageNavAgent
from .memory import MemoryGraph, MemoryNode
from .runtime import TrustScorer, WriterController
from .sft_reader import SFTResponseMixin, SFT_QUESTION_PROMPT, trained_video_response


EXTRACT_SYSTEM = ("从机器人 RGB 图像提取可见物体的类别和边界框。只输出紧凑 JSON："
                  "{\"objects\":[{\"kind\":\"object\",\"class\":\"kettle\",\"bbox\":[0,0,1000,1000]}]}。"
                  "框坐标按图像宽高归一到 0..1000，最多 8 个清晰可见物体；不要猜测被遮挡物体。")


class NBRecallImageNavAgent(SFTResponseMixin, QwenImageNavAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PeftModel.from_pretrained(self.model, config.extractor_adapter).eval()
        self.reader_model_path = config.get("reader_model")
        self.reader_prompt_mode = str(config.get("sft_prompt_mode", "official"))
        self.reader_memory_mode = str(config.get("reader_memory_mode", "prompt"))
        if self.reader_memory_mode not in {
                "prompt", "fallback", "targeted_fallback", "gated", "memory_guided"}:
            raise ValueError(
                "reader_memory_mode must be prompt, fallback, targeted_fallback, gated or memory_guided")
        self.reader_model = None
        self.reader_processor = None
        self._sft_reader_active = False
        if self.reader_model_path:
            self.reader_processor = AutoProcessor.from_pretrained(self.reader_model_path)
            self.reader_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.reader_model_path, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="auto").eval()
        self.writer = WriterController(Path(config.writer_dir))
        self.memory_write_mode = str(config.get("memory_write_mode", "append_all"))
        if self.memory_write_mode not in {"append_all", "learned"}:
            raise ValueError("memory_write_mode must be append_all or learned")
        self.trust_head = TrustScorer(Path(config.trust_dir))
        self.tau_lo = float(config.tau_lo)
        self.tau_hi = (float(config.tau_hi_override) if config.get("tau_hi_override") is not None
                       else self.trust_head.tau_hi)
        if self.tau_hi is None or not 0 <= self.tau_lo < self.tau_hi <= 1:
            raise ValueError("Pilot gate needs calibrated tau_hi and 0 <= tau_lo < tau_hi")
        self.max_reobserve_steps = int(config.max_reobserve_steps)
        self.memory_context_enabled = bool(config.get("memory_context_enabled", False))
        self.context_event_budget = int(config.get("context_event_budget", 18))
        self.context_fact_budget = int(config.get("context_fact_budget", 8))
        self.context_selection = str(config.get("context_selection", "legacy"))
        if self.context_selection not in {"legacy", "balanced"}:
            raise ValueError("context_selection must be legacy or balanced")
        self.extractor_max_pixels = int(config.get("extractor_max_pixels", 384 * 384))
        self.single_goal_collapse = str(config.get("single_goal_collapse", "off"))
        if self.single_goal_collapse not in {"off", "first_cluster", "evidence_rank"}:
            raise ValueError("single_goal_collapse must be off, first_cluster or evidence_rank")
        self.span_role_prior = bool(config.get("span_role_prior", False))
        self.multi_goal_count_prior = bool(config.get("multi_goal_count_prior", False))
        self.spatial_memory_selector = bool(config.get("spatial_memory_selector", False))
        self.spatial_distance_margin_m = float(config.get("spatial_distance_margin_m", 1.0))
        self.merge_position_radius_m = float(config.merge_position_radius_m)
        self._entity_counter = 0
        self._acquisition_groups = {}
        self._memory_ready = False
        self._selected_indices = None
        self._query_phase = "query"
        self._reobserve_steps = 0
        self._reobserve_target = None
        self._previous_event_id = None
        self._memory_context = ""

    def reset(self):
        super().reset()
        self._observation_clock = 0
        self.graph = MemoryGraph(self.merge_position_radius_m)
        self._entity_counter = 0
        self._acquisition_groups = {}
        self._memory_ready = False
        self._query_phase = "query"
        self._selected_indices = None
        self._reobserve_steps = 0
        self._reobserve_target = None
        self._previous_event_id = None
        self._memory_context = ""

    def reset_new_task(self, task_id):
        super().reset_new_task(task_id)
        self._query_phase = "query"
        self._selected_indices = None
        self._reobserve_steps = 0
        self._reobserve_target = None
        self._memory_context = ""

    def _record(self, event):
        event.update({"episode": self.episode_index, "task_id": self.task_id})
        path = Path(self.output_folder) / f"ep_{self.episode_index:04d}" / "nb_recall_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _extract(self, obs, sampled_index, original_step, source):
        folder = Path(self.output_folder) / f"ep_{self.episode_index:04d}" / "memory_frames"
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{source}_{original_step:05d}.png"
        image_path = folder / filename
        rgb = np.asarray(obs["head_rgb"])[..., :3]
        if not image_path.exists():
            Image.fromarray(rgb).save(image_path)
        messages = [{"role": "system", "content": [{"type": "text", "text": EXTRACT_SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": str(image_path),
                                                  "max_pixels": self.extractor_max_pixels},
                                                 {"type": "text", "text": "提取这张图像中的有证据物体。"}]}]
        image_inputs, _ = process_vision_info(messages)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=256, do_sample=False)
        answer = self.processor.decode(generated[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            payload = json.loads(answer[answer.index("{"):answer.rindex("}")+1])
            objects = payload["objects"]
            if not isinstance(objects, list):
                objects = []
        except (ValueError, KeyError, json.JSONDecodeError):
            objects = []
        self._record({"kind": "extract", "source": source, "frame": sampled_index,
                      "step": original_step, "candidate_count": len(objects),
                      "classes": [o.get("class") for o in objects if isinstance(o, dict)]})
        return objects, image_path, hashlib.sha256(rgb.tobytes()).hexdigest()

    def _extract_batch(self, items):
        """Decode several independent RGB observations in one VLM generation."""
        folder = Path(self.output_folder) / f"ep_{self.episode_index:04d}" / "memory_frames"
        folder.mkdir(parents=True, exist_ok=True)
        prompts, images, paths, hashes = [], [], [], []
        for sampled_index, original_step, obs in items:
            rgb = np.asarray(obs["head_rgb"])[..., :3]
            path = folder / f"collection_{original_step:05d}.png"
            if not path.exists():
                Image.fromarray(rgb).save(path)
            messages = [{"role": "system", "content": [{"type": "text", "text": EXTRACT_SYSTEM}]},
                        {"role": "user", "content": [{"type": "image", "image": str(path),
                                                      "max_pixels": self.extractor_max_pixels},
                                                     {"type": "text", "text": "提取这张图像中的有证据物体。"}]}]
            image_input, _ = process_vision_info(messages)
            prompts.append(self.processor.apply_chat_template(messages, tokenize=False,
                                                              add_generation_prompt=True))
            images.extend(image_input)
            paths.append(path)
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        batch = self.processor(text=prompts, images=images, padding=True,
                               return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=256, do_sample=False)
        results = []
        for j, (sampled_index, original_step, _obs) in enumerate(items):
            answer = self.processor.decode(generated[j][batch["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            try:
                payload = json.loads(answer[answer.index("{"):answer.rindex("}")+1])
                objects = payload["objects"]
                if not isinstance(objects, list):
                    objects = []
            except (ValueError, KeyError, json.JSONDecodeError):
                objects = []
            self._record({"kind": "extract", "source": "collection", "frame": sampled_index,
                          "step": original_step, "candidate_count": len(objects),
                          "classes": [o.get("class") for o in objects if isinstance(o, dict)]})
            results.append((objects, paths[j], hashes[j]))
        return results

    def _position_from_box(self, obs, bbox):
        if (not isinstance(bbox, list) or len(bbox) != 4 or
            not all(isinstance(v, (int, float)) for v in bbox)):
            return None
        x0, y0, x1, y1 = [float(x) for x in bbox]
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            return None
        depth = np.asarray(obs["head_depth"]).squeeze()
        height, width = depth.shape
        # Central crop avoids the background around a predicted object box.
        u0 = max(0, int(width * (0.7*x0+0.3*x1) / 1000))
        u1 = min(width, int(width * (0.3*x0+0.7*x1) / 1000) + 1)
        v0 = max(0, int(height * (0.7*y0+0.3*y1) / 1000))
        v1 = min(height, int(height * (0.3*y0+0.7*y1) / 1000) + 1)
        center = depth[v0:v1, u0:u1]
        values = center[(center > 0) & (center < 1)]
        if not len(values):
            return None
        distance = float(np.median(values) * 10.0)  # official normalized 0..10 m depth
        u = (u0+u1)/2
        v = (v0+v1)/2
        focal = width / (2 * math.tan(math.radians(43)/2))  # official Stretch head HFOV
        camera_point = np.array([(u-width/2)/focal*distance,
                                 -(v-height/2)/focal*distance, -distance, 1])
        position = (np.asarray(obs["camera_pose"], dtype=float) @ camera_point)[:3]
        pixels = max(1, (u1-u0)*(v1-v0))
        return [round(float(v), 3) for v in position], distance, pixels

    def _appearance_from_box(self, obs, bbox):
        rgb = np.asarray(obs["head_rgb"])[..., :3]
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = [float(x) for x in bbox]
        crop = rgb[max(0, int(y0*height/1000)):min(height, int(y1*height/1000)),
                   max(0, int(x0*width/1000)):min(width, int(x1*width/1000))]
        if not crop.size:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        norm = float(np.linalg.norm(hist))
        return (hist / norm).tolist() if norm > 0 else None

    def _entity_id(self, kind, category, position, appearance, original_step):
        options = [n for n in self.graph.nodes.values() if n.type == "Entity" and
                   n.content == category and n.support["scope"]["task_condition"] == kind]
        nearest = None
        nearest_distance = float("inf")
        appearance_matches = []
        for node in options:
            facts = [f for f in self.graph.nodes.values() if f.type == "Fact" and
                     f.entity_id == node.entity_id and f.validity["state"] == "valid"]
            if facts:
                latest = max(facts, key=lambda f: f.validity["t_valid_from"])
                d = math.dist(position, latest.content)
                if d < nearest_distance:
                    nearest, nearest_distance = node.entity_id, d
                prior_hist = latest.support.get("appearance_hist")
                if kind == "object" and appearance is not None and prior_hist is not None:
                    similarity = float(np.dot(appearance, prior_hist))
                    age = original_step - latest.provenance[-1]["time"]
                    if 0 <= age <= 100:
                        appearance_matches.append((similarity, node.entity_id))
        if nearest is not None and nearest_distance <= self.merge_position_radius_m:
            return nearest
        if appearance_matches:
            appearance_matches.sort(reverse=True)
            top_similarity, top_id = appearance_matches[0]
            second_similarity = appearance_matches[1][0] if len(appearance_matches) > 1 else 0.0
            if top_similarity >= 0.92 and top_similarity-second_similarity >= 0.08:
                return top_id
        self._entity_counter += 1
        return f"ep{self.episode_index}:{kind}:{category}:{self._entity_counter}"

    def _insert_objects(self, objects, obs, sampled_index, original_step, source, image_path, sha):
        scene = f"episode:{self.episode_index}"
        inserted = 0
        for candidate_index, obj in enumerate(objects[:8]):
            if not isinstance(obj, dict):
                continue
            kind, category = obj.get("kind"), obj.get("class")
            if kind not in {"object", "receptacle"} or not isinstance(category, str) or not category:
                continue
            position_data = self._position_from_box(obs, obj.get("bbox"))
            if position_data is None:
                continue
            position, distance, pixels = position_data
            appearance = self._appearance_from_box(obs, obj["bbox"])
            eid = self._entity_id(kind, category, position, appearance, original_step)
            support = {"scope": {"entity": eid, "scene": scene, "task_condition": "object_location"},
                       "kind": kind, "category": category, "pixel_count": pixels,
                       "depth_m": distance, "appearance_hist": appearance}
            provenance = {"source": source, "time": original_step, "episode": str(self.episode_index),
                          "action": None, "observation_ref": str(image_path), "sha256": sha,
                          **({"sampled_frame": sampled_index} if sampled_index is not None else {})}
            entity_id = f"entity:{eid}"
            if entity_id not in self.graph.nodes:
                self.graph.apply("WRITE", MemoryNode(entity_id, "Entity", category, [provenance],
                    {"state": "valid", "t_valid_from": original_step, "t_valid_to": None},
                    {"scope": {"entity": eid, "scene": scene, "task_condition": kind}}, entity_id=eid))
            previous_group, previous_step = self._acquisition_groups.get(eid, (0, -1000))
            group_number = previous_group if original_step - previous_step <= 10 else previous_group + 1
            self._acquisition_groups[eid] = (group_number, original_step)
            group = f"{eid}:{group_number}"
            features = {"log_pixels": math.log1p(pixels), "depth_m": distance, "age_steps": 0,
                        "seen_again": 0, "latest_observed_displacement_m": 0,
                        "snap_or_desnap_since": 0}
            # Every grounded observation gets its own immutable Fact in append_all
            # mode. Entity association remains a retrieval link; it never erases
            # or averages the underlying observation.
            frame_key = sampled_index if sampled_index is not None else f"live-{original_step}"
            node = MemoryNode(
                f"fact:{eid}:{original_step}:{frame_key}:{candidate_index}", "Fact", position,
                [provenance], {"state": "valid", "t_valid_from": original_step, "t_valid_to": None},
                support, trust=self.trust_head.score(features)["tau"], entity_id=eid,
                attribute="observed_position", acquisition_group=group)
            if self.memory_write_mode == "append_all":
                relation = self.graph.append_observation(node, entity_id)["relation"]
                operation = "WRITE"
            else:
                try:
                    result = self.writer.insert(self.graph, node)
                    operation = result["operation"]
                except (ValueError, KeyError):
                    operation = "DROP"
            if operation in {"WRITE", "UPDATE"} and self.memory_write_mode != "append_all":
                self.graph.add_edge(node.id, entity_id, "supports")
            if operation in {"WRITE", "UPDATE"}:
                inserted += 1
            self._record({"kind": "memory_write", "operation": operation,
                          "write_mode": self.memory_write_mode, "node": node.id,
                          "category": category, "frame": sampled_index,
                          "step": original_step,
                          **({"relation": relation} if self.memory_write_mode == "append_all" else {})})
        return inserted

    def _build_memory(self):
        if self._memory_ready:
            return
        if self._frame_num_to_original_frame_num is None:
            raise RuntimeError("Frame mapping missing before memory construction")
        for offset in range(0, len(self._observations), 8):
            items = [(i, int(self._frame_num_to_original_frame_num[i]), self._observations[i])
                     for i in range(offset, min(offset+8, len(self._observations)))]
            for (i, original, obs), (objects, path, sha) in zip(items, self._extract_batch(items)):
                self._insert_objects(objects, obs, i, original, "collection", path, sha)
                event_id = f"event:ep{self.episode_index}:frame:{i}"
                event = MemoryNode(event_id, "Event",
                    {"sampled_frame": i, "time_of_day": obs.get("time_of_day"),
                     "visible_classes": [o.get("class") for o in objects if isinstance(o, dict)],
                     "manipulation_mode": bool(obs.get("manipulation_mode", False))},
                    [{"source": "collection", "time": original, "episode": str(self.episode_index),
                      "action": None, "observation_ref": str(path), "sha256": sha}],
                    {"state": "valid", "t_valid_from": original, "t_valid_to": None},
                    {"scope": {"entity": None, "scene": f"episode:{self.episode_index}",
                               "task_condition": "observation_order"}})
                self.graph.apply("WRITE", event)
                if self._previous_event_id:
                    self.graph.add_edge(self._previous_event_id, event_id, "before")
                    self.graph.add_edge(event_id, self._previous_event_id, "after")
                self._previous_event_id = event_id
            print(f"NB-Recall memory extraction {offset+len(items)}/{len(self._observations)}", flush=True)
        if self.span_role_prior or self.multi_goal_count_prior:
            for index, group in enumerate(self._interaction_spans()):
                role = "likely_pick" if index % 2 == 0 else "likely_place"
                for event in group:
                    annotation_id = f"procedure:{event.id}:{role}"
                    annotation = MemoryNode(
                        annotation_id, "Procedure",
                        {"interaction_role_prior": role,
                         "basis": "observed manipulation span order; alternating pick/place "
                                  "verified on completed training replay, not observed directly "
                                  "at inference"},
                        list(event.provenance),
                        {"state": "valid", "t_valid_from":
                         event.validity["t_valid_from"], "t_valid_to": None},
                        {"scope": dict(event.support["scope"])})
                    self.graph.apply("WRITE", annotation)
                    self.graph.add_edge(annotation_id, event.id, "supports")
        self._memory_ready = True
        graph_path = Path(self.output_folder) / f"ep_{self.episode_index:04d}" / "memory_graph.json"
        graph_path.write_text(json.dumps(self.graph.to_dict(), ensure_ascii=False))
        self._record({"kind": "memory_ready", "frames": len(self._observations),
                      "nodes": len(self.graph.nodes), "edges": len(self.graph.edges),
                      "write_mode": self.memory_write_mode,
                      "graph_ref": str(graph_path)})

    def _select(self, goal):
        text = goal.lower().replace("_", " ")
        # Relational/temporal tasks need interaction provenance; category-only
        # sightings do not justify an answer to them yet.
        if re.search(r"\b(that|which|first|second|last|yesterday|interacted|picked|placed|farthest|closest|not)\b", text):
            return None
        words = {w.removesuffix("s") for w in re.findall(r"[a-z]+", text)
                 if w not in {"navigate", "to", "a", "an", "the", "of", "and", "object", "receptacle"}}
        candidates = []
        for node in self.graph.nodes.values():
            if node.type != "Fact" or node.validity["state"] != "valid":
                continue
            category = node.support.get("category", "").replace("_", " ").lower()
            category_words = {w.removesuffix("s") for w in re.findall(r"[a-z]+", category)}
            if not category_words or not category_words.issubset(words):
                continue
            if not any(isinstance(p.get("sampled_frame"), int) for p in node.provenance):
                continue
            latest_observation = max(node.provenance, key=lambda p: p["time"])
            age = max(0, getattr(self, "_observation_clock", int(self._frame_num_to_original_frame_num[-1])) - latest_observation["time"])
            related = [other for other in self.graph.nodes.values()
                       if other.type == "Fact" and other.entity_id == node.entity_id and
                       other.attribute == node.attribute]
            earlier = [other for other in related
                       if other.validity["t_valid_from"] < node.validity["t_valid_from"]]
            previous = max(earlier, key=lambda other: other.validity["t_valid_from"],
                           default=None)
            features = {"log_pixels": math.log1p(node.support["pixel_count"]),
                        "depth_m": node.support["depth_m"], "age_steps": age,
                        "seen_again": int(len(related) > 1),
                        "latest_observed_displacement_m": (
                            math.dist(node.content, previous.content) if previous else 0),
                        "snap_or_desnap_since": 0}
            score = self.trust_head.score(features)
            frame = max((p for p in node.provenance if isinstance(p.get("sampled_frame"), int)),
                        key=lambda p: p["time"])["sampled_frame"]
            candidates.append((score["tau"], frame, node, score, len(category_words)))
        if not candidates:
            return None
        _tau, frame, node, score, _specificity = max(candidates, key=lambda x: (x[4], x[0]))
        return score["tau"], frame, node, score

    def _prepare_reobserve(self, frame, node):
        self._query_phase = "reobserve"
        self._reobserve_target = node
        self._reobserve_steps = 0
        self._reobserve_frame = frame
        self._reset_policy_tensors()
        self.nav_indices = [frame]
        self.num_nav_targets = 1
        self._clipped_indices = [frame]
        self._set_imagenav_goals = False
        self._cur_subgoal_idx = None
        self.nav_goal_images = []
        self.action_index = 1  # bypass high-level verification during physical re-observation

    def _answer(self, obs, selected):
        self._reset_policy_tensors()
        self._selected_indices = selected
        self._query_phase = "answered"
        self.action_index = 0
        self._set_imagenav_goals = False
        self._cur_subgoal_idx = None
        self.nav_goal_images = []
        return super().act(obs)

    def _context_for_goal(self, goal):
        """Ground the fallback reader in a compact, traceable event timeline."""
        words = set(re.findall(r"[a-z]+", goal.lower().replace("_", " ")))
        relational = bool(re.search(
            r"\b(that|which|first|second|last|yesterday|interacted|picked|placed|"
            r"farthest|closest|not|where|room|immediately|longest|shortest|order|all)\b",
            goal.lower()))
        events = [n for n in self.graph.nodes.values() if n.type == "Event" and
                  isinstance(n.content, dict) and isinstance(n.content.get("sampled_frame"), int)]
        events.sort(key=lambda n: n.content["sampled_frame"])
        scored = []
        for event in events:
            categories = [str(x).replace("_", " ") for x in event.content.get("visible_classes", []) if x]
            overlap = sum(len(set(re.findall(r"[a-z]+", c.lower())) & words) for c in categories)
            manipulation = bool(event.content.get("manipulation_mode"))
            if overlap or (relational and manipulation):
                scored.append((overlap*3 + int(manipulation), event))
        scored.sort(key=lambda x: (-x[0], x[1].content["sampled_frame"]))
        manipulation_groups = (self._interaction_spans()
                               if self.span_role_prior and relational else [])
        if not relational:
            selected = sorted((event for _, event in scored[:min(8, self.context_event_budget)]),
                              key=lambda n: n.content["sampled_frame"])
        elif self.context_selection == "legacy":
            selected = sorted((event for _, event in scored[:self.context_event_budget]),
                              key=lambda n: n.content["sampled_frame"])
        else:
            chosen = {}
            for score, event in scored:
                if score <= 1 or len(chosen) >= self.context_event_budget//2:
                    break
                chosen[event.content["sampled_frame"]] = event
            if not manipulation_groups:
                manipulation_groups = self._interaction_spans(strict=False)
            for group in manipulation_groups:
                for event in (group[0], group[len(group)//2], group[-1]):
                    if len(chosen) < self.context_event_budget:
                        chosen[event.content["sampled_frame"]] = event
            if len(chosen) < self.context_event_budget and events:
                for index in np.linspace(0, len(events)-1, self.context_event_budget, dtype=int):
                    event = events[int(index)]
                    if len(chosen) >= self.context_event_budget:
                        break
                    chosen[event.content["sampled_frame"]] = event
            for _, event in scored:
                if len(chosen) >= self.context_event_budget:
                    break
                chosen[event.content["sampled_frame"]] = event
            selected = [chosen[i] for i in sorted(chosen)]
        lines = []
        for i, group in enumerate(manipulation_groups):
            line = (f"observed manipulation span {i+1}: sampled frames "
                    f"{group[0].content['sampled_frame']}..{group[-1].content['sampled_frame']}; ")
            if self.span_role_prior:
                role = "likely_pick" if i % 2 == 0 else "likely_place"
                classes = [str(c) for event in group for c in event.content.get("visible_classes", []) if c]
                frequent = [name for name, _ in Counter(classes).most_common(4)]
                line += (f"role={role}; visible_candidates={frequent}; "
                         "role is a training-trajectory prior, not directly verified")
            else:
                line += "interaction type not verified"
            lines.append(line)
        for event in selected:
            classes = list(dict.fromkeys(str(x) for x in event.content.get("visible_classes", []) if x))[:6]
            lines.append(f"sampled frame {event.content['sampled_frame']}: visible={classes}; "
                         f"manipulation_mode={bool(event.content.get('manipulation_mode'))}")
        facts = [n for n in self.graph.nodes.values() if n.type == "Fact" and
                 n.validity["state"] == "valid" and isinstance(n.content, list)]
        def relevant(node):
            category = str(node.support.get("category", "")).replace("_", " ").lower()
            return len(set(re.findall(r"[a-z]+", category)) & words)
        facts.sort(key=lambda n: (-relevant(n), -(n.trust or 0),
                                  -n.validity["t_valid_from"]))
        if not relational:
            facts = [fact for fact in facts if relevant(fact)]
        if self.context_selection == "balanced" and not any(relevant(n) for n in facts):
            diverse, seen_categories = [], set()
            for fact in facts:
                category = fact.support.get("category")
                if category not in seen_categories:
                    diverse.append(fact)
                    seen_categories.add(category)
            diverse_ids = {n.id for n in diverse}
            facts = diverse + [n for n in facts if n.id not in diverse_ids]
        for fact in facts[:self.context_fact_budget]:
            frames = [p["sampled_frame"] for p in fact.provenance
                      if isinstance(p.get("sampled_frame"), int)]
            if frames:
                lines.append(f"fact: {fact.support.get('category')} near position {fact.content} "
                             f"at sampled frames {frames[-3:]}; "
                             f"stored_trust={float(fact.trust or 0):.4f}; source image saved")
        if not lines:
            return ""
        instruction = (
            "For this direct category query, prioritize category-matching memory frames and use "
            "the video only to verify which candidate is closest to the destination."
            if not relational else
            "Use this event timeline as retrieval evidence and verify the required relation in the video.")
        return ("\nObserved memory evidence (sampled frame numbers are 0-based; uncertain detections "
                "may be wrong; manipulation_mode alone does not identify pick versus place):\n" +
                "\n".join(lines) + "\n" + instruction)

    def _targeted_context_for_goal(self, goal):
        """Offer RGB-grounded candidates only after the SFT reader abstains."""
        words = set(re.findall(r"[a-z]+", goal.lower().replace("_", " ")))
        matches = []
        for node in self.graph.nodes.values():
            if (node.type != "Fact" or node.validity["state"] != "valid" or
                    node.trust is None or node.trust < self.tau_lo):
                continue
            category = str(node.support.get("category", "")).replace("_", " ").lower()
            overlap = len(set(re.findall(r"[a-z]+", category)) & words)
            frames = [p.get("sampled_frame") for p in node.provenance
                      if isinstance(p.get("sampled_frame"), int)]
            if overlap and frames:
                matches.append((overlap, node.trust, max(frames), category))
        matches.sort(reverse=True)
        unique = []
        for _, _, frame, category in matches:
            if (frame, category) not in unique:
                unique.append((frame, category))
            if len(unique) >= 3:
                break
        if not unique:
            return self._memory_context
        candidates = "; ".join(f"{category} visible near sampled frame {frame}"
                               for frame, category in unique)
        return ("\nRGB memory candidates: " + candidates +
                ". These sightings do not prove interaction identity. Verify in the video before answering.")

    def get_vlm_response(self, images, prompt, path=None):
        if self._memory_context:
            prompt = prompt + self._memory_context
        if self._sft_reader_active:
            return trained_video_response(self, images, prompt, path)
        return super().get_vlm_response(images, prompt, path=path)

    def _interaction_spans(self, strict=True):
        events = [n for n in self.graph.nodes.values() if n.type == "Event" and
                  isinstance(n.content, dict) and isinstance(n.content.get("sampled_frame"), int) and
                  n.content.get("manipulation_mode")]
        events.sort(key=lambda n: n.content["sampled_frame"])
        groups = []
        for event in events:
            frame = event.content["sampled_frame"]
            if not groups or frame > groups[-1][-1].content["sampled_frame"] + 1:
                groups.append([])
            groups[-1].append(event)
        if not strict:
            return groups
        groups = [group for group in groups if len(group) >= 2]
        return groups if len(groups) >= 2 and len(groups) % 2 == 0 else []

    def _expected_multi_goal_count(self, goal):
        lower = goal.lower()
        if "all the table" in lower or "all the bed" in lower:
            return None  # visual class filter is not reliable enough to override SFT
        groups = self._interaction_spans()
        if not groups:
            return None
        if "receptacles you interacted" in lower:
            return len(groups)
        if "picked objects from" in lower or "you picked objects from" in lower:
            return len(groups[::2])
        if "placed objects on" in lower or "you placed objects on" in lower:
            return len(groups[1::2])
        if "objects you interacted" in lower:
            return len(groups[1::2])
        return None

    def _repair_multi_goal_count(self, goal, indices):
        if not self.multi_goal_count_prior or not goal.strip().lower().startswith("revisit all"):
            return indices
        if self.reader_model is not None and self._expected_multi_goal_count(goal) is None:
            return indices
        groups = self._interaction_spans()
        if not groups:
            return indices
        lower = goal.lower()
        if "receptacles you interacted" in lower:
            relevant = groups
        elif "picked objects from" in lower or "you picked objects from" in lower:
            relevant = groups[::2]
        elif "placed objects on" in lower or "you placed objects on" in lower:
            relevant = groups[1::2]
        elif "objects you interacted" in lower:
            relevant = groups[1::2]
        else:
            return indices
        if "all the table" in lower or "all the bed" in lower:
            category = "table" if "all the table" in lower else "bed"
            relevant = [group for group in relevant if any(
                category in str(c).lower().replace("_", " ")
                for event in group for c in event.content.get("visible_classes", []))]
            if not relevant:
                return indices
        if "specific order" in lower:
            suffix = lower.split("order to revisit them is:", 1)
            order_names = re.findall(r"\b(first|second|third|fourth|fifth|sixth)\b",
                                     suffix[1] if len(suffix) == 2 else "")
            ordinal = {"first": 0, "second": 1, "third": 2,
                       "fourth": 3, "fifth": 4, "sixth": 5}
            order = [ordinal[name] for name in order_names]
            if len(order) != len(relevant) or sorted(order) != list(range(len(relevant))):
                return indices
            relevant = [relevant[i] for i in order]
        if len(indices) == len(relevant) or len(relevant) > 8:
            return indices
        chosen = []
        for group in relevant:
            frames = [event.content["sampled_frame"] for event in group]
            midpoint = frames[len(frames)//2]
            candidates = [i for i in indices if isinstance(i, int) and
                          frames[0] - 1 <= i <= frames[-1] + 1]
            chosen.append(min(candidates, key=lambda i: abs(i - midpoint)) if candidates else midpoint)
        self._record({"kind": "multi_goal_count_prior", "original_count": len(indices),
                      "repaired_count": len(chosen), "chosen_frames": chosen,
                      "basis": "observed manipulation spans with pick/place order prior"})
        return chosen

    def _spatial_memory_choice(self, obs):
        """Resolve supported farthest queries from observed interaction positions."""
        if not self.spatial_memory_selector or not self.span_role_prior:
            return None
        goal = str(obs.get("lang_memory_goal", "")).lower()
        if "farthest from your current location" not in goal or "did not interact" in goal:
            return None
        groups = self._interaction_spans()
        if not groups:
            return None
        if "receptacle" in goal and "picked an object from" in goal:
            relevant = groups[::2]
        elif "receptacle" in goal and "placed an object on" in goal:
            relevant = groups[1::2]
        elif "receptacle" in goal and "interacted with" in goal:
            relevant = groups
        elif "object" in goal and "which you interacted with" in goal:
            relevant = groups[1::2]
        else:
            return None
        if len(relevant) < 2:
            return None
        now = np.asarray(obs.get("gps", []), dtype=float).ravel()
        if now.size != 2 or not np.all(np.isfinite(now)):
            return None
        choices = []
        for group in relevant:
            frame = group[len(group)//2].content["sampled_frame"]
            if frame >= len(self._observations):
                return None
            prior = np.asarray(self._observations[frame].get("gps", []), dtype=float).ravel()
            if prior.size != 2 or not np.all(np.isfinite(prior)):
                return None
            choices.append((float(np.linalg.norm(now - prior)), frame))
        choices.sort(reverse=True)
        margin = choices[0][0] - choices[1][0]
        if margin < self.spatial_distance_margin_m:
            return None
        return choices[0][1], margin, choices

    def run_vlm(self, frames, lang_goal):
        if self._selected_indices is not None:
            return self._selected_indices
        if self.reader_model is None:
            with self.model.disable_adapter():
                indices = super().run_vlm(frames, lang_goal)
        else:
            extractor, extractor_processor, old_prompt = self.model, self.processor, self.prompt
            context = self._memory_context
            try:
                self.model, self.processor = self.reader_model, self.reader_processor
                self._sft_reader_active = True
                self._sft_current_goal = lang_goal
                if self.reader_prompt_mode == "question_only":
                    self.prompt = SFT_QUESTION_PROMPT
                if self.reader_memory_mode in {"fallback", "targeted_fallback"}:
                    self._memory_context = ""
                indices = super().run_vlm(frames, lang_goal)
                if (self.reader_memory_mode in {"fallback", "targeted_fallback"} and context and
                        not any(isinstance(i, int) and i >= 0 for i in indices)):
                    self._memory_context = context
                    self._memory_context = (self._targeted_context_for_goal(lang_goal)
                                            if self.reader_memory_mode == "targeted_fallback" else context)
                    indices = super().run_vlm(frames, lang_goal)
                    self._record({"kind": "sft_memory_fallback", "reason": "no_valid_base_frame",
                                  "mode": self.reader_memory_mode, "result_frames": indices})
            finally:
                self.model, self.processor, self.prompt = extractor, extractor_processor, old_prompt
                self._sft_reader_active = False
                self._sft_current_goal = ""
                self._memory_context = context
        indices = self._repair_multi_goal_count(lang_goal, indices)
        if self.reader_memory_mode == "memory_guided":
            self._record({"kind": "memory_guided_answer", "question": lang_goal,
                          "result_frames": indices})
        if (self.single_goal_collapse == "off" or len(indices) <= 1 or
                lang_goal.strip().lower().startswith("revisit all")):
            return indices
        valid = sorted(set(i for i in indices if isinstance(i, int) and 0 <= i < len(frames)))
        if not valid:
            return indices
        clusters = [[valid[0]]]
        for index in valid[1:]:
            if index - clusters[-1][-1] <= 2:
                clusters[-1].append(index)
            else:
                clusters.append([index])
        chosen = clusters[0][len(clusters[0]) // 2]
        if self.single_goal_collapse == "evidence_rank":
            words = set(re.findall(r"[a-z]+", lang_goal.lower().replace("_", " ")))
            events = {n.content["sampled_frame"]: n for n in self.graph.nodes.values()
                      if n.type == "Event" and isinstance(n.content, dict) and
                      isinstance(n.content.get("sampled_frame"), int)}
            def evidence_score(index):
                best = 0
                for neighbor in range(max(0, index - 1), min(len(frames), index + 2)):
                    event = events.get(neighbor)
                    if event is None:
                        continue
                    categories = [str(c).replace("_", " ") for c in
                                  event.content.get("visible_classes", []) if c]
                    overlap = max((len(set(re.findall(r"[a-z]+", c.lower())) & words)
                                   for c in categories), default=0)
                    best = max(best, 3 * overlap + int(bool(event.content.get("manipulation_mode"))))
                return best
            ranked = [(evidence_score(i), i) for i in valid]
            if any(score > 0 for score, _ in ranked):
                top = max(score for score, _ in ranked)
                options = [i for score, i in ranked if score == top]
                chosen = min(options, key=lambda i: (abs(i - chosen), i))
        self._record({"kind": "single_goal_collapse", "mode": self.single_goal_collapse,
                      "original_count": len(indices), "chosen_frame": chosen,
                      "candidate_frames": indices})
        return [chosen]

    def act(self, obs):
        self._observation_clock = getattr(self, "_observation_clock", 0) + 1
        if not obs["can_take_action"]:
            return super().act(obs)
        if self._query_phase == "answered":
            return super().act(obs)
        if not self._memory_ready:
            self._frames = self.process_frames_for_vlm(save_images=False)
            self._vlm_frames_processed = True
            self._build_memory()
        if self._query_phase == "reobserve":
            # The input observation follows the last executed navigation action.
            # Do not emit a subgoal-switch/stop to the environment while inspecting:
            # those actions terminate or advance the benchmark answer itself.
            if self._reobserve_steps < self.max_reobserve_steps:
                action, rollouts = super().act(obs)
                finished = (isinstance(action, dict) and
                            action.get("action") in {"low_level_subgoal_switch_action", "pddl_intermediate_stop", 0})
                if not finished:
                    self._reobserve_steps += 1
                    return action, rollouts
            else:
                finished = False
            self._complete_reobserve(obs, finished)
        if self.reader_model is not None and self.reader_memory_mode == "memory_guided":
            self._memory_context = self._context_for_goal(obs["lang_memory_goal"])
            evidence_frames = sorted({int(frame) for frame in
                re.findall(r"(?:sampled frame|frames?)\s+(\d+)", self._memory_context)})
            self._record({"kind": "memory_guided_query",
                          "question": obs["lang_memory_goal"],
                          "context_chars": len(self._memory_context),
                          "evidence_frames": evidence_frames,
                          "memory_context": self._memory_context})
            return self._answer(obs, None)
        if self.reader_memory_mode == "gated":
            return self._gated_answer(obs)
        if self.reader_model is not None and self.reader_memory_mode in {"fallback", "targeted_fallback"}:
            spatial = self._spatial_memory_choice(obs)
            if spatial is not None:
                frame, margin, distances = spatial
                self._record({"kind": "gate", "decision": "spatial_memory_answer",
                              "frame": frame, "distance_margin_m": margin,
                              "candidate_distances_m": distances,
                              "basis": "observed interaction spans and current GPS"})
                return self._answer(obs, [frame])
            self._memory_context = (self._context_for_goal(obs["lang_memory_goal"])
                                    if self.memory_context_enabled else "")
            self._record({"kind": "gate", "decision": "reader_first",
                          "reason": "use_retrieved_memory_only_after_reader_abstains"})
            return self._answer(obs, None)
        choice = self._select(obs["lang_memory_goal"])
        if choice is None:
            self._memory_context = (self._context_for_goal(obs["lang_memory_goal"])
                                    if self.memory_context_enabled else "")
            self._record({"kind": "gate", "decision": "base_policy", "reason": "outside_memory_support"})
            return self._answer(obs, None)
        self._memory_context = ""
        tau, frame, node, scores = choice
        if tau >= self.tau_hi:
            decision = "answer"
        elif tau >= self.tau_lo and self._query_phase == "query":
            decision = "re-observe"
        else:
            decision = "abstain"
        self._record({"kind": "gate", "decision": decision, "frame": frame,
                      "node": node.id, **scores})
        if decision == "re-observe":
            self._prepare_reobserve(frame, node)
            action, rollouts = super().act(obs)
            self._reobserve_steps += 1
            return action, rollouts
        return self._answer(obs, [frame] if decision == "answer" else [])

    def _complete_reobserve(self, obs, finished):
        step = self._observation_clock
        objects, path, sha = self._extract(obs, None, step, "reobserve")
        inserted = self._insert_objects(objects, obs, None, step, "reobserve", path, sha)
        self._refreshed_choice = None
        target = self._reobserve_target
        # A current observation cannot establish a past interaction. Only confirm
        # a simple location query, and only if the observed position agrees with
        # the historical navigation anchor. Never invent a benchmark frame index.
        for node in self.graph.nodes.values():
            if (node.type != "Fact" or node.validity["state"] != "valid" or
                    node.support.get("category") != target.support.get("category")):
                continue
            if not any(p["source"] == "reobserve" and p["time"] == step for p in node.provenance):
                continue
            if math.dist(node.content, target.content) > self.merge_position_radius_m:
                continue
            # MERGE preserves the original support record. Score the actual
            # new RGB/depth evidence, not the old crop masquerading as fresh.
            measurements = []
            for obj in objects:
                if not isinstance(obj, dict) or obj.get("class") != node.support.get("category"):
                    continue
                measured = self._position_from_box(obs, obj.get("bbox"))
                if measured and math.dist(measured[0], node.content) <= self.merge_position_radius_m:
                    measurements.append(measured)
            if not measurements:
                continue
            position, depth, pixels = min(measurements, key=lambda item: math.dist(item[0], node.content))
            scores = self.trust_head.score({
                "log_pixels": math.log1p(pixels),
                "depth_m": depth, "age_steps": 0,
                "seen_again": int(node.entity_id == target.entity_id),
                "latest_observed_displacement_m": math.dist(node.content, target.content),
                "snap_or_desnap_since": 0})
            candidate = (scores["tau"], self._reobserve_frame, node, scores)
            if self._refreshed_choice is None or candidate[0] > self._refreshed_choice[0]:
                self._refreshed_choice = candidate
        graph_path = Path(self.output_folder) / f"ep_{self.episode_index:04d}" / "memory_graph.json"
        graph_path.write_text(json.dumps(self.graph.to_dict(), ensure_ascii=False))
        self._record({"kind": "reobserve_complete", "steps": self._reobserve_steps,
                      "new_candidates": len(objects), "inserted": inserted,
                      "navigation_stopped": finished,
                      "target_confirmed": self._refreshed_choice is not None,
                      "graph_ref": str(graph_path)})
        self._query_phase = "after_reobserve"

    def _gated_answer(self, obs):
        after = self._query_phase == "after_reobserve"
        choice = self._refreshed_choice if after else self._select(obs["lang_memory_goal"])
        if choice is None:
            if after:
                self._record({"kind": "gate", "decision": "abstain",
                              "reason": "reobservation_did_not_confirm_target"})
                return self._answer(obs, [])
            # The current heads are calibrated for object location evidence, not
            # arbitrary historical, relational or multi-goal answers. Be explicit
            # about this coverage limit rather than inventing confidence scores.
            self._memory_context = ""
            self._record({"kind": "gate", "decision": "reader_outside_support",
                          "reason": "no_supported_location_fact"})
            return self._answer(obs, None)
        tau, frame, node, scores = choice
        decision = ("answer" if tau >= self.tau_hi else
                    "re-observe" if tau >= self.tau_lo and not after else "abstain")
        self._record({"kind": "gate", "decision": decision, "node": node.id,
                      "frame": frame, "after_reobserve": after,
                      "tau_lo": self.tau_lo, "tau_hi": self.tau_hi, **scores})
        if decision == "re-observe":
            self._prepare_reobserve(frame, node)
            action, rollouts = super().act(obs)
            finished = (isinstance(action, dict) and action.get("action") in
                        {"low_level_subgoal_switch_action", "pddl_intermediate_stop", 0})
            if finished:
                self._complete_reobserve(obs, True)
                return self._gated_answer(obs)
            self._reobserve_steps = 1
            return action, rollouts
        if decision == "abstain":
            return self._answer(obs, [])
        self._memory_context = (f"Confirmed location evidence: category={node.support.get('category')}; "
                                f"historical frame={frame}; validity={node.validity}; "
                                f"tau_evidence={scores['tau_evidence']:.4f}; "
                                f"tau_fresh={scores['tau_fresh']:.4f}.")
        return self._answer(obs, [frame])
