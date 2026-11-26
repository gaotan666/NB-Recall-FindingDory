"""Isolated FindingDory reader/count experiment; the full validation run uses v1."""
import ast
import math
import re
from collections import Counter

from findingdory.policies.end_to_end.qwen_imagenav_agent import QwenImageNavAgent

from .findingdory_agent import NBRecallImageNavAgent


class NBRecallImageNavAgentV2(NBRecallImageNavAgent):
    """Preserve SFT answer groups, then repair only train-supported count errors."""

    def _interaction_spans(self, strict=True):
        raw = super()._interaction_spans(strict=False)
        if not strict:
            return raw
        # In long episodes the 96-frame subsample can hit a pick interaction
        # only once. Dropping single-frame spans removes every pick and shifts
        # all role labels; train replay supports retaining complete even pairs.
        if len(raw) >= 2 and len(raw) % 2 == 0:
            return raw
        return super()._interaction_spans(strict=True)

    def extract_info_from_response(self, response):
        if "assistant\n" in response:
            response = response.split("assistant\n", 1)[1].strip()
        parsed = QwenImageNavAgent.extract_info_from_response(self, response)
        if parsed is not None:
            return parsed
        try:
            groups = ast.literal_eval(response.strip().removeprefix("```python")
                                      .removesuffix("```").strip())
        except (SyntaxError, ValueError, AttributeError):
            return None
        if not isinstance(groups, list):
            return None
        if groups and all(isinstance(item, int) for item in groups):
            groups = [groups]
        if not all(isinstance(group, list) for group in groups):
            return None
        goal = str(getattr(self, "_sft_current_goal", "")).strip().lower()
        if (len(groups) == 1 and getattr(self, "single_goal_collapse", "off") != "off"
                and not goal.startswith("revisit all")):
            candidates = [i for i in groups[0] if isinstance(i, int) and 0 <= i < 96]
            return {"frame_indices": candidates or [-1]}
        # A group is one requested destination. Flattening a group into many
        # destinations creates definite num_targets_error failures.
        indices = []
        for group in groups:
            candidates = [i for i in group if isinstance(i, int) and 0 <= i < 96]
            if candidates:
                indices.append(candidates[0])
        return {"frame_indices": indices or [-1]}

    def _dedup_spans_by_gps(self, spans, radius_m):
        centers, chosen = [], []
        for group in spans:
            frame = group[len(group) // 2].content["sampled_frame"]
            if frame >= len(self._observations):
                return []
            gps = self._observations[frame].get("gps")
            if not isinstance(gps, (list, tuple)) or len(gps) != 2:
                return []
            if any(math.dist(gps, other) <= radius_m for other in centers):
                continue
            centers.append(gps)
            chosen.append(frame)
        return chosen

    def _repair_multi_goal_count(self, goal, indices):
        lower = goal.strip().lower()
        if not self.multi_goal_count_prior or not lower.startswith("revisit all"):
            return indices
        groups = self._interaction_spans()
        if not groups:
            return indices
        class_query = re.search(r"revisit all the (.+?) you (picked objects from|placed objects on)",
                                lower)
        if class_query and class_query.group(1) not in {"receptacles", "objects"}:
            category = class_query.group(1).strip().replace(" ", "_")
            relevant = (groups[::2] if class_query.group(2) == "picked objects from"
                        else groups[1::2])
            chosen = []
            for group in relevant:
                counts = Counter(str(name).lower() for event in group
                                 for name in event.content.get("visible_classes", []))
                if counts[category] >= 2:
                    chosen.append(group[len(group) // 2].content["sampled_frame"])
            # RGB co-visibility does not prove which receptacle was used. A
            # broad replacement regressed a correctly answered place query in
            # the pilot, so only add one strongly repeated pick candidate.
            if (class_query.group(2) == "picked objects from" and len(indices) >= 2 and
                    len(chosen) == len(indices) + 1):
                self._record({"kind": "multi_goal_rgb_class_repair", "category": category,
                              "original_count": len(indices), "repaired_count": len(chosen),
                              "chosen_frames": chosen,
                              "basis": "repeated RGB class evidence during observed interaction"})
                return chosen
            return indices
        if not any(term in lower for term in ("all the receptacles", "all the objects")):
            return indices
        if "receptacles you picked" in lower:
            relevant, kind = groups[::2], "pick_receptacle"
        elif "receptacles you placed" in lower:
            relevant, kind = groups[1::2], "place_receptacle"
        elif "objects you interacted" in lower:
            relevant, kind = groups[1::2], "interacted_object"
        else:
            return indices
        if "specific order" in lower:
            suffix = lower.split("order to revisit them is:", 1)
            if len(suffix) != 2:
                return indices
            names = re.findall(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\b",
                               suffix[1])
            ordinal = {"first": 0, "second": 1, "third": 2, "fourth": 3,
                       "fifth": 4, "sixth": 5, "seventh": 6, "eighth": 7}
            order = [ordinal[name] for name in names]
            if not order or len(set(order)) != len(order) or max(order) >= len(relevant):
                return indices
            if len(indices) == len(order):
                return indices
            chosen = [relevant[i][len(relevant[i]) // 2].content["sampled_frame"]
                      for i in order]
            self._record({"kind": "multi_goal_order_repair", "original_count": len(indices),
                          "repaired_count": len(chosen), "chosen_frames": chosen,
                          "basis": "observed interaction order; train replay audit"})
            return chosen
        if kind == "interacted_object":
            if len(indices) == len(relevant):
                return indices
            chosen = [group[len(group) // 2].content["sampled_frame"]
                      for group in relevant]
            self._record({"kind": "multi_goal_object_count_repair",
                          "original_count": len(indices), "repaired_count": len(chosen),
                          "chosen_frames": chosen,
                          "basis": "observed pick/place spans; train replay audit"})
            return chosen
        unique = list(dict.fromkeys(indices))
        if len(unique) < len(indices):
            self._record({"kind": "multi_goal_duplicate_frame_removed",
                          "original_count": len(indices), "repaired_count": len(unique),
                          "basis": "the same sampled frame cannot denote two distinct destinations"})
            indices = unique
        radius = 0.75 if kind == "pick_receptacle" else 0.5
        chosen = self._dedup_spans_by_gps(relevant, radius)
        if (not chosen or len(chosen) == len(indices) or
                len(indices) < len(relevant)):
            return indices
        self._record({"kind": "multi_goal_gps_dedup", "original_count": len(indices),
                      "repaired_count": len(chosen), "chosen_frames": chosen,
                      "radius_m": radius, "basis": "observed interaction GPS; train replay audit"})
        return chosen
