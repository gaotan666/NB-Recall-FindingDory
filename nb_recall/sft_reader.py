"""Parse the official FindingDory SFT checkpoint's nested-frame output."""
import ast

import torch
from qwen_vl_utils import process_vision_info

from findingdory.policies.end_to_end.qwen_imagenav_agent import QwenImageNavAgent

SFT_QUESTION_PROMPT = "The robot's goal is: {goal}"


class SFTResponseMixin:
    def extract_info_from_response(self, response):
        if "assistant\n" in response:
            response = response.split("assistant\n", 1)[1].strip()
        parsed = super().extract_info_from_response(response)
        if parsed is not None:
            return parsed
        try:
            groups = ast.literal_eval(response.strip().removeprefix("```python").removesuffix("```").strip())
        except (SyntaxError, ValueError, AttributeError):
            return None
        if not isinstance(groups, list):
            return None
        if groups and all(isinstance(item, int) for item in groups):
            groups = [groups]
        if not all(isinstance(group, list) for group in groups):
            return None
        goal = str(getattr(self, "_sft_current_goal", "")).strip().lower()
        expected_count = (self._expected_multi_goal_count(goal)
                          if hasattr(self, "_expected_multi_goal_count") else None)
        if (getattr(self, "multi_goal_count_prior", False) and goal.startswith("revisit all") and
                expected_count is not None and len(groups) != expected_count):
            candidates = [i for group in groups for i in group
                          if isinstance(i, int) and 0 <= i < 96]
            return {"frame_indices": sorted(set(candidates)) or [-1]}
        if (len(groups) == 1 and getattr(self, "single_goal_collapse", "off") != "off" and
                not goal.startswith("revisit all")):
            candidates = [i for i in groups[0] if isinstance(i, int) and 0 <= i < 96]
            return {"frame_indices": candidates or [-1]}
        indices = []
        for group in groups:
            candidates = [i for i in group if isinstance(i, int) and 0 <= i < 96]
            if candidates:
                # The released trained agent uses the first candidate in each
                # answer group. Keep that decoding for a comparable raw run.
                indices.append(candidates[0])
        return {"frame_indices": indices or [-1]}


def trained_video_response(agent, images, prompt, path=None):
    """Use the same video sampling and pixel budget as FindingDory SFT."""
    video_path = path or agent.save_video(images, fps=1)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]}]
    # Mirror the released FindingDory SFT evaluator's preprocessing exactly.
    text = agent.processor.apply_chat_template(messages, tokenize=False)
    video = process_vision_info(messages)[1][0]
    inputs = agent.processor(text=text, videos=video,
                             return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = agent.model.generate(**inputs, max_new_tokens=512,
                                         do_sample=False, use_cache=True)
    output = agent.processor.decode(generated[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    del inputs, generated
    return output


class SFTQwenImageNavAgent(SFTResponseMixin, QwenImageNavAgent):
    """Official raw SFT model with its published answer format decoded."""

    def __init__(self, config):
        super().__init__(config)
        if config.get("sft_prompt_mode", "official") == "question_only":
            self.prompt = SFT_QUESTION_PROMPT

    def get_vlm_response(self, images, prompt, path=None):
        return trained_video_response(self, images, prompt, path)
