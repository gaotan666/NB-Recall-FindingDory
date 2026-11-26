"""Hard separation between NB-Recall memory-builder and answer LoRAs.

The two adapters share only the frozen Qwen2.5-VL-3B base model.  They are
trained independently and are never merged.  Inference may keep both named
adapters on one PEFT model, but exactly one role is active for each pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MEMORY_BUILDER_ROLE = "memory_builder"
ANSWER_ROLE = "answer"


def validate_dual_lora_config(config: dict[str, Any]) -> None:
    """Reject configurations that can accidentally resume Stage C from Stage A."""
    architecture = config["adapter_architecture"]
    stage_a = config["stage_a"]
    stage_c = config["stage_c"]

    if architecture.get("shared_adapter") is not False:
        raise ValueError("NB-Recall requires two independent LoRAs")
    if architecture.get("merge_adapters") is not False:
        raise ValueError("Memory-builder and answer LoRAs must never be merged")
    if stage_a.get("role") != MEMORY_BUILDER_ROLE:
        raise ValueError("Stage A must have the memory_builder role")
    if stage_c.get("role") != ANSWER_ROLE:
        raise ValueError("Stage C must have the answer role")
    if stage_c.get("initialize_fresh_from_base_model") is not True:
        raise ValueError("The answer LoRA must be initialized from the base model")
    if stage_c.get("resume_same_lora_from_stage_a") is not False:
        raise ValueError("Stage C may not resume the memory-builder LoRA")
    if float(stage_c.get("stage_a_replay_fraction", -1)) != 0.0:
        raise ValueError("Answer-LoRA training may not replay Stage-A writer targets")
    if Path(stage_a["adapter"]) == Path(stage_c["adapter_output"]):
        raise ValueError("The two adapter paths must be different")


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    validate_dual_lora_config(config)
    return config


def load_named_inference_adapters(model: Any, memory_builder: str | Path,
                                  answer: str | Path) -> Any:
    """Attach two separately trained adapters without merging their weights.

    This helper deliberately does not train either adapter.  Call
    ``model.set_adapter(MEMORY_BUILDER_ROLE)`` while constructing memory and
    ``model.set_adapter(ANSWER_ROLE)`` while answering.
    """
    from peft import PeftModel

    memory_builder = Path(memory_builder)
    answer = Path(answer)
    if memory_builder.resolve() == answer.resolve():
        raise ValueError("memory_builder and answer adapters resolve to one path")
    wrapped = PeftModel.from_pretrained(
        model, memory_builder, adapter_name=MEMORY_BUILDER_ROLE, is_trainable=False)
    wrapped.load_adapter(answer, adapter_name=ANSWER_ROLE, is_trainable=False)
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    return wrapped


def activate_role(model: Any, role: str) -> None:
    if role not in {MEMORY_BUILDER_ROLE, ANSWER_ROLE}:
        raise ValueError(f"Unknown adapter role: {role}")
    model.set_adapter(role)

