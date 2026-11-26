"""Shared prompts, retrieval and data contracts for the independent answer LoRA."""
import ast
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are the NB-Recall memory reader. Use the question, retrieved append-only "
    "memories and supplied evidence images. Clip indices are 0..95; ignore any "
    "original Frame label in images. A destination group [-1] means no historical "
    "frame would achieve that destination; it is not an image index. Preserve temporal order and separate "
    "destinations into nested groups. Return JSON only with decision "
    "ANSWER, REOBSERVE or ABSTAIN, frames (nested lists), and observe_frames "
    "(a flat list). REOBSERVE requests available evidence images and must have "
    "frames=[]. ABSTAIN also has frames=[]. The context task separates two skills: "
    "READ asks you to find destination frames from memory/evidence; a controller handles "
    "whether to call this reader, so do not turn READ into a copy of the gate decision. "
    "For GATE, use the supplied tau thresholds to choose ANSWER/REOBSERVE/ABSTAIN. "
    "For READ choose one supported clip per destination group, preserving order. "
    "Do not invent unseen evidence."
)


def validate_answer(answer):
    value = ast.literal_eval(answer) if isinstance(answer, str) else answer
    if not isinstance(value, list) or not all(isinstance(group, list) for group in value):
        raise ValueError("Expected a nested list of frame groups")
    for group in value:
        if group == [-1] and type(group[0]) is int:
            continue
        if not all(type(i) is int and 0 <= i < 96 for i in group):
            raise ValueError("Expected clip indices 0..95 or the singleton sentinel [-1]")
    return value


def validate_prediction(value):
    if set(value) != {"decision", "frames", "observe_frames"}:
        raise ValueError("Unexpected answer policy schema")
    if value["decision"] not in {"ANSWER", "REOBSERVE", "ABSTAIN"}:
        raise ValueError("Invalid answer-policy decision")
    validate_answer(value["frames"])
    if not isinstance(value["observe_frames"], list) or not all(
            type(i) is int and 0 <= i < 96 for i in value["observe_frames"]):
        raise ValueError("Invalid observe_frames")
    if value["decision"] != "ANSWER" and value["frames"]:
        raise ValueError("Only ANSWER can return answer frames")
    return value


def all_destination_groups_hit(predicted,gold,unordered=False):
    """One selected frame per valid destination suffices; missing groups fail."""
    if len(predicted)!=len(gold):return False
    def compatible(selected,valid):
        return bool(set(selected)&set(valid)) if valid else not selected
    if not unordered:return all(compatible(p,g) for p,g in zip(predicted,gold))
    def assign(index,used):
        if index==len(predicted):return True
        return any(j not in used and compatible(predicted[index],g) and
                   assign(index+1,used|{j}) for j,g in enumerate(gold))
    return assign(0,set())


def make_messages(row):
    content = [{"type": "text", "text": json.dumps(row["context"], ensure_ascii=False)}]
    for evidence in row.get("evidence", []):
        content.extend([
            {"type": "text", "text": f"Evidence Clip index: {evidence['clip_index']}"},
            {"type": "image", "image": str(ROOT / evidence["image"]),
             "max_pixels": 256 * 256},
        ])
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def tokens(text):
    words=re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return {word[:-1] if word.endswith('s') and not word.endswith('ss') else word for word in words}


def query_scope_match(question, node):
    """Map dataset superclasses to the predicted entity scope, without gold labels."""
    words=tokens(question)
    content=node.get('content',{})
    cat=str(content.get('category') or '')
    cat_words=tokens(cat)
    if cat_words and words & cat_words:
        return True
    entity=str(node.get('support',{}).get('scope',{}).get('entity') or '')
    if 'receptacle' in words and ':receptacle:' in entity:
        return True
    if words & {'object','item','thing'} and (':object:' in entity or ':obj:' in entity):
        return True
    if node.get('type')=='Event':
        action_words=tokens(str(content.get('action') or ''))
        if words & action_words & {'pick','place','open','close','snap','desnap'}:
            return True
        if words & {'room','first','last','before','after','order'}:
            return bool(content.get('visible_categories'))
    return False


def retrieval_features(question, candidate, candidates, source_reliability, embedder):
    """Exactly six fields; input contains predictions and trusted frame metadata only."""
    node = candidate["node"]
    cat = str(node.get("content", {}).get("category") or "unknown")
    text = cat + " " + json.dumps(node.get("content", {}), ensure_ascii=False)
    semantic = embedder.similarity(text, question)
    relevant = query_scope_match(question,node)
    # Preserve Stage-B feature scaling while computing it from the actual question.
    semantic = (0.7 + 0.3 * semantic) if relevant else 0.3 * semantic
    scope = node.get("support", {}).get("scope", {})
    entity = scope.get("entity")
    time = candidate["clip_index"]
    # Historical questions must not equate later versions with invalid evidence.
    historical = bool(re.search(r"\b(first|earliest|before|previous|yesterday|originally)\b", question.lower()))
    reference_time = time if historical else 95
    newer = [c for c in candidates if entity and
             c["node"].get("support", {}).get("scope", {}).get("entity") == entity
             and time < c["clip_index"] <= reference_time]
    conflicts = sum(e.get("type") == "conflicts-with" and
                    node.get("id") in (e.get("source"), e.get("target"))
                    for e in candidate.get("edges", []))
    return {
        "semantic_match": semantic,
        "provenance_freshness": math.exp(-max(0, reference_time-time) / 100.0),
        "version_consistency": 1 / (1 + len(newer)),
        "conflict_degree": min(1.0, conflicts / max(1, len(newer)+1)),
        "support_coverage": float(relevant),
        "source_reliability": source_reliability.get(cat, 0.5),
    }
