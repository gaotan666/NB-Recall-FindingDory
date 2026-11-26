"""Question-specific evidence repair; diagnoses are hypotheses, never gold labels."""
import json
import re
from nb_recall.memory_evidence_patch import validate_patch

KINDS = {'missing_evidence', 'wrong_relation', 'wrong_identity', 'missing_order',
         'retrieval_gap', 'reader_misuse', 'supported_answer', 'insufficient_visual_evidence'}
SYSTEM = '''Audit ONE question and the exact memory the answer model actually read.
The previous answer may be correct or incorrect. Do not assume it is wrong.
You see its selected Clip indices, its memory context, independently extracted visual evidence,
and source images. The independent pass saw all 96 chronological RGB images WITHOUT the old answer or memory.
Step 1: interpret what the question asks (object vs receptacle, source vs destination, event order).
Identify the entity satisfying the question from visual events BEFORE considering the previous selected entity.
Never assume the entity visible in the previous answer is the entity requested by the question.
Step 2: inspect the previously selected frames. Describe what is actually visible there.
Step 3: compare the needed evidence with the supplied memory. State the SPECIFIC missing or mistaken link.
If sufficient evidence already exists, diagnose reader_misuse/retrieval_gap instead of inventing missing facts.
Step 4: append only observations that fill this question's gap. Attach before/after Clip evidence.
For a moved object, connect its pickup source to its later placement; seeing a pickup alone may not locate it now.
For ordering, recover the complete relevant event sequence. Repeated views of one event are not separate interactions.
For negative conditions, absence from a sparse memory does not prove no interaction.
An object disappearing when the arm retracts means it may be carried, not placed back.
Object/category names in old memory can be wrong; identify objects from RGB. Control modes are not interaction events.
Do not justify a proposed fact using 'likely', 'probably', 'same surface', or analogy with another object.
The diagnosis must separate an incorrect selected entity from missing information about that entity.
Do not output an answer frame list or an instruction to select specific frames. Do not use benchmark answer labels.
Do not delete/overwrite memory. A corrected observation is an additional source-linked record.
Return compact JSON:
{"diagnosis":{"kind":"missing_evidence","selected_frame_observation":"what the old selected frame shows",
"memory_gap":"specific missing or false link, or none","needed_evidence":"specific evidence to resolve it",
"memory_refs":["clip:17"],"evidence_clips":[15,17,41],"certainty":"clear"},
"operation":"APPEND","facts":[{"s":"subject","r":"picked_from","o":"support","b":15,"a":17,
"q":[0],"c":"clear","why":"short visual justification"}],"unresolved":[]}.
kind is missing_evidence/wrong_relation/wrong_identity/missing_order/retrieval_gap/reader_misuse/supported_answer/insufficient_visual_evidence.
memory_refs must refer to IDs present in input memory or clip:N for a frame present in input memory.
selected_frame_observation must describe the ACTUAL previous_selected_clips, not a different retrieved frame.
diagnosis.evidence_clips must include every nonnegative previous_selected_clips entry, plus needed evidence frames.
Only supplied Clip 0..95 labels are valid; ignore the original Frame overlay numbers.
Allowed fact relations: observed_on, picked_from, placed_on, carried, before, same_entity.
Use facts=[] for supported_answer, reader_misuse or insufficient_visual_evidence unless a genuinely new fact is necessary.
Give only distinct question-relevant facts, at most 24; do not enumerate unchanged objects frame by frame.
Do not include free-form reasoning. Keep each diagnostic field under 45 words and each why under 15 words.'''

GUIDES = {
 'missing_evidence': 'Use the source-linked additions to fill the identified evidence gap; distinguish pickup source from later placement.',
 'wrong_relation': 'Check the relation against the cited before/after images. Being visible nearby is not being supported on or picked from.',
 'wrong_identity': 'Reconcile the object by visible appearance and continuity across events; category labels alone do not establish identity.',
 'missing_order': 'Build a chronological list of distinct physical interactions, then apply the order requested in the question. Repeated views are one event.',
 'retrieval_gap': 'Use the focused evidence references as well as the original memory. Retrieval rank is not temporal order.',
 'reader_misuse': 'Re-read the existing evidence: distinguish the entity selected by the question from its location, preserve every requested destination and requested order.',
}


def known_refs(context):
    refs = set()
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('id'), str): refs.add(value['id'])
            if type(value.get('clip_index')) is int: refs.add(f"clip:{value['clip_index']}")
            if type(value.get('clip')) is int: refs.add(f"clip:{value['clip']}")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(context)
    return refs


def parse_case(raw, job):
    decoder = json.JSONDecoder()
    value = decoder.raw_decode(raw[raw.index('{'):])[0]
    d = value.get('diagnosis')
    if not isinstance(d, dict) or d.get('kind') not in KINDS: raise ValueError('Invalid diagnosis kind')
    for key in ['selected_frame_observation', 'memory_gap', 'needed_evidence']:
        if not isinstance(d.get(key), str) or not d[key].strip(): raise ValueError(f'Missing diagnostic field {key}')
    clips = d.get('evidence_clips'); refs = d.get('memory_refs')
    if not isinstance(clips, list) or not all(type(i)==int and 0<=i<96 for i in clips): raise ValueError('Invalid diagnostic frame references')
    selected={i for i in job['previous_selected_clips'] if type(i)==int and 0<=i<96}
    if not selected<=set(clips):raise ValueError('Diagnosis did not inspect all actual selected frames')
    if not isinstance(refs, list) or not all(isinstance(r, str) and r in known_refs(job['reader_context']) for r in refs):
        raise ValueError('Diagnostic memory reference was not in model input')
    if d.get('certainty') not in {'clear', 'uncertain'}: raise ValueError('Missing diagnostic certainty')
    if value.get('operation') != 'APPEND' or not isinstance(value.get('facts'), list): raise ValueError('Expected APPEND facts')
    if len(value['facts']) > 24: raise ValueError('Repetitive per-question output exceeds 24 facts')
    nodes = []; rejected = []
    for fact in value['facts']:
        if re.search(r'\b(likely|probably|presumably|must have|by analogy)\b',str(fact.get('why','')),re.I):
            raise ValueError('Speculation cannot be written as a visual fact')
        accepted, errors = validate_patch({'operation':'APPEND','facts':[fact]}, job)
        nodes.extend(accepted); rejected.extend(errors)
    if rejected: raise ValueError(f'Invalid diagnostic facts: {rejected[:1]}')
    for node in nodes:
        node['provenance'].update(diagnosis_kind=d['kind'], diagnosis_is_hypothesis=True,
                                  method='single-question comparison with actual reader input')
    return d, list({n['id']:n for n in nodes}.values())
