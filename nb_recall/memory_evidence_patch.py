"""Append-only, source-linked evidence supplements; no answer labels accepted."""
import hashlib
import json


def parse_evidence_output(raw, job):
    """Keep only fully decoded, schema-valid facts; never complete a truncated fact.

    Candidate budget enforcement is deterministic and question/label blind beyond
    the supplied question references. Visual verification still follows this step.
    """
    decoder = json.JSONDecoder()
    partial = False
    try:
        value, end = decoder.raw_decode(raw[raw.index('{'):])
        if not isinstance(value, dict) or value.get('operation') != 'APPEND':
            raise ValueError('Expected APPEND object')
        facts = value.get('facts')
        if not isinstance(facts, list):
            raise ValueError('Expected facts list')
    except (ValueError, TypeError):
        # Salvage only the complete prefix of an APPEND facts array.
        import re
        start = raw.find('{')
        if start < 0 or not re.search(r'"operation"\s*:\s*"APPEND"', raw[start:]):
            raise ValueError('No APPEND prefix to recover')
        match = re.search(r'"facts"\s*:\s*\[', raw[start:])
        if not match:
            raise ValueError('No facts array to recover')
        cursor = start + match.end(); facts = []; partial = True
        while cursor < len(raw):
            while cursor < len(raw) and raw[cursor] in ' \r\n\t,':
                cursor += 1
            if cursor >= len(raw) or raw[cursor] == ']':
                break
            try:
                fact, cursor = decoder.raw_decode(raw, cursor)
            except ValueError:
                break
            facts.append(fact)
    valid = []; rejected = []; seen = set()
    for index, fact in enumerate(facts):
        nodes, errors = validate_patch({'operation': 'APPEND', 'facts': [fact]}, job)
        if errors:
            rejected.extend({**e, 'source_index': index} for e in errors)
        for node in nodes:
            if node['id'] not in seen:
                valid.append(node); seen.add(node['id'])
    if partial and not valid:
        raise ValueError('Truncated output has no complete valid fact')
    # Spread the limited candidate budget over distinct subjects/relations and time.
    selected = []; covered_questions = set(); covered_events = set(); covered_time = set()
    pool = list(enumerate(valid))
    while pool and len(selected) < 12:
        def priority(item):
            index, node = item; c = node['content']
            event = (c['subject'], c['relation'], c['object'])
            return (len(set(node['provenance']['question_ids']) - covered_questions),
                    event not in covered_events, c['before_clip'] // 12 not in covered_time, -index)
        chosen = max(pool, key=priority); pool.remove(chosen); node = chosen[1]; c = node['content']
        selected.append(node); covered_questions.update(node['provenance']['question_ids'])
        covered_events.add((c['subject'], c['relation'], c['object'])); covered_time.add(c['before_clip'] // 12)
    metadata = {'normalizer': 'complete_facts_temporal_budget_v1', 'partial_raw_output': partial,
                'complete_facts_decoded': len(facts), 'valid_candidates': len(valid),
                'omitted_by_budget': max(0, len(valid) - len(selected)),
                'missing_content_invented': False}
    return selected, rejected, metadata

SYSTEM = '''You inspect a robot's chronological video to supplement incomplete memory.
You receive questions to identify what evidence is needed, but you must NOT answer them or output destination answers.
Inspect ALL supplied images. Supplied memory categories and action modes are noisy predictions, not ground truth.
Record concrete observations needed to distinguish objects, pickup sources, placement destinations and event order.
Seeing an object is not interacting with it. full_body_action is a control mode, not proof of pickup/placement.
A pickup requires before/after evidence that an object leaves a support and is carried by the gripper.
A placement requires before/after evidence that the carried object is released onto a support.
Preserve object identity across events when visually supported; otherwise explicitly say identity is uncertain.
For ordering questions, cover every relevant pickup and placement, not only the first visible object.
For negation, describe what was observed and unknown coverage; do not assume unobserved means never interacted.
Use supplied Clip 0..95 labels, NEVER the original Frame text printed in images.
Only append evidence; do not alter existing memory. Do not guess unseen events or use any benchmark answer.
Return compact JSON with this exact short-key schema:
{"operation":"APPEND","facts":[{"s":"object identity/category","r":"picked_from","o":"support identity/category","b":12,"a":15,"q":[0,2],"c":"clear","why":"object leaves support and becomes carried"}],"unresolved":[]}.
s=subject, r=relation, o=object/support, b=before Clip, a=after Clip, q=zero-based question numbers, c=clear or uncertain.
Allowed relations: observed_on, picked_from, placed_on, carried, before, same_entity.
For static observation b may equal a. For pickup, placement or before, b must be less than a.
The why field must be a short visual justification, not reasoning or a final answer.
STRICT BUDGET: at most 12 facts TOTAL, not per question. Do not enumerate the same static object frame by frame.
Prioritize distinct pickup/placement transitions and their order across the whole video. Include at most 2 static observations.
Use the SAME subject name for the same physical object and different descriptive names for different instances.
observed_on means physically supported on, not merely near. A chair beside a bed is NOT a chair on a bed.
Output one compact JSON object without markdown or indentation. Finish the object before stopping. Never output final destination answers.'''


def validate_patch(value, job):
    if not isinstance(value,dict) or value.get('operation')!='APPEND' or not isinstance(value.get('facts'),list):
        raise ValueError('Invalid APPEND evidence schema')
    known={q['id'] for q in job['questions']};accepted=[];rejected=[]
    for idx,fact in enumerate(value['facts']):
        reason=None
        if isinstance(fact,dict) and 's' in fact:
            short=fact
            qindices=short.get('q',[])
            if isinstance(qindices,list) and all(type(i)==int and 0<=i<len(job['questions']) for i in qindices):
                relation=short.get('r')
                fact={'kind':'interaction' if relation in {'picked_from','placed_on','carried'} else 'order' if relation=='before' else 'identity' if relation=='same_entity' else 'observation',
                    'subject':short.get('s'),'relation':relation,'object':short.get('o'),
                    'before_clip':short.get('b'),'after_clip':short.get('a'),
                    'evidence_clips':list(dict.fromkeys([short.get('b'),short.get('a')])),
                    'description':short.get('why'),'question_ids':[job['questions'][i]['id'] for i in qindices],
                    'certainty':short.get('c')}
        if not isinstance(fact,dict):reason='not_object'
        else:
            clips=fact.get('evidence_clips');qids=fact.get('question_ids');before=fact.get('before_clip');after=fact.get('after_clip')
            if fact.get('kind') not in {'observation','interaction','order','identity'}:reason='kind'
            elif fact.get('relation') not in {'observed_on','picked_from','placed_on','carried','before','same_entity'}:reason='relation'
            elif not all(isinstance(fact.get(k),str) and fact[k].strip() for k in ['subject','object','description']):reason='missing_content'
            elif not isinstance(clips,list) or not clips or not all(type(i)==int and 0<=i<96 for i in clips):reason='invalid_evidence_clips'
            elif not isinstance(qids,list) or not qids or not set(qids)<=known:reason='unknown_question'
            elif type(before)!=int or type(after)!=int or not 0<=before<=after<96:reason='invalid_interval'
            elif before not in clips or after not in clips:reason='endpoints_without_evidence'
            elif fact.get('relation') in {'picked_from','placed_on','before'} and before==after:reason='transition_requires_two_frames'
            elif fact.get('certainty') not in {'clear','uncertain'}:reason='certainty'
        if reason:rejected.append({'index':idx,'reason':reason,'raw':fact});continue
        digest=hashlib.sha256(json.dumps(fact,sort_keys=True).encode()).hexdigest()[:16]
        accepted.append({'id':f"evidence:{job['episode']}:{digest}",'type':'Event' if fact['kind'] in {'interaction','order'} else 'Fact',
            'content':{k:fact[k] for k in ['kind','subject','relation','object','before_clip','after_clip','description','certainty']},
            'support':{'evidence_clips':sorted(set(clips)),'source_images':[f"data/stage_c_video_memory/{job['memory_split']}/{job['episode']}/clip_{i:02d}.jpg" for i in sorted(set(clips))]},
            'provenance':{'extractor':'frozen Qwen3.5-9B','job_id':job['job_id'],'question_ids':qids,
                'uses_answer_labels':False,'semantic_truth_verified':False},'calibrated_trust':None})
    if len(value['facts'])>12:
        raise ValueError('Fact budget exceeded; likely repetitive output')
    return accepted,rejected


READER_GUIDE = (
    'The appended evidence records describe observed events, with supporting clip indices; '
    'they are predictions, not verified answers. Use clear evidence and check conflicts against images when available. '
    'Separate WHICH entity the question selects from WHERE that entity should be revisited. '
    'An object selected by pickup order may later have been moved: use its last supported placement location '
    'unless the question explicitly asks for its pickup/source location. '
    'A receptacle picked FROM refers to the pickup support; placed ON refers to the placement support. '
    'For multiple destinations, keep every distinct physical entity and preserve the requested order; '
    'two different chairs are not one destination just because both are chairs. '
    'Use chronological before/after evidence to establish first/last/second; do not rank by retrieval order. '
    'Unknown or merely unseen interactions do not establish a did-not-interact condition.'
)
