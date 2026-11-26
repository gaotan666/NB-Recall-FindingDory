"""Query-focused, append-only evidence for the full-image revisit branch.

Visible clocks are supplied by a separate question-blind OCR cache. They remain
predictions, not oracle annotations. No answer labels are accepted by the builder.
"""
import copy
import re


def query_evidence(question, observations, clock_by_clip):
    timeline = []
    for observation in observations:
        i = observation['clip_index']
        objects = []
        for node in (observation.get('prediction') or {}).get('nodes', []):
            if node.get('type') != 'Fact':
                continue
            c = node.get('content', {})
            box = c.get('bbox_xyxy_1000')
            area = round(max(0, box[2]-box[0])*max(0, box[3]-box[1])/1e6, 3) if box and len(box)==4 else None
            objects.append({'category': c.get('category'),
                'entity': node.get('support', {}).get('scope', {}).get('entity'),
                'image_area': area, 'depth_m_estimate': c.get('depth_m')})
        timeline.append({'clip': i, 'clock': clock_by_clip.get(i), 'objects': objects})
    addition = {}
    requested_time = re.search(r'\bat\s+(\d{1,2}):(\d{2})\b', question, re.I)
    if requested_time:
        minutes = 60*int(requested_time[1])+int(requested_time[2])
        readable = [t for t in timeline if t['clock']]
        if readable:
            def delta(t):
                return abs(60*int(t['clock'][:2])+int(t['clock'][3:])-minutes)
            addition['clock_candidates'] = [{'clip': t['clip'], 'clock': t['clock']} for t in sorted(readable, key=delta)[:5]]
            addition['instruction'] = 'These times were read from the visible Time of Day overlay. Inspect the interaction near the requested time, not the original Frame label or an unrelated later event.'
    elif re.search(r'\b(shortest|longest|most time|least time)\b', question, re.I):
        addition['visible_clock_by_clip'] = [[t['clip'], t['clock']] for t in timeline]
        addition['instruction'] = 'Use elapsed visible clock time across a pickup-to-placement interval of the same object (or total visits to the same room if asked). Control-mode labels alone do not prove a pickup or placement.'
    obj = re.search(r'Navigate to (?:a|an) ([\w ]+?)(?: you | that you |\.$)', question, re.I)
    if obj:
        name = obj[1].lower().replace('_', ' ')
        matches = [(t,o) for t in timeline for o in t['objects'] if str(o['category']).replace('_',' ').lower()==name]
        if matches:
            addition['requested_category'] = name
            addition['category_observations'] = [{'clip': t['clip'], 'area': o['image_area'], 'depth_estimate': o['depth_m_estimate'], 'entity': o['entity']} for t,o in matches]
            addition['category_instruction'] = 'These are predicted category observations, not verified destinations. Track the object into nearby views and choose a frame closest to the requested object/place, not merely a distant view. Check any interaction condition separately. Nearby frames may mislabel the same object.'
            if len(matches)<=3:
                neighbors = {j for t,_ in matches for j in range(max(0,t['clip']-4), min(96,t['clip']+9))}
                addition['neighbor_observations'] = [{'clip': t['clip'], 'clock': t['clock'], 'objects': [o['category'] for o in t['objects']]} for t in timeline if t['clip'] in neighbors]
    return addition


def augment_revisited_row(row, observations, clock_by_clip):
    """Keep the normal reader unchanged; append only after full-frame reobservation."""
    if len(row.get('evidence', [])) != 96:
        return row
    result = copy.deepcopy(row)
    extra = query_evidence(row['context']['question'], observations, clock_by_clip)
    if extra:
        result['context']['query_evidence'] = extra
    return result
