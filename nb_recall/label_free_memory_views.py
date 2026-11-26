"""Question/original-observation-only readers; no answer-label API."""
import copy
import collections
import re

METHODS = ('baseline8_index96', 'event_chain', 'all_facts96')


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower().replace('_', ' ')))


def contents(observation, kind):
    return [n for n in (observation.get('prediction') or {}).get('nodes', [])
            if n.get('type') == kind]


def fact(n):
    c=n.get('content', {}); entity=n.get('support', {}).get('scope', {}).get('entity') or ''
    return {'category':c.get('category'), 'predicted_entity':entity,
            'bbox':c.get('bbox_xyxy_1000'), 'estimated_position':c.get('estimated_position'),
            'depth':c.get('depth_m')}


def timeline(observations):
    return [{'clip':o['clip_index'],
             'objects':sorted({str(n.get('content',{}).get('category')) for n in contents(o,'Fact')}),
             'actions':[n.get('content',{}).get('action') for n in contents(o,'Event')],
             'predicted_clock':[n.get('content',{}).get('time_of_day') for n in contents(o,'Event')]}
            for o in observations]


def runs(indices):
    result=[]
    for i in sorted(set(indices)):
        if result and i==result[-1][1]+1:result[-1][1]=i
        else:result.append([i,i])
    return result


def build_view(question, observations, original_context, method):
    assert method in METHODS
    assert [o['clip_index'] for o in observations]==list(range(96))
    # Explicit whitelist; never accept repaired-memory fields or answer groups.
    context={k:copy.deepcopy(original_context[k]) for k in
             ('memory','timeline','tau_lo','tau_hi') if k in original_context}
    context.update(question=question,task='READ',phase='initial',evidence_revisited=False)
    if method=='baseline8_index96':return context
    context['timeline']=timeline(observations)
    context['evidence_notes']=(
        'All entries are model predictions, not oracle truth. Clip indices are chronological; '
        'predicted clock, position and entity ids may be inaccurate. Co-occurrence alone '
        'does not prove which object was picked or placed. Use adjacent observations to '
        'establish the interaction object, then track that object to the requested destination. '
        'An interaction-source frame is not automatically the requested final location. '
        'Determine the number and order of destinations from the question only.')
    if method=='all_facts96':
        context['memory']=[{'clip_index':o['clip_index'],'type':'FrameMemory',
            'content':{'facts':[fact(n) for n in contents(o,'Fact')],
                       'events':[n.get('content',{}) for n in contents(o,'Event')]}}
            for o in observations]
        return context
    query=words(question)
    context['query_conditions']={
        'mentions_pick':bool(query & {'picked','pick','grabbed','grab'}),
        'mentions_place':bool(query & {'placed','place','put'}),
        'mentions_order':bool(query & {'first','last','before','after','order','ordered','then'}),
        'mentions_duration':bool(query & {'longest','shortest','duration','long','time'}),
        'mentions_distance':bool(query & {'nearest','closest','farthest','furthest','distance'})}
    tracks=collections.defaultdict(list); kinds=collections.defaultdict(set)
    for o in observations:
        for n in contents(o,'Fact'):
            category=str(n.get('content',{}).get('category') or '')
            if not category:continue
            tracks[category].append(o['clip_index'])
            entity=str(n.get('support',{}).get('scope',{}).get('entity') or '')
            kinds[category].add('object' if ':object:' in entity or ':obj:' in entity else 'unknown_or_receptacle')
    context['category_occurrences']=[{'category':cat,'clip_ranges':runs(ids),
         'predicted_kinds':sorted(kinds[cat]),'not_an_instance_identity':True}
         for cat,ids in sorted(tracks.items())]
    anchors=[]
    for o in observations:
        i=o['clip_index']; actions={str(n.get('content',{}).get('action')) for n in contents(o,'Event')}
        if not actions & {'snap_object','desnap_object','manipulation_mode'}:continue
        nearby=observations[max(0,i-2):min(96,i+3)]
        categories={str(n.get('content',{}).get('category') or '') for v in nearby for n in contents(v,'Fact')}
        score=5*sum(bool(query & words(cat)) for cat in categories)
        score+=4*bool(actions & {'snap_object','desnap_object'})
        score+=3*bool(context['query_conditions']['mentions_pick'] and 'snap_object' in actions)
        score+=3*bool(context['query_conditions']['mentions_place'] and 'desnap_object' in actions)
        anchors.append((score,i))
    # All 96 frames remain in the index. Up to 12 nonredundant interaction windows
    # are expanded, selected exclusively from the query and predicted actions.
    chosen=[]
    for _,i in sorted(anchors,key=lambda x:(-x[0],x[1])):
        if all(abs(i-j)>2 for j in chosen):chosen.append(i)
        if len(chosen)>=12:break
    expanded=sorted({j for i in chosen for j in range(max(0,i-2),min(96,i+3))})
    context['interaction_windows']=[{'anchor_clip':i,'window':[max(0,i-2),min(95,i+2)]}
                                    for i in sorted(chosen)]
    context['interaction_observations']=[{'clip':i,
       'facts':[fact(n) for n in contents(observations[i],'Fact')],
       'events':[n.get('content',{}) for n in contents(observations[i],'Event')]}
       for i in expanded]
    return context
