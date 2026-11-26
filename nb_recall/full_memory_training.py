"""One frozen full-memory encoding shared by answer training and inference.

Every frame, node content and relationship survives. Opaque ids are shortened
consistently; repeated provenance, null trust and duplicated scene wrappers are
not reader evidence. Original records remain untouched on disk.
"""
import json

FORMAT='full_memory_nodes_edges_v1'
SYSTEM=(
 'You answer questions about a robot video using its complete recorded memory. '
 'The memory contains all 96 sampled frames; frame indices are 0..95. '
 'Records are model predictions and may be imperfect. Resolve the requested '
 'objects, interactions, locations and temporal order from the records. '
 'Co-occurrence does not by itself prove a pick/place relation. Track the object '
 'from the relevant interaction to the location requested by the question. '
 'Use clip order for chronology; predicted clock/coordinates/entity identity may be inaccurate. '
 'Return exactly JSON with decision, frames and observe_frames. Choose one frame '
 'per requested destination, retaining all destinations and their requested order. '
 'Example format only: {"decision":"ANSWER","frames":[[12],[34]],"observe_frames":[]}. '
 'If a destination has no available historical frame, use [-1] for that destination. '
 'Do not return prose, metadata, every visible object, or all alternative frames.')
QUESTION_SUFFIX='\nAnswer with the specified JSON only. One selected frame per destination.'

CONTENT_KEYS={'attribute':'attr','category':'category','bbox_xyxy_1000':'box',
              'estimated_position':'xyz','depth_m':'depth','time_of_day':'clock',
              'visible_categories':'visible','action':'action'}

def encode_memory(observations):
    assert [o['clip_index'] for o in observations]==list(range(96))
    aliases={}
    def alias(value):
        if value is None:return None
        value=str(value)
        if value not in aliases:aliases[value]=len(aliases)
        return aliases[value]
    frames=[];node_count=edge_count=0
    for o in observations:
        p=o.get('prediction') or {};nodes=[];edges=[]
        for n in p.get('nodes',[]):
            content={CONTENT_KEYS.get(k,k):v for k,v in n.get('content',{}).items()}
            scope=n.get('support',{}).get('scope',{})
            nodes.append([alias(n.get('id')),n.get('type'),content,alias(scope.get('entity'))])
        for e in p.get('edges',[]):
            edges.append([alias(e.get('source')),e.get('type'),alias(e.get('target'))])
        frames.append([o['clip_index'],nodes,edges]);node_count+=len(nodes);edge_count+=len(edges)
    memory={'format':FORMAT,'frame_columns':['clip','nodes','edges'],
        'node_columns':['id','type','content','entity_id'],
        'edge_columns':['source_id','relation','target_id'],
        'content_legend':{v:k for k,v in CONTENT_KEYS.items()},
        'id_note':'Integer aliases consistently reference predicted node/entity ids within this video; aliases alone are not verified identity.',
        'frames':frames}
    return memory,{'frames':96,'nodes':node_count,'edges':edge_count,'id_aliases':aliases}

def prefix_messages(memory):
    return [{'role':'system','content':SYSTEM},
            {'role':'user','content':'Complete video memory:\n'+json.dumps(memory,ensure_ascii=False,separators=(',',':'))},
            {'role':'assistant','content':'Memory received.'}]

def prefix_text(tokenizer,memory):
    return tokenizer.apply_chat_template(prefix_messages(memory),tokenize=False,add_generation_prompt=False)

def question_tail(tokenizer,question):
    stub=prefix_messages({'format':FORMAT,'frames':[]})
    before=tokenizer.apply_chat_template(stub,tokenize=False,add_generation_prompt=False)
    after=tokenizer.apply_chat_template(stub+[{'role':'user','content':question+QUESTION_SUFFIX}],
                                       tokenize=False,add_generation_prompt=True)
    assert after.startswith(before)
    return after[len(before):]

def canonical_answer(groups):
    # All official alternatives remain scoring labels; choose a representative
    # solely for training supervision, never to retrieve input memory.
    return {'decision':'ANSWER','frames':[([-1] if g==[-1] else [g[len(g)//2]] if g else []) for g in groups],
            'observe_frames':[]}
