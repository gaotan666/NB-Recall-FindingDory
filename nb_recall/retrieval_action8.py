"""Frozen action-first stratified retrieval. No answer labels are inputs."""
import re
ACTIONS=frozenset({'snap_object','desnap_object','manipulation_mode','full_body_action'})
GENERIC={'object','item','thing','receptacle','room','place','where','robot','navigate','goal','yesterday','all','any'}
def words(value):
    return {w[:-1] if w.endswith('s') and not w.endswith('ss') else w for w in re.findall(r'[a-z0-9]+',str(value).lower().replace('_',' '))}
def select_action8(question, observations):
    if [o['clip_index'] for o in observations]!=list(range(96)):
        raise ValueError('Expected exactly 96 aligned observations')
    query=words(question)-GENERIC
    categories=[];actions=[]
    for observation in observations:
        nodes=(observation.get('prediction') or {}).get('nodes',[])
        categories.append({str(n.get('content',{}).get('category') or '') for n in nodes if n.get('type')=='Fact'})
        actions.append({str(n.get('content',{}).get('action') or '') for n in nodes if n.get('type')=='Event'})
    mentioned={c for cs in categories for c in cs if words(c)&query}
    selected=[]
    for start in range(0,96,12):
        selected.append(max(range(start,start+12),key=lambda i:(bool(actions[i]&ACTIONS),len(categories[i]&mentioned),-abs(i-(start+5.5)))))
    return selected
