"""Replace a mismatched video's predictions with same-checkpoint replay predictions."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, required=True)
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
ep = f'ep_{args.episode}'
base = ROOT / f'reports/stage_c_validation_{args.step}_framepack'
original = json.loads((base / 'hlsr_cached_predictions.json').read_text())
merged_path = base / 'hlsr_sim_replay_predictions.json'
merged = json.loads(merged_path.read_text()) if merged_path.exists() else original
if merged['adapter_sha256'] != original['adapter_sha256']:
    raise RuntimeError('Mixed model checkpoints in replay prediction manifest')
answer_path = base / f'sim_replay_answer_{ep}/shard_-1.jsonl'
answers = {row['question_id']: row for row in map(json.loads, answer_path.open())}
expected = merged['episodes'][ep]
if set(answers) != {item['question_id'] for item in expected.values()}:
    raise RuntimeError(f'Incomplete simulator-replay answers for {ep}')
for item in expected.values():
    answer = answers[item['question_id']]
    groups = answer['prediction']
    selected = [group[0] for group in groups] if answer['valid'] and groups and all(groups) else [-1]
    if any(type(index) is not int or not 0 <= index < 96 for index in selected):
        selected = [-1]
    item['frame_indices'] = selected
    item['model_decision'] = answer['final_decision']
merged.setdefault('sim_replay_episode_ids', [])
merged['sim_replay_episode_ids'] = sorted(set(merged['sim_replay_episode_ids']) | {ep},
                                          key=lambda value: int(value.split('_')[-1]))
temporary = merged_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
temporary.replace(merged_path)
print(json.dumps({'step': args.step, 'episode': ep,
                  'questions': len(answers),
                  'manifest': str(merged_path.relative_to(ROOT))}), flush=True)
