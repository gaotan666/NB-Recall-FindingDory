"""Freeze validation predictions for official simulator HL-SR verification."""
import gzip
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=700)
args = parser.parse_args()
run = ROOT / f'reports/stage_c_validation_{args.step}_framepack'
pred_file = run / 'shard_-1.jsonl'
rows = pq.read_table(ROOT / 'data/findingdory-subsampled-96/data/validation-00000-of-00001.parquet',
                     columns=['ep_id', 'task_id', 'high_level_category', 'low_level_category']).to_pylist()
predictions = {r['question_id']: r for r in map(json.loads, pred_file.open())}
if len(rows) != 5876 or len(predictions) != len(rows):
    raise RuntimeError('Expected all 5,876 frozen validation predictions')
sim_episodes = {e['episode_id']: e for e in json.load(gzip.open(
    ROOT / 'data/findingdory-habitat/findingdory/val/episodes.json.gz', 'rt'))['episodes']}
grouped = defaultdict(dict)
missing = []
skipped = []
indices = defaultdict(int)
for row in rows:
    ep = row['ep_id']
    question_id = f'{ep}:{indices[ep]}'
    indices[ep] += 1
    item = predictions[question_id]
    if row['task_id'] in grouped[ep]:
        raise RuntimeError(f'Duplicate simulator instruction: {ep} {row["task_id"]}')
    if row['task_id'] not in sim_episodes[ep.split('_')[-1]]['instructions']:
        missing.append(question_id)
        continue
    # The official FindingDory task runner skips task_30 on these four
    # episodes because its solution entity is absent from the scene.
    if row['task_id'] == 'task_30' and ep in {'ep_17', 'ep_35', 'ep_59', 'ep_66'}:
        skipped.append(question_id)
        continue
    value = item['prediction']
    # The official high-level action accepts one waypoint per destination.
    # This conversion uses only the model response, never a gold frame label.
    frame_indices = [group[0] for group in value] if item['valid'] and value and all(group for group in value) else [-1]
    if any(type(index) is not int or index < 0 or index >= 96 for index in frame_indices):
        frame_indices = [-1]
    grouped[ep][row['task_id']] = {
        'question_id': question_id,
        'frame_indices': frame_indices,
        'model_decision': item['final_decision'],
        'category': row['high_level_category'],
        'low_level_category': row['low_level_category'],
    }
for episode_id, episode in sim_episodes.items():
    ep = f'ep_{episode_id}'
    expected = set(episode['instructions'])
    if ep in {'ep_17', 'ep_35', 'ep_59', 'ep_66'}:
        expected.discard('task_30')
    if expected != set(grouped[ep]):
        raise RuntimeError(f'Missing prediction for simulator task: {ep}')
adapter = ROOT / f'models/nb_recall_answer_lora_stage_c/checkpoint-{args.step}/adapter_model.safetensors'
payload = {
    'scope': f'official simulator high-level success, frozen stage-C step-{args.step} predictions',
    'prediction_source': str(pred_file.relative_to(ROOT)),
    'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
    'official_simulator_episode_count': len(sim_episodes),
    'official_simulator_task_count': sum(map(len, grouped.values())),
    'video_rows_without_simulator_instruction': missing,
    'official_runner_skipped_questions': skipped,
    'episodes': grouped,
}
out = run / 'hlsr_cached_predictions.json'
temporary = out.with_suffix('.json.tmp')
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
temporary.replace(out)
print(json.dumps({k: payload[k] for k in ('adapter_sha256', 'official_simulator_episode_count',
                                          'official_simulator_task_count',
                                          'video_rows_without_simulator_instruction')}, ensure_ascii=False))
