"""Aggregate all official simulator high-level success decisions."""
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=700)
args = parser.parse_args()
base = ROOT / f'reports/stage_c_validation_{args.step}_framepack'
source = json.loads((base / 'hlsr_cached_predictions.json').read_text())
records = []
alignments = []
history_video_sources = {}
for episode_id in range(1, 101):
    episode = f'ep_{episode_id}'
    path = base / 'hlsr_official' / f'{episode}.json'
    if not path.exists():
        raise RuntimeError(f'Official simulator episode not yet evaluated: {episode}')
    payload = json.loads(path.read_text())
    if payload['adapter_sha256'] != source['adapter_sha256']:
        raise RuntimeError(f'Checkpoint mismatch for {episode}')
    if payload['questions'] != len(source['episodes'][episode]):
        raise RuntimeError(f'Question count mismatch for {episode}')
    alignments.append(payload['alignment'])
    history_video_sources[episode] = payload.get('alignment_video',
        f'data/findingdory-subsampled-96/videos/val/{episode}.mp4')
    records.extend(payload['results'])
ids = [row['question_id'] for row in records]
if len(records) != source['official_simulator_task_count'] or len(ids) != len(set(ids)):
    raise RuntimeError('Official task coverage is incomplete or contains duplicates')


def summarize(items):
    count = len(items)
    successes = sum(bool(row['high_level_goal_success']) for row in items)
    return {'tasks': count, 'successes': successes,
            'hlsr': successes / count if count else None}


by_high = defaultdict(list)
by_low = defaultdict(list)
for record in records:
    by_high[record['category']].append(record)
    by_low[record['low_level_category']].append(record)
report = {
    'metric': 'official FindingDory High-Level Success Rate (HL-SR)',
    'official_simulator_validation_episodes': 100,
    'model_checkpoint': f'models/nb_recall_answer_lora_stage_c/checkpoint-{args.step}',
    'adapter_sha256': source['adapter_sha256'],
    'overall': summarize(records),
    'high_level_categories': {key: summarize(value) for key, value in sorted(by_high.items())},
    'low_level_categories': {key: summarize(value) for key, value in sorted(by_low.items())},
    'offline_video_rows_without_simulator_instruction': source['video_rows_without_simulator_instruction'],
    'official_runner_skipped_questions': source['official_runner_skipped_questions'],
    'history_video_sources': history_video_sources,
    'alignment': {
        'first_frame_mae_max': max(item['video_first_frame_mae'] for item in alignments),
        'last_frame_mae_max': max(item['video_last_frame_mae'] for item in alignments),
        'all_episodes_verified': all(item['video_last_frame_mae'] <= 18 and
                                     all(item.get(f'video_frame_{index}_mae', 0) <= 18
                                         for index in (10, 47, 90)) for item in alignments),
    },
    'results': records,
}
(base / 'hlsr_official_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({key: report[key] for key in ('metric', 'overall', 'high_level_categories',
                                               'alignment')}, ensure_ascii=False), flush=True)
