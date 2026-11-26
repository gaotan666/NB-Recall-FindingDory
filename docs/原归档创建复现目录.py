"""Create a separate workspace; never run training in the frozen archive."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['evaluate', 'retrain'], required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
out = Path(a.out).expanduser().resolve()
if out.exists() or out.is_relative_to(root):
    raise RuntimeError('Output must be a new directory outside the archive')
out.mkdir(parents=True)

def copy(rel):
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-a', '--reflink=auto', '--', str(root / rel), str(dst)], check=True)

for rel in ['scripts', 'nb_recall', 'configs', 'vendor', 'envs', 'environment']:
    copy(rel)
(out / 'models').mkdir()
(out / 'models/Qwen2.5-VL-3B-Instruct').symlink_to(root / 'models/Qwen2.5-VL-3B-Instruct', target_is_directory=True)
(out / 'data').mkdir()
for src in (root / 'data').iterdir():
    if src.is_symlink():
        (out / 'data' / src.name).symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif a.mode == 'evaluate' or src.name == 'stage_a_append_only':
        copy(str(src.relative_to(root)))
if a.mode == 'evaluate':
    for rel in ['models/nb_recall_unified_lora_stage_a', 'models/nb_recall_answer_lora_stage_c',
                'models/nb_recall_dual_lora_trust_heads']:
        copy(rel)
for name in ['reports', 'logs']:
    (out / name).mkdir()
(out / 'REPRODUCTION_WORKSPACE.json').write_text(json.dumps({
    'mode': a.mode, 'archive': str(root), 'shared_raw_data_read_only': True,
    'historical_result': '2498/5866; new runs may differ'}, ensure_ascii=False, indent=2))
print(out)
