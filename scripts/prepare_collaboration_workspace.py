"""Prepare a new experiment from external archive assets and this repository's code."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['evaluate', 'retrain'], default='evaluate')
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.out.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(archive):
        parser.error('Output must be a new directory outside the repository and archive.')
    required = ['创建复现目录.py', '归档清单.json', 'data', 'models', 'envs']
    missing = [name for name in required if not (archive / name).exists()]
    if missing:
        parser.error('Full archive is missing: ' + ', '.join(missing))
    subprocess.run([sys.executable, str(archive / '创建复现目录.py'),
                    '--mode', args.mode, '--out', str(output)], check=True)
    for name in ['scripts', 'nb_recall', 'configs', 'vendor']:
        shutil.copytree(repo / name, output / name, dirs_exist_ok=True)
    marker = output / 'REPRODUCTION_WORKSPACE.json'
    info = json.loads(marker.read_text())
    revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    info.update(code_repository=str(repo), code_commit=revision.stdout.strip() or None,
                code_overlaid_from_repository=True)
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print('Workspace prepared. No training or evaluation was launched:', output)


if __name__ == '__main__':
    main()
