"""Collect independent official oracle trajectories from the training split."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--stop", type=int, default=20)
parser.add_argument("--save-every", type=int, default=5)
args = parser.parse_args()
manifest = ROOT / f"data/sim_replay/train/batch_{args.start}_{args.stop}.jsonl"
manifest.parent.mkdir(parents=True, exist_ok=True)
for episode in range(args.start, args.stop + 1):
    summary = ROOT / f"data/sim_replay/train/ep_{episode}/summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("collection_complete"):
        continue
    log = ROOT / f"logs/replay_ep{episode}.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/replay_train_experience.py"),
             "--episode", str(episode), "--max-steps", "2500",
             "--save-every", str(args.save_every)],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
        )
    row = {"episode": episode, "exit_code": result.returncode,
           "summary": json.loads(summary.read_text()) if summary.exists() else None,
           "log": str(log.relative_to(ROOT))}
    with manifest.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
