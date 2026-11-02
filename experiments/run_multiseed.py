"""
Run the four multi-seed experiments only, sequentially.
Skips any run that already has a complete metrics.json.
On any non-zero return code, stops immediately and reports the error.
Does NOT run robustness, ablations, EfficientNet, or ViT.
"""
import os
import subprocess
import sys
import time

# ── paths ──────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))   # experiments/
RESULTS = os.path.join(HERE, "results")
logdir  = os.path.join(RESULTS, "logs")
os.makedirs(logdir, exist_ok=True)
status_file = os.path.join(RESULTS, "priority_queue_status.txt")

PY     = sys.executable
FROZEN = ["--epochs", "12", "--lr", "1e-3"]

# ── the only four runs this script will attempt ────────────────────────────
MULTI_SEED = [
    ("agrivision_s1",  ["--model", "agrivision",  "--seed", "1",  *FROZEN]),
    ("mnv2_frozen_s1", ["--model", "mnv2_frozen",  "--seed", "1",  *FROZEN]),
    ("agrivision_s7",  ["--model", "agrivision",  "--seed", "7",  *FROZEN]),
    ("mnv2_frozen_s7", ["--model", "mnv2_frozen",  "--seed", "7",  *FROZEN]),
]


def log_status(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with open(status_file, "a") as f:
        f.write(line)
    print(msg, flush=True)


def run_train(tag, args):
    metrics_path = os.path.join(RESULTS, "runs", tag, "metrics.json")
    if os.path.exists(metrics_path):
        log_status(f"SKIP {tag} (metrics.json already exists)")
        return 0

    log_path = os.path.join(logdir, tag + ".log")
    log_status(f"START {tag}")
    with open(log_path, "w") as log:
        rc = subprocess.call(
            [PY, "-W", "ignore", "train.py", "--tag", tag, *args],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=HERE,
        )
    log_status(f"END {tag} rc={rc}")
    return rc


if __name__ == "__main__":
    log_status("=== MULTI-SEED RUNS START ===")
    for tag, args in MULTI_SEED:
        rc = run_train(tag, args)
        if rc != 0:
            log_status(f"ABORT — {tag} returned rc={rc}. Stopping queue.")
            print(f"\nERROR: {tag} failed with rc={rc}. "
                  f"Check log: {os.path.join(logdir, tag + '.log')}", flush=True)
            sys.exit(rc)

    log_status("=== MULTI-SEED RUNS COMPLETE ===")
    print("\nAll four multi-seed runs completed successfully.", flush=True)
