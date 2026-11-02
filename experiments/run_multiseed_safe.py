"""
run_multiseed_safe.py — hardened replacement for run_multiseed.py

Differences from the original:
  1. SKIP guard checks all four required artifacts (metrics.json, test_preds.npz,
     model.pt, confusion.npy) AND verifies that the logged epoch count equals the
     configured epoch count.  A run whose metrics.json exists but is incomplete
     (e.g. a truncated write) is not silently skipped — it is flagged and aborted.
  2. Before opening a log in write mode, any pre-existing partial log is renamed to
     <tag>.log.interrupted_<timestamp> so it is never silently overwritten.
  3. Directories for s7 runs are created by train.py (makedirs exist_ok=True) only
     when training starts; this runner does not pre-create them.
  4. Sequential execution via subprocess.call (blocking); stops on first non-zero rc.
  5. After every successful run, the runner recomputes acc / macro-F1 / weighted-F1
     from test_preds.npz and compares them to metrics.json.  A mismatch aborts the
     queue.

Does NOT run robustness, ablations, EfficientNet-B0, ViT, or any other experiment.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ── paths ────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))   # experiments/
RESULTS = os.path.join(HERE, "results")
LOGDIR  = os.path.join(RESULTS, "logs")
os.makedirs(LOGDIR, exist_ok=True)

STATUS_FILE = os.path.join(RESULTS, "priority_queue_status.txt")
PY          = sys.executable
FROZEN      = ["--epochs", "12", "--lr", "1e-3"]

# ── the four runs — order matters (sequential) ────────────────────────────────
MULTI_SEED = [
    ("agrivision_s1",  ["--model", "agrivision",  "--seed", "1",  *FROZEN]),
    ("mnv2_frozen_s1", ["--model", "mnv2_frozen", "--seed", "1",  *FROZEN]),
    ("agrivision_s7",  ["--model", "agrivision",  "--seed", "7",  *FROZEN]),
    ("mnv2_frozen_s7", ["--model", "mnv2_frozen", "--seed", "7",  *FROZEN]),
]

REQUIRED_ARTIFACTS = ["metrics.json", "test_preds.npz", "model.pt", "confusion.npy"]

# ── helpers ───────────────────────────────────────────────────────────────────

def log_status(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with open(STATUS_FILE, "a") as f:
        f.write(line)
    print(msg, flush=True)


def _all_artifacts_present(run_dir):
    """Return (ok: bool, missing: list[str])."""
    missing = [a for a in REQUIRED_ARTIFACTS
               if not os.path.isfile(os.path.join(run_dir, a))]
    return len(missing) == 0, missing


def _epochs_complete(metrics_path, expected_epochs):
    """Return True only if history length == expected_epochs."""
    try:
        m = json.load(open(metrics_path))
        logged = len(m.get("history", []))
        return logged == expected_epochs, logged
    except Exception as e:
        return False, f"parse error: {e}"


def check_skip(tag, args):
    """
    Condition 1: skip only if all four artifacts exist AND epochs are complete.
    Returns:
        "skip"   – run is verified complete; do not retrain
        "abort"  – metrics.json exists but run is detectably incomplete (safe abort)
        "run"    – run has not been completed; proceed with training
    """
    run_dir      = os.path.join(RESULTS, "runs", tag)
    metrics_path = os.path.join(run_dir, "metrics.json")

    if not os.path.isfile(metrics_path):
        return "run"   # no metrics at all – start fresh

    # metrics.json exists; check everything else before skipping
    all_ok, missing = _all_artifacts_present(run_dir)
    if not all_ok:
        log_status(
            f"WARN  {tag}: metrics.json exists but missing artifacts: {missing}. "
            f"Treating as incomplete — cannot safely retrain without review."
        )
        return "abort"

    # Check the expected epoch count from args
    epochs_flag = int(args[args.index("--epochs") + 1]) if "--epochs" in args else 15
    ok, logged = _epochs_complete(metrics_path, epochs_flag)
    if not ok:
        log_status(
            f"WARN  {tag}: metrics.json exists but only {logged}/{epochs_flag} epochs "
            f"logged. Treating as incomplete — manual review required."
        )
        return "abort"

    return "skip"


def archive_partial_log(tag):
    """
    Condition 2: if a prior log for this tag exists, rename it to
    <tag>.log.interrupted_<timestamp> before opening a new one in write mode.
    """
    log_path = os.path.join(LOGDIR, tag + ".log")
    if os.path.isfile(log_path):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(LOGDIR, f"{tag}.log.interrupted_{ts}")
        shutil.move(log_path, dest)
        log_status(f"ARCHIVE partial log → {os.path.basename(dest)}")


def verify_run(tag):
    """
    Condition 5: recompute acc / macro-F1 / weighted-F1 from test_preds.npz
    and compare with metrics.json.  Returns True on match, False on mismatch.
    """
    run_dir   = os.path.join(RESULTS, "runs", tag)
    preds_p   = os.path.join(run_dir, "test_preds.npz")
    metrics_p = os.path.join(run_dir, "metrics.json")

    d     = np.load(preds_p)
    probs = d["probs"].astype(np.float32)
    y     = d["y"]
    pred  = probs.argmax(1)

    acc           = accuracy_score(y, pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    _, _, wf1, _  = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)

    m = json.load(open(metrics_p))
    t = m["test"]

    eps = 1e-5
    match = (
        abs(acc  - t["acc"])         < eps and
        abs(mf1  - t["macro_f1"])    < eps and
        abs(wf1  - t["weighted_f1"]) < eps
    )

    log_status(
        f"VERIFY {tag}: n={len(y)}  "
        f"acc={acc:.6f}  macro_f1={mf1:.6f}  wf1={wf1:.6f}  "
        f"{'MATCH' if match else 'MISMATCH'}"
    )
    if not match:
        log_status(
            f"  metrics.json: acc={t['acc']:.6f}  "
            f"macro_f1={t['macro_f1']:.6f}  wf1={t['weighted_f1']:.6f}"
        )
    return match


def run_train(tag, args):
    """
    Orchestrates skip-check, log archival, subprocess launch, and post-run
    verification for a single run.

    Returns 0 on success (or verified skip), non-zero on any failure.
    """
    # ── Condition 1: skip / abort / run decision ──────────────────────────────
    decision = check_skip(tag, args)

    if decision == "skip":
        log_status(f"SKIP {tag} (all artifacts verified complete)")
        return 0

    if decision == "abort":
        log_status(
            f"ABORT {tag}: metrics.json present but run is detectably incomplete. "
            f"Manual review required — not retraining automatically."
        )
        return 1   # non-zero → outer loop stops

    # decision == "run"

    # ── Condition 2: archive any pre-existing partial log ─────────────────────
    archive_partial_log(tag)

    # ── Conditions 3 & 4: launch training (train.py handles makedirs) ─────────
    log_path = os.path.join(LOGDIR, tag + ".log")
    log_status(f"START {tag}")
    with open(log_path, "w") as log:            # safe: old log already archived
        rc = subprocess.call(
            [PY, "-W", "ignore", "train.py", "--tag", tag, *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=HERE,
        )
    log_status(f"END {tag} rc={rc}")

    if rc != 0:
        return rc   # caller will stop the queue

    # ── Condition 5: verify the run that just completed ───────────────────────
    if not verify_run(tag):
        log_status(f"ABORT {tag}: post-run verification MISMATCH — stopping queue.")
        return 2   # non-zero → outer loop stops

    return 0


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_status("=== MULTI-SEED SAFE RUNS START ===")

    for tag, args in MULTI_SEED:
        rc = run_train(tag, args)
        if rc != 0:
            log_status(f"ABORT — {tag} returned rc={rc}. Queue halted.")
            print(
                f"\nERROR: {tag} failed (rc={rc}). "
                f"Check log: {os.path.join(LOGDIR, tag + '.log')}",
                flush=True,
            )
            sys.exit(rc)

    log_status("=== MULTI-SEED SAFE RUNS COMPLETE ===")
    print("\nAll four multi-seed runs completed and verified successfully.", flush=True)
