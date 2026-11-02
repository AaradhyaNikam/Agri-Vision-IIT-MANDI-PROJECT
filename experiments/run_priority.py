"""Run priority experiments sequentially: robustness, ablations, multi-seed.
   Skips any run whose metrics.json already exists.
"""
import os, subprocess, sys, time
from common import RESULTS

PY = sys.executable
FROZEN = ["--epochs", "12", "--lr", "1e-3"]

# Priority queue: robustness first (reads existing checkpoints), 
# then ablations, then multi-seed
ABLATIONS = [
    ("abl_noattn_s42",  ["--model", "agri_noattn", "--seed", "42", *FROZEN]),
    ("abl_heads1_s42",  ["--model", "agrivision", "--heads", "1", "--seed", "42", *FROZEN]),
    ("abl_heads2_s42",  ["--model", "agrivision", "--heads", "2", "--seed", "42", *FROZEN]),
    ("abl_heads4_s42",  ["--model", "agrivision", "--heads", "4", "--seed", "42", *FROZEN]),
    ("abl_pos_s42",     ["--model", "agrivision", "--pos", "--seed", "42", *FROZEN]),
    ("abl_ffn_s42",     ["--model", "agrivision", "--ffn", "--seed", "42", *FROZEN]),
    ("abl_wce_s42",     ["--model", "agrivision", "--wce", "--seed", "42", *FROZEN]),
    ("abl_unfreeze_s42",["--model", "agrivision", "--unfreeze", "--seed", "42",
                         "--epochs", "10", "--lr", "1e-3"]),
]
MULTI_SEED = [
    ("agrivision_s1",   ["--model", "agrivision", "--seed", "1",  *FROZEN]),
    ("mnv2_frozen_s1",  ["--model", "mnv2_frozen", "--seed", "1",  *FROZEN]),
    ("agrivision_s7",   ["--model", "agrivision", "--seed", "7",  *FROZEN]),
    ("mnv2_frozen_s7",  ["--model", "mnv2_frozen", "--seed", "7",  *FROZEN]),
]

HERE = os.path.dirname(os.path.abspath(__file__))
logdir = os.path.join(RESULTS, "logs")
os.makedirs(logdir, exist_ok=True)
status_file = os.path.join(RESULTS, "priority_queue_status.txt")

def run_train(tag, args):
    if os.path.exists(os.path.join(RESULTS, "runs", tag, "metrics.json")):
        print(f"SKIP {tag} (already complete)", flush=True)
        return 0
    with open(status_file, "a") as s:
        s.write(f"{time.strftime('%H:%M:%S')} START {tag}\n")
    print(f"START {tag}", flush=True)
    log_path = os.path.join(logdir, tag + ".log")
    with open(log_path, "w") as log:
        rc = subprocess.call(
            [PY, "-W", "ignore", "train.py", "--tag", tag, *args],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=HERE
        )
    with open(status_file, "a") as s:
        s.write(f"{time.strftime('%H:%M:%S')} END {tag} rc={rc}\n")
    print(f"END {tag} rc={rc}", flush=True)
    return rc

if __name__ == "__main__":
    # Step 1: Robustness (uses existing checkpoints, no training)
    rob_out = os.path.join(RESULTS, "robustness.json")
    if not os.path.exists(rob_out):
        print("START robustness_eval", flush=True)
        with open(status_file, "a") as s:
            s.write(f"{time.strftime('%H:%M:%S')} START robustness_eval\n")
        rob_log = os.path.join(logdir, "robustness_eval.log")
        with open(rob_log, "w") as log:
            rc = subprocess.call(
                [PY, "-W", "ignore", "robust_eval.py",
                 "agrivision_s42", "mnv2_frozen_s42",
                 "cnn_scratch_s42", "mnv2_ft_s42", "resnet18_ft_s42"],
                stdout=log, stderr=subprocess.STDOUT,
                cwd=HERE
            )
        with open(status_file, "a") as s:
            s.write(f"{time.strftime('%H:%M:%S')} END robustness_eval rc={rc}\n")
        print(f"END robustness_eval rc={rc}", flush=True)
    else:
        print("SKIP robustness_eval (already exists)", flush=True)

    # Step 2: Ablations
    print("\n=== ABLATION RUNS ===", flush=True)
    for tag, args in ABLATIONS:
        run_train(tag, args)

    # Step 3: Multi-seed
    print("\n=== MULTI-SEED RUNS ===", flush=True)
    for tag, args in MULTI_SEED:
        run_train(tag, args)

    with open(status_file, "a") as s:
        s.write(f"{time.strftime('%H:%M:%S')} ALL PRIORITY DONE\n")
    print("\nALL PRIORITY EXPERIMENTS DONE", flush=True)
