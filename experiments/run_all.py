"""Run the full experiment queue sequentially. Resumable: runs whose metrics.json exists are skipped."""
import os
import subprocess
import sys
import time

from common import RESULTS

PY = sys.executable
FROZEN = ["--epochs", "12", "--lr", "1e-3"]
FT = ["--epochs", "10", "--lr", "3e-4"]

QUEUE = [
    # main comparison (seed 42)
    ("agrivision_s42", ["--model", "agrivision", "--seed", "42", *FROZEN]),
    ("mnv2_frozen_s42", ["--model", "mnv2_frozen", "--seed", "42", *FROZEN]),
    ("cnn_scratch_s42", ["--model", "cnn_scratch", "--seed", "42", "--epochs", "15", "--lr", "1e-3"]),
    ("mnv2_ft_s42", ["--model", "mnv2_ft", "--seed", "42", *FT]),
    ("resnet18_ft_s42", ["--model", "resnet18_ft", "--seed", "42", *FT]),
    ("effb0_ft_s42", ["--model", "effb0_ft", "--seed", "42", *FT]),
    ("vitb16_ft_s42", ["--model", "vitb16_ft", "--seed", "42", "--epochs", "5", "--lr", "5e-5", "--bs", "32", "--wd", "0.05"]),
    # ablations (seed 42)
    ("abl_noattn_s42", ["--model", "agri_noattn", "--seed", "42", *FROZEN]),
    ("abl_heads1_s42", ["--model", "agrivision", "--heads", "1", "--seed", "42", *FROZEN]),
    ("abl_heads2_s42", ["--model", "agrivision", "--heads", "2", "--seed", "42", *FROZEN]),
    ("abl_heads4_s42", ["--model", "agrivision", "--heads", "4", "--seed", "42", *FROZEN]),
    ("abl_pos_s42", ["--model", "agrivision", "--pos", "--seed", "42", *FROZEN]),
    ("abl_ffn_s42", ["--model", "agrivision", "--ffn", "--seed", "42", *FROZEN]),
    ("abl_wce_s42", ["--model", "agrivision", "--wce", "--seed", "42", *FROZEN]),
    ("abl_unfreeze_s42", ["--model", "agrivision", "--unfreeze", "--seed", "42", "--epochs", "10", "--lr", "1e-3"]),
    # extra seeds for the two key models
    ("agrivision_s1", ["--model", "agrivision", "--seed", "1", *FROZEN]),
    ("mnv2_frozen_s1", ["--model", "mnv2_frozen", "--seed", "1", *FROZEN]),
    ("agrivision_s7", ["--model", "agrivision", "--seed", "7", *FROZEN]),
    ("mnv2_frozen_s7", ["--model", "mnv2_frozen", "--seed", "7", *FROZEN]),
]

if __name__ == "__main__":
    logdir = os.path.join(RESULTS, "logs")
    os.makedirs(logdir, exist_ok=True)
    status = os.path.join(RESULTS, "queue_status.txt")
    for tag, args in QUEUE:
        if os.path.exists(os.path.join(RESULTS, "runs", tag, "metrics.json")):
            continue
        with open(status, "a") as s:
            s.write(f"{time.strftime('%H:%M:%S')} START {tag}\n")
        with open(os.path.join(logdir, tag + ".log"), "w") as log:
            rc = subprocess.call([PY, "-W", "ignore", "train.py", "--tag", tag, *args], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
        with open(status, "a") as s:
            s.write(f"{time.strftime('%H:%M:%S')} END {tag} rc={rc}\n")
    with open(status, "a") as s:
        s.write(f"{time.strftime('%H:%M:%S')} ALL DONE\n")
