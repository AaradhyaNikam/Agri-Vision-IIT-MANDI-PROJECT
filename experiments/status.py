"""Print queue progress: finished runs with test scores, and the epoch count of the active run."""
import glob
import json
import os
import time

from common import RESULTS

print(time.strftime("%H:%M:%S"))
for log in sorted(glob.glob(os.path.join(RESULTS, "logs", "*.log")), key=os.path.getmtime):
    tag = os.path.basename(log)[:-4]
    m = os.path.join(RESULTS, "runs", tag, "metrics.json")
    if os.path.exists(m):
        t = json.load(open(m))["test"]
        print(f"{tag:22s} done  acc={t['acc']:.4f}  macroF1={t['macro_f1']:.4f}")
    else:
        lines = [l for l in open(log) if l.startswith("{")]
        last = json.loads(lines[-1]) if lines else {}
        print(f"{tag:22s} running epoch {len(lines)}  val_f1={last.get('val_macro_f1', 0):.4f}  sec/ep={last.get('sec', 0):.0f}")
        err = [l for l in open(log) if "Error" in l]
        if err:
            print("   ERROR:", err[-1].strip()[:200])
