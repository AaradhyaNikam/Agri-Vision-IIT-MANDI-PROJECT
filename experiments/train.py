"""Train one model on the fixed split and evaluate it on the held-out test set.

Example:
    python train.py --model agrivision --seed 42 --epochs 15 --lr 1e-3 --tag agrivision_s42
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.flop_counter import FlopCounterMode

from common import NUM_CLASSES, RESULTS, BatchStream, build, gpu_prep


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(model, loader, dev):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            probs.append(torch.softmax(model(gpu_prep(x.to(dev, non_blocking=True), False)).float(), 1).cpu())
            labels.append(y)
    p = torch.cat(probs).numpy(); y = torch.cat(labels).numpy()
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"acc": accuracy_score(y, pred), "macro_p": pr, "macro_r": rc, "macro_f1": f1}, p, y


def efficiency(model, dev):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(1, 3, 224, 224)
    m_cpu = model.to("cpu")
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m_cpu(x)
        flops = fc.get_total_flops()
        for _ in range(5):
            m_cpu(x)
        t = time.perf_counter(); n = 30
        for _ in range(n):
            m_cpu(x)
        cpu_ms = (time.perf_counter() - t) / n * 1000
    model.to(dev)
    xg = x.to(dev)
    with torch.no_grad():
        for _ in range(20):
            model(xg)
        torch.cuda.synchronize(); t = time.perf_counter(); n = 100
        for _ in range(n):
            model(xg)
        torch.cuda.synchronize(); gpu_ms = (time.perf_counter() - t) / n * 1000
    return {"params": params, "trainable": trainable, "gflops": flops / 1e9, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
            "size_mb": params * 4 / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos", action="store_true")
    ap.add_argument("--ffn", action="store_true")
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--wce", action="store_true", help="class-weighted cross-entropy")
    a = ap.parse_args()

    out = os.path.join(RESULTS, "runs", a.tag)
    os.makedirs(out, exist_ok=True)
    set_seed(a.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    tl = BatchStream("train", a.bs, shuffle=True, seed=a.seed, drop_last=True)
    vl = BatchStream("val", 128); tel = BatchStream("test", 128)

    model = build(a.model, heads=a.heads, pos=a.pos, ffn=a.ffn, unfreeze=a.unfreeze).to(dev)
    if a.unfreeze and hasattr(model, "backbone"):
        groups = [{"params": model.backbone.parameters(), "lr": a.lr * 0.1},
                  {"params": [p for n_, p in model.named_parameters() if not n_.startswith("backbone")], "lr": a.lr}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    steps = a.epochs * len(tl); warm = len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    weight = None
    if a.wce:
        counts = np.bincount(tl.y, minlength=NUM_CLASSES)
        weight = torch.tensor(len(tl.y) / (NUM_CLASSES * counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler()

    hist, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); te0 = time.time(); tot = cor = 0; ls = 0.0
        for x, y in tl:
            x, y = gpu_prep(x.to(dev, non_blocking=True), True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x); loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ls += loss.item() * len(y); tot += len(y); cor += (logits.argmax(1) == y).sum().item()
        vm, _, _ = evaluate(model, vl, dev)
        rec = {"epoch": ep + 1, "train_loss": ls / tot, "train_acc": cor / tot, "val_acc": vm["acc"], "val_macro_f1": vm["macro_f1"], "sec": time.time() - te0}
        hist.append(rec); print(json.dumps(rec), flush=True)
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_min = (time.time() - t0) / 60

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out, "model.pt"))
    tm, probs, y = evaluate(model, tel, dev)
    pred = probs.argmax(1)
    pc = precision_recall_fscore_support(y, pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    np.savez_compressed(os.path.join(out, "test_preds.npz"), probs=probs.astype(np.float16), y=y)
    np.save(os.path.join(out, "confusion.npy"), confusion_matrix(y, pred, labels=list(range(NUM_CLASSES))))
    eff = efficiency(model, dev)
    res = {"args": vars(a), "test": {**tm, "weighted_f1": wf1}, "best_val_macro_f1": best, "train_minutes": train_min,
           "per_class": {"precision": pc[0].tolist(), "recall": pc[1].tolist(), "f1": pc[2].tolist(), "support": pc[3].tolist()},
           "efficiency": eff, "history": hist}
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("TEST", json.dumps({**tm, "weighted_f1": wf1}), json.dumps(eff), flush=True)


if __name__ == "__main__":
    main()
