"""Aggregate all runs into paper tables, statistics and figures."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from common import RESULTS, class_names

RUNS = os.path.join(RESULTS, "runs")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
RED, GOLD, CHAR, GREY = "#B22B2F", "#CDA23B", "#424346", "#9A9A9A"

MAIN = [("cnn_scratch_s42", "CNN (from scratch)"), ("mnv2_frozen_s42", "MobileNetV2 (frozen)"), ("mnv2_ft_s42", "MobileNetV2 (fine-tuned)"),
        ("resnet18_ft_s42", "ResNet-18 (fine-tuned)"), ("effb0_ft_s42", "EfficientNet-B0 (fine-tuned)"), ("vitb16_ft_s42", "ViT-B/16 (fine-tuned)"),
        ("agrivision_s42", "Agri-Vision (proposed)")]
ABL = [("abl_noattn_s42", "Frozen MobileNetV2, token average (no attention)"), ("abl_heads1_s42", "Agri-Vision, 1 head"), ("abl_heads2_s42", "Agri-Vision, 2 heads"),
       ("abl_heads4_s42", "Agri-Vision, 4 heads"), ("agrivision_s42", "Agri-Vision, 8 heads (default)"), ("abl_pos_s42", "+ learnable positional embedding"),
       ("abl_ffn_s42", "+ feed-forward sub-layer (full encoder block)"), ("abl_wce_s42", "+ class-weighted loss"), ("abl_unfreeze_s42", "+ fine-tuned backbone")]


def load(tag):
    p = os.path.join(RUNS, tag, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def preds(tag):
    z = np.load(os.path.join(RUNS, tag, "test_preds.npz"))
    return z["probs"].astype(np.float32).argmax(1), z["y"]


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); accs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[i], p[i])); f1s.append(f1_score(y[i], p[i], average="macro"))
    return np.percentile(accs, [2.5, 97.5]).tolist(), np.percentile(f1s, [2.5, 97.5]).tolist()


def short(c):
    crop, dis = c.split("___")
    crop = crop.replace("_(including_sour)", "").replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("_", " ")
    dis = (dis.replace("Cercospora_leaf_spot Gray_leaf_spot", "Cercospora/Gray leaf spot").replace("Spider_mites Two-spotted_spider_mite", "Spider mites")
           .replace("Haunglongbing_(Citrus_greening)", "Huanglongbing").replace("Tomato_Yellow_Leaf_Curl_Virus", "Yellow Leaf Curl Virus")
           .replace("Leaf_blight_(Isariopsis_Leaf_Spot)", "Leaf blight").replace("Esca_(Black_Measles)", "Esca").replace("Tomato_mosaic_virus", "Mosaic virus")
           .replace("_", " ").strip())
    return f"{crop} – {dis}"


def main():
    names = class_names(); sn = [short(c) for c in names]
    out = {}
    # ---------------- main table
    rows = []
    bpath = os.path.join(RESULTS, "bench.json")
    bench = json.load(open(bpath)) if os.path.exists(bpath) else {}
    ref_p, y = preds("agrivision_s42")
    for tag, label in MAIN:
        m = load(tag)
        if not m:
            continue
        p, _ = preds(tag)
        ca, cf = boot_ci(y, p)
        t, e = m["test"], m["efficiency"]
        row = {"tag": tag, "model": label, "acc": t["acc"], "acc_ci": ca, "macro_p": t["macro_p"], "macro_r": t["macro_r"], "macro_f1": t["macro_f1"],
               "macro_f1_ci": cf, "weighted_f1": t["weighted_f1"], "params_M": e["params"] / 1e6, "trainable_M": e["trainable"] / 1e6,
               "gflops": e["gflops"], "size_mb": e["size_mb"], "cpu_ms": e["cpu_ms"], "gpu_ms": e["gpu_ms"], "train_min": m["train_minutes"],
               "epochs": m["args"]["epochs"]}
        if tag in bench:
            bb = bench[tag]
            row.update({"params_M": bb["params_M"], "trainable_M": bb["trainable_M"], "gflops": bb["gflops"], "size_mb": bb["size_mb"],
                        "cpu_ms": bb["cpu_ms_b1"], "gpu_ms": bb["gpu_ms_b1"], "gpu_throughput": bb["gpu_imgs_per_s_b64_fp16"]})
        if tag != "agrivision_s42":
            b = ((ref_p == y) & (p != y)).sum(); c = ((ref_p != y) & (p == y)).sum()
            r = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
            row.update({"mcnemar_b_agri_only": int(b), "mcnemar_c_other_only": int(c), "mcnemar_p": float(r.pvalue)})
        rows.append(row)
    out["main"] = rows
    # ---------------- ablation
    out["ablation"] = [{"tag": t, "config": l, **{k: load(t)["test"][k] for k in ("acc", "macro_f1")},
                        "trainable_M": load(t)["efficiency"]["trainable"] / 1e6, "gflops": load(t)["efficiency"]["gflops"]}
                       for t, l in ABL if load(t)]
    # ---------------- seeds
    seeds = {}
    for base in ("agrivision", "mnv2_frozen"):
        ms = [load(f"{base}_s{s}") for s in (42, 1, 7)]
        ms = [m for m in ms if m]
        if ms:
            seeds[base] = {"n": len(ms), **{k: [float(np.mean([m["test"][k] for m in ms])), float(np.std([m["test"][k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0]
                                         for k in ("acc", "macro_f1", "macro_p", "macro_r")}}
    out["seeds"] = seeds
    # ---------------- per class + confusions
    m = load("agrivision_s42"); pc = m["per_class"]
    out["per_class"] = [{"class": names[i], "short": sn[i], "precision": pc["precision"][i], "recall": pc["recall"][i], "f1": pc["f1"][i], "support": pc["support"][i]}
                        for i in range(len(names))]
    cm = np.load(os.path.join(RUNS, "agrivision_s42", "confusion.npy"))
    off = [(cm[i, j], i, j) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    out["top_confusions"] = [{"true": sn[i], "pred": sn[j], "count": int(c), "rate": float(c / cm[i].sum())} for c, i, j in sorted(off, reverse=True)[:8]]
    split = json.load(open(os.path.join(RESULTS, "split_info.json")))
    out["split"] = {k: v for k, v in split.items() if k != "classes"}
    for f in ("xai_metrics.json", "robustness.json"):
        if os.path.exists(os.path.join(RESULTS, f)):
            out[f[:-5]] = json.load(open(os.path.join(RESULTS, f)))
    json.dump(out, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1)

    # ================= figures
    counts = pd.read_csv(os.path.join(RESULTS, "class_counts.csv"), index_col=0).sum(1)
    counts.index = [short(c) for c in counts.index]; counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(3.5, 5.0))
    ax.barh(counts.index, counts.values, color=[RED if v < 500 else CHAR for v in counts.values], height=0.72)
    for i, v in enumerate(counts.values):
        ax.text(v + 60, i, f"{v:,}", va="center", fontsize=4.6)
    ax.set_xlabel("Number of images", fontsize=7); ax.tick_params(axis="y", labelsize=4.8, length=1.5); ax.tick_params(axis="x", labelsize=6)
    ax.set_xlim(0, counts.max() * 1.18); ax.margins(y=0.005)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "class_distribution.png")); plt.close(fig)

    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(np.sqrt(cmn), cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(sn))); ax.set_yticks(range(len(sn)))
    ax.set_xticklabels(range(1, len(sn) + 1), rotation=90, fontsize=4); ax.set_yticklabels(range(1, len(sn) + 1), fontsize=4)
    ax.tick_params(length=1)
    ax.set_xlabel("Predicted class index", fontsize=7); ax.set_ylabel("True class index", fontsize=7)
    cb = fig.colorbar(im, fraction=0.045, pad=0.02); cb.set_ticks([0, 0.1 ** 0.5, 0.5 ** 0.5, 1]); cb.set_ticklabels(["0", "0.1", "0.5", "1"]); cb.ax.tick_params(labelsize=5); fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrix.png")); plt.close(fig)

    base = load("mnv2_frozen_s42")
    if base:
        f_a = np.array(pc["f1"]); f_b = np.array(base["per_class"]["f1"]); o = np.argsort(f_a)
        fig, ax = plt.subplots(figsize=(3.5, 5.0))
        yy = np.arange(len(o))
        ax.barh(yy + 0.2, f_a[o], height=0.4, color=RED, label="Agri-Vision")
        ax.barh(yy - 0.2, f_b[o], height=0.4, color=GREY, label="MobileNetV2 (frozen)")
        ax.set_yticks(yy); ax.set_yticklabels([sn[i] for i in o], fontsize=4.8); ax.set_xlabel("Test F1-score", fontsize=7); ax.tick_params(axis="x", labelsize=6); ax.margins(y=0.005)
        ax.set_xlim(min(f_a.min(), f_b.min()) - 0.03, 1.0); ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "per_class_f1.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for (tag, label), col in zip(MAIN, [GREY, GOLD, "#7A6A9A", "#4A8A6A", "#3A6EA5", "#888844", RED]):
        mm = load(tag)
        if mm:
            h = mm["history"]; ax.plot([r["epoch"] for r in h], [r["val_macro_f1"] for r in h], color=col, lw=1.4 if "agri" in tag else 1.0, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1"); ax.legend(fontsize=5.5, frameon=False); ax.set_ylim(0.6, 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curves.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for r in rows:
        col = RED if "agri" in r["tag"] else CHAR
        ax.scatter(r["params_M"], r["macro_f1"] * 100, s=20 + 25 * r["gflops"], color=col, alpha=0.85)
        ax.annotate(r["model"].split(" (")[0], (r["params_M"], r["macro_f1"] * 100), fontsize=6, xytext=(4, -3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Parameters (millions, log scale)"); ax.set_ylabel("Test macro-F1 (%)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f1_vs_params.png")); plt.close(fig)

    if "robustness" in out:
        rb = out["robustness"]; cors = [c for c in next(iter(rb.values())) if c != "clean"]
        sel = [t for t in ("cnn_scratch_s42", "mnv2_frozen_s42", "mnv2_ft_s42", "vitb16_ft_s42", "agrivision_s42") if t in rb]
        labels = dict(MAIN)
        fig, ax = plt.subplots(figsize=(3.5, 2.4)); w = 0.8 / len(sel)
        for k, (t, col) in enumerate(zip(sel, [GREY, GOLD, "#7A6A9A", "#888844", RED])):
            ax.bar(np.arange(len(cors)) + k * w, [rb[t][c]["acc"] * 100 for c in cors], width=w, color=col, label=labels[t])
        ax.set_xticks(np.arange(len(cors)) + w * (len(sel) - 1) / 2)
        ax.set_xticklabels([c.replace("_", " ").replace("low contrast", "low contr.") for c in cors], fontsize=5.5, rotation=25, ha="right")
        ax.set_ylabel("Test accuracy (%)", fontsize=7); ax.tick_params(axis="y", labelsize=6); ax.legend(fontsize=5, frameon=False, ncol=2, loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness.png")); plt.close(fig)
    print(json.dumps({"main": [(r["model"], round(r["acc"], 4), round(r["macro_f1"], 4)) for r in rows], "seeds": seeds}, indent=1))


if __name__ == "__main__":
    main()
