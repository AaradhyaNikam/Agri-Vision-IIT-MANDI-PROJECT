"""Create a reproducible stratified 70/15/15 split of PlantVillage (colour).

Exact duplicate files (same MD5) are grouped so that every copy lands in the same
partition, which prevents train/test leakage through identical images.
"""
import hashlib
import json
import os
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = r"C:/Users/AARADHYA/Downloads/plantvillage/plantvillage dataset/color"
OUT = os.path.join(os.path.dirname(__file__), "results")
SEED = 42


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    classes = sorted(os.listdir(ROOT))
    rows = []
    for ci, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(ROOT, c))):
            p = os.path.join(ROOT, c, fn)
            rows.append({"path": os.path.join(c, fn), "label": ci, "cls": c, "md5": md5(p)})
    df = pd.DataFrame(rows)

    # one representative per duplicate group; the group inherits that row's split
    groups = df.groupby("md5")
    dup_groups = sum(1 for _, g in groups if len(g) > 1)
    dup_files = sum(len(g) for _, g in groups if len(g) > 1)
    cross_class = sum(1 for _, g in groups if g["label"].nunique() > 1)
    rep = groups.first().reset_index()

    tr, tmp = train_test_split(rep, test_size=0.30, stratify=rep["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
    split_of = {**{m: "train" for m in tr["md5"]}, **{m: "val" for m in va["md5"]}, **{m: "test" for m in te["md5"]}}
    df["split"] = df["md5"].map(split_of)
    df.to_csv(os.path.join(OUT, "split.csv"), index=False)

    counts = df.groupby(["cls", "split"]).size().unstack(fill_value=0)
    counts.to_csv(os.path.join(OUT, "class_counts.csv"))
    totals = df["cls"].value_counts()
    info = {
        "n_images": int(len(df)), "n_classes": len(classes), "classes": classes,
        "n_unique_md5": int(df["md5"].nunique()), "duplicate_groups": int(dup_groups),
        "files_in_duplicate_groups": int(dup_files), "duplicate_groups_spanning_classes": int(cross_class),
        "split_sizes": {k: int(v) for k, v in df["split"].value_counts().items()},
        "max_class": [totals.idxmax(), int(totals.max())], "min_class": [totals.idxmin(), int(totals.min())],
        "imbalance_ratio": round(float(totals.max() / totals.min()), 2), "seed": SEED,
    }
    json.dump(info, open(os.path.join(OUT, "split_info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=1))


if __name__ == "__main__":
    main()
