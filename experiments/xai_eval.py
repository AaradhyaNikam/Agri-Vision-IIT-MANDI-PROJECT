"""Quantitative evaluation of Grad-CAM and LIME explanations on the held-out test set.

Grad-CAM: leaf-energy ratio and pointing game (leaf masks from the PlantVillage
'segmented' release), plus deletion/insertion AUC against a random-order baseline.
LIME: stability across sampling seeds, surrogate fidelity (R^2) and the confidence
drop when the top-5 superpixels are hidden versus 5 random superpixels.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from common import MEAN, ROOT, SPLIT_CSV, STD, RESULTS, build

SEG_ROOT = r"C:/Users/AARADHYA/Downloads/plantvillage/plantvillage dataset/segmented"
dev = "cuda"
MEAN_T = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
STD_T = torch.tensor(STD, device=dev).view(1, 3, 1, 1)


def load_model(tag):
    m = json.load(open(os.path.join(RESULTS, "runs", tag, "metrics.json")))["args"]
    model = build(m["model"], heads=m["heads"], pos=m["pos"], ffn=m["ffn"], unfreeze=m["unfreeze"])
    model.load_state_dict(torch.load(os.path.join(RESULTS, "runs", tag, "model.pt"), map_location="cpu"))
    return model.to(dev).eval()


def norm(x01):
    return (x01 - MEAN_T) / STD_T


def gradcam(model, x01, mode="gradcam", return_raw=False):
    """x01: (B,3,224,224) in [0,1]. Returns maps in [0,1] (B,224,224) and predicted labels.
    mode="gradcam": channel weights = spatially averaged gradients (Selvaraju et al.).
    mode="hirescam": element-wise gradient x activation summed over channels, which keeps the
    spatial structure of the gradient (useful when the head is not a pooled linear layer)."""
    with torch.no_grad():
        fmap = model.features(norm(x01))
    fmap = fmap.detach().float().requires_grad_(True)
    logits = model.head(fmap)
    pred = logits.argmax(1)
    score = logits.gather(1, pred[:, None]).sum()
    g, = torch.autograd.grad(score, fmap)
    if mode == "gradcam":
        cam = F.relu((g.mean((2, 3), keepdim=True) * fmap).sum(1, keepdim=True))
    else:
        cam = F.relu((g * fmap).sum(1, keepdim=True))
    degenerate = (cam.flatten(1).sum(1) == 0).detach()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True)
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
    if return_raw:
        return cam.detach(), pred.detach(), degenerate
    return cam.detach(), pred.detach()


def prob_of(model, x01, cls):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p = torch.softmax(model(norm(x01)).float(), 1)
    return p.gather(1, cls[:, None])[:, 0]


def del_ins(model, x01, sal, cls, steps=20):
    """Deletion: progressively replace the most salient pixels with the dataset mean.
    Insertion: progressively reveal them on a heavily blurred copy. Returns per-image AUCs."""
    B = x01.shape[0]
    order = sal.view(B, -1).argsort(1, descending=True)
    n = order.shape[1]
    base_del = MEAN_T.expand(B, 3, 224, 224)
    blurred = F.avg_pool2d(F.pad(x01, (15, 15, 15, 15), mode="reflect"), 31, stride=1)
    dels, inss = [], []
    for k in range(steps + 1):
        m = torch.zeros(B, n, device=dev)
        if k:
            m.scatter_(1, order[:, : int(n * k / steps)], 1.0)
        m = m.view(B, 1, 224, 224)
        dels.append(prob_of(model, x01 * (1 - m) + base_del * m, cls))
        inss.append(prob_of(model, blurred * (1 - m) + x01 * m, cls))
    d = torch.stack(dels, 1); i = torch.stack(inss, 1)
    auc = lambda c: ((c[:, 1:] + c[:, :-1]) / 2).mean(1)
    return auc(d).cpu().numpy(), auc(i).cpu().numpy()


def load_img(path, seg=False):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def sample_test(per_class=10, seed=0):
    df = pd.read_csv(SPLIT_CSV)
    te = df[df.split == "test"].copy()
    te["seg"] = te["path"].map(lambda p: os.path.join(SEG_ROOT, os.path.splitext(p)[0] + "_final_masked.jpg"))
    te = te[te["seg"].map(os.path.exists)]
    return te.groupby("label", group_keys=False).apply(lambda g: g.sample(min(per_class, len(g)), random_state=seed)).reset_index(drop=True)


def main(tags):
    s = sample_test()
    imgs = np.stack([load_img(os.path.join(ROOT, p)) for p in s["path"]])
    masks = np.stack([load_img(p).max(2) > 25 for p in s["seg"]])
    labels = torch.tensor(s["label"].to_numpy(), device=dev)
    out = {"n_images": len(s)}
    rng = torch.Generator(device=dev); rng.manual_seed(0)
    for tag in tags:
        model = load_model(tag)
        summ = {}
        for mode in ("gradcam", "hirescam"):
            res = {"leaf_energy": [], "pointing": [], "leaf_area": [], "del_cam": [], "ins_cam": [], "del_rand": [], "ins_rand": [], "correct": [], "degenerate": []}
            for b in range(0, len(s), 38):
                x = torch.from_numpy(imgs[b:b + 38]).to(dev).permute(0, 3, 1, 2).float() / 255
                mk = torch.from_numpy(masks[b:b + 38]).to(dev).float()
                cam, pred, deg = gradcam(model, x, mode, return_raw=True)
                res["correct"] += (pred == labels[b:b + 38]).cpu().tolist()
                res["degenerate"] += deg.float().cpu().tolist()
                en = (cam * mk).sum((1, 2)) / cam.sum((1, 2)).clamp_min(1e-8)
                res["leaf_energy"] += [float("nan") if dg else float(e) for e, dg in zip(en.cpu(), deg.cpu())]
                peak = cam.view(len(x), -1).argmax(1)
                res["pointing"] += [float("nan") if dg else float(v) for v, dg in zip(mk.view(len(x), -1).gather(1, peak[:, None])[:, 0].cpu(), deg.cpu())]
                res["leaf_area"] += mk.mean((1, 2)).cpu().tolist()
                d, i = del_ins(model, x, cam, pred)
                res["del_cam"] += d.tolist(); res["ins_cam"] += i.tolist()
                if mode == "gradcam":
                    rnd = torch.rand(cam.shape, device=dev, generator=rng)
                    d, i = del_ins(model, x, rnd, pred)
                    res["del_rand"] += d.tolist(); res["ins_rand"] += i.tolist()
            for k, v in res.items():
                if v:
                    summ[f"{mode}_{k}"] = float(np.nanmean(v))
            summ[f"{mode}_leaf_energy_std"] = float(np.nanstd(res["leaf_energy"]))
            # paired comparison of deletion AUC against random order (Wilcoxon signed-rank)
            if mode == "gradcam":
                rand_del, rand_ins = res["del_rand"], res["ins_rand"]
            from scipy.stats import wilcoxon
            summ[f"{mode}_del_vs_rand_p"] = float(wilcoxon(res["del_cam"], rand_del).pvalue)
            summ[f"{mode}_ins_vs_rand_p"] = float(wilcoxon(res["ins_cam"], rand_ins).pvalue)
        summ["del_rand"] = float(np.mean(rand_del)); summ["ins_rand"] = float(np.mean(rand_ins))

        # ---------------- LIME (smaller subset: 1 image per class)
        lime_idx = s.groupby("label").head(1).index.to_numpy()
        seg_fn = SegmentationAlgorithm("quickshift", kernel_size=4, max_dist=200, ratio=0.2, random_seed=0)

        def clf(batch):
            xb = torch.from_numpy(batch).to(dev).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return torch.softmax(model(norm(xb)).float(), 1).cpu().numpy()

        jacc, r2, drop_top, drop_rand, lime_leaf = [], [], [], [], []
        for j in lime_idx:
            img = imgs[j]
            tops, segs = [], None
            for seed in (1, 2):
                exp = lime_image.LimeImageExplainer(random_state=seed)
                e = exp.explain_instance(img, clf, top_labels=1, hide_color=0, num_samples=1000, segmentation_fn=seg_fn, batch_size=100)
                lab = e.top_labels[0]
                top = [f for f, w in sorted(e.local_exp[lab], key=lambda t: -t[1])[:5]]
                tops.append(set(top)); segs = e.segments
                if seed == 1:
                    r2.append(e.score)
                    p0 = clf(img[None])[0, lab]
                    hid = img.copy(); hid[np.isin(segs, top)] = 0
                    drop_top.append(p0 - clf(hid[None])[0, lab])
                    rs = np.random.default_rng(j).choice(np.unique(segs), size=min(5, len(np.unique(segs))), replace=False)
                    hid = img.copy(); hid[np.isin(segs, rs)] = 0
                    drop_rand.append(p0 - clf(hid[None])[0, lab])
                    area = np.isin(segs, top)
                    lime_leaf.append(float((area & masks[j]).sum() / max(1, area.sum())))
            jacc.append(len(tops[0] & tops[1]) / len(tops[0] | tops[1]))
        summ.update({"lime_n": int(len(lime_idx)), "lime_jaccard_top5": float(np.mean(jacc)), "lime_r2": float(np.mean(r2)),
                     "lime_drop_top5": float(np.mean(drop_top)), "lime_drop_rand5": float(np.mean(drop_rand)),
                     "lime_leaf_fraction": float(np.mean(lime_leaf))})
        out[tag] = summ
        print(tag, json.dumps(summ), flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(RESULTS, "xai_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["agrivision_s42", "mnv2_frozen_s42"])
