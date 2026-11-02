"""Qualitative XAI figures: correct predictions and failure cases (original | Grad-CAM | LIME)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from analyze import short
from common import RESULTS, ROOT, SPLIT_CSV, class_names
from xai_eval import dev, gradcam, load_img, load_model, norm

FIG = os.path.join(RESULTS, "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 7, "savefig.dpi": 300})
SHOW = ["Tomato___Early_blight", "Apple___Apple_scab", "Grape___Black_rot", "Corn_(maize)___Common_rust_", "Potato___Late_blight", "Pepper,_bell___Bacterial_spot"]


def panel(model, items, names, path):
    seg_fn = SegmentationAlgorithm("quickshift", kernel_size=4, max_dist=200, ratio=0.2, random_seed=0)

    def clf(b):
        x = torch.from_numpy(b).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return torch.softmax(model(norm(x)), 1).cpu().numpy()

    fig, axes = plt.subplots(len(items), 3, figsize=(3.5, 1.2 * len(items)))
    for r, (img, y) in enumerate(items):
        x = torch.from_numpy(img[None]).to(dev).permute(0, 3, 1, 2).float() / 255
        cam, pred = gradcam(model, x)
        conf = clf(img[None])[0]
        p = int(pred[0])
        e = lime_image.LimeImageExplainer(random_state=1).explain_instance(img, clf, labels=(p,), top_labels=None, hide_color=0,
                                                                            num_samples=1000, segmentation_fn=seg_fn, batch_size=100)
        tmp, mask = e.get_image_and_mask(p, positive_only=True, num_features=5, hide_rest=False)
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(img); axes[r, 1].imshow(cam[0].cpu().numpy(), cmap="jet", alpha=0.45)
        axes[r, 2].imshow(mark_boundaries(img / 255.0, mask, color=(1, 1, 0)))
        ok = "✓" if p == y else "✗"
        axes[r, 0].set_ylabel(f"True: {short(names[y])}\nPred: {short(names[p])} ({conf[p]*100:.0f}%) {ok}", fontsize=5.2, rotation=0, ha="right", va="center")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(["Input", "Grad-CAM", "LIME (top-5)"]):
        axes[0, c].set_title(t, fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    names = class_names()
    df = pd.read_csv(SPLIT_CSV); te = df[df.split == "test"].reset_index(drop=True)
    z = np.load(os.path.join(RESULTS, "runs", "agrivision_s42", "test_preds.npz"))
    pred = z["probs"].astype(np.float32).argmax(1); y = z["y"]
    model = load_model("agrivision_s42")
    good = []
    for c in SHOW:
        li = names.index(c)
        idx = np.where((y == li) & (pred == li))[0][0]
        good.append((load_img(os.path.join(ROOT, te.loc[idx, "path"])), li))
    panel(model, good, names, os.path.join(FIG, "xai_examples.png"))
    wrong = np.where(pred != y)[0]
    conf = z["probs"].astype(np.float32).max(1)
    pick = wrong[np.argsort(-conf[wrong])][:4]  # most confident mistakes
    bad = [(load_img(os.path.join(ROOT, te.loc[i, "path"])), int(y[i])) for i in pick]
    panel(model, bad, names, os.path.join(FIG, "xai_failures.png"))
    print("done")


if __name__ == "__main__":
    main()
