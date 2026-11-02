"""Test-set accuracy and macro-F1 under synthetic corruptions (field-condition proxies)."""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score

from common import MEAN, RESULTS, STD, BatchStream
from xai_eval import load_model

dev = "cuda"
MEAN_T = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
STD_T = torch.tensor(STD, device=dev).view(1, 3, 1, 1)


def rotate(x, deg):
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0]], device=dev).expand(len(x), 2, 3)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)


def occlude(x, size=64, g=None):
    x = x.clone()
    for i in range(len(x)):
        r, c = torch.randint(0, 224 - size, (2,), generator=g).tolist()
        x[i, :, r:r + size, c:c + size] = 0
    return x


CORRUPTIONS = {
    "clean": lambda x, g: x,
    "blur": lambda x, g: TF.gaussian_blur(x, 9, 2.0),
    "noise": lambda x, g: (x + 0.08 * torch.randn(x.shape, device=dev, generator=g)).clamp(0, 1),
    "dark": lambda x, g: (x * 0.6).clamp(0, 1),
    "bright": lambda x, g: (x * 1.4).clamp(0, 1),
    "low_contrast": lambda x, g: ((x - x.mean((1, 2, 3), keepdim=True)) * 0.5 + x.mean((1, 2, 3), keepdim=True)).clamp(0, 1),
    "rotate30": lambda x, g: rotate(x, 30),
    "occlusion": lambda x, g: occlude(x, 64, torch.Generator().manual_seed(0)),
}


def main(tags):
    out = {}
    stream = BatchStream("test", 128)
    for tag in tags:
        model = load_model(tag)
        out[tag] = {}
        for name, fn in CORRUPTIONS.items():
            g = torch.Generator(device=dev); g.manual_seed(0)
            preds, ys = [], []
            for xb, yb in stream:
                x = xb.to(dev).permute(0, 3, 1, 2).float() / 255
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    preds.append(model((fn(x, g) - MEAN_T) / STD_T).argmax(1).cpu())
                ys.append(yb)
            p = torch.cat(preds).numpy(); y = torch.cat(ys).numpy()
            out[tag][name] = {"acc": float((p == y).mean()), "macro_f1": float(f1_score(y, p, average="macro"))}
        print(tag, json.dumps({k: round(v["acc"], 4) for k, v in out[tag].items()}), flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(RESULTS, "robustness.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["agrivision_s42", "mnv2_frozen_s42", "cnn_scratch_s42", "mnv2_ft_s42", "resnet18_ft_s42", "effb0_ft_s42", "vitb16_ft_s42"])
