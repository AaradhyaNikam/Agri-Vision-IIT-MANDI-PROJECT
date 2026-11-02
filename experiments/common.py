"""Shared data pipeline and model definitions for the Agri-Vision experiments (PyTorch)."""
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

ROOT = r"C:/Users/AARADHYA/Downloads/plantvillage/plantvillage dataset/color"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SPLIT_CSV = os.path.join(RESULTS, "split.csv")
NUM_CLASSES = 38
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TRAIN_TF = T.Compose([
    T.Resize(224),
    T.RandomAffine(degrees=20, scale=(0.8, 1.2)),  # rotation ±20°, zoom 0.2 (as in the original Keras pipeline)
    T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class PlantVillage(Dataset):
    def __init__(self, split, transform):
        df = pd.read_csv(SPLIT_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(ROOT, r["path"])).convert("RGB")
        return self.transform(img), int(r["label"])


def _decode(b):
    import io
    img = Image.open(io.BytesIO(b)).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class BatchStream:
    """Batches decoded on a thread pool from JPEG bytes held in RAM (about 0.85 GB for the
    whole dataset). PIL releases the GIL while decoding, so threads run in parallel and we
    avoid both slow worker processes on Windows and random disk reads."""

    def __init__(self, split, bs, shuffle=False, seed=0, drop_last=False, prefetch=6, threads=8):
        df = pd.read_csv(SPLIT_CSV)
        d = df[df["split"] == split].reset_index(drop=True)
        self.blobs = [open(os.path.join(ROOT, p), "rb").read() for p in d["path"]]
        self.y = d["label"].to_numpy(np.int64)
        self.bs, self.shuffle, self.drop_last, self.prefetch, self.threads = bs, shuffle, drop_last, prefetch, threads
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        order = self.rng.permutation(len(self.y)) if self.shuffle else np.arange(len(self.y))
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.drop_last and len(batches[-1]) < self.bs:
            batches = batches[:-1]
        q = queue.Queue(self.prefetch)
        pool = ThreadPoolExecutor(self.threads)

        def work():
            for b in batches:
                imgs = list(pool.map(_decode, [self.blobs[i] for i in b]))
                q.put((torch.from_numpy(np.stack(imgs)).pin_memory(), torch.from_numpy(self.y[b])))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                pool.shutdown(wait=False)
                return
            yield item


_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(STD).view(1, 3, 1, 1)


def gpu_prep(xb, train):
    """uint8 (B,H,W,3) on GPU -> normalised float (B,3,H,W); training adds random
    rotation (±20°), zoom (0.8–1.2) and horizontal flip, matching the original pipeline."""
    x = xb.permute(0, 3, 1, 2).float().div_(255)
    if train:
        B = x.shape[0]
        ang = (torch.rand(B, device=x.device) * 40 - 20) * math.pi / 180
        s = torch.rand(B, device=x.device) * 0.4 + 0.8
        f = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)
        c, sn = torch.cos(ang) / s, torch.sin(ang) / s
        theta = torch.stack([torch.stack([c * f, sn, torch.zeros_like(c)], 1),
                             torch.stack([-sn * f, c, torch.zeros_like(c)], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def class_names():
    df = pd.read_csv(SPLIT_CSV)
    return list(df.drop_duplicates("label").sort_values("label")["cls"])


# ---------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """'Standard CNN' baseline trained from scratch."""

    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))  # 224 -> 112
        self.features = nn.Sequential(stem, block(32, 64), block(64, 128), block(128, 256), block(256, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


class KerasStyleMHA(nn.Module):
    """Multi-head attention equivalent to keras.layers.MultiHeadAttention(num_heads=h, key_dim=k):
    per-head key/value size k, output projected back to the model dimension."""

    def __init__(self, dim=1280, heads=8, key_dim=256):
        super().__init__()
        self.h, self.k = heads, key_dim
        self.q = nn.Linear(dim, heads * key_dim)
        self.kp = nn.Linear(dim, heads * key_dim)
        self.v = nn.Linear(dim, heads * key_dim)
        self.o = nn.Linear(heads * key_dim, dim)
        self.last_attn = None

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.k).transpose(1, 2)
        k = self.kp(x).view(B, N, self.h, self.k).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.k).transpose(1, 2)
        # attention logits in fp32: with key_dim 256 they can overflow fp16 under autocast
        with torch.autocast("cuda", enabled=False):
            attn = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / self.k ** 0.5, dim=-1)
            out = (attn @ v.float()).transpose(1, 2).reshape(B, N, self.h * self.k)
        self.last_attn = attn.detach()
        out = out.to(x.dtype) if not torch.is_autocast_enabled() else out
        return self.o(out)


class AgriVision(nn.Module):
    """Frozen MobileNetV2 -> 7x7x1280 map -> 49 tokens -> multi-head self-attention
    (+ residual, LayerNorm) -> token average -> dropout -> softmax over 38 classes."""

    def __init__(self, heads=8, attention=True, pos=False, ffn=False, freeze=True, n=NUM_CLASSES):
        super().__init__()
        self.backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.attention = attention
        if attention:
            self.mha = KerasStyleMHA(1280, heads, 256)
            self.norm = nn.LayerNorm(1280)
        self.pos = nn.Parameter(torch.zeros(1, 49, 1280)) if pos else None
        if pos:
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.ffn = None
        if ffn:
            self.ffn = nn.Sequential(nn.Linear(1280, 2048), nn.GELU(), nn.Dropout(0.1), nn.Linear(2048, 1280))
            self.norm2 = nn.LayerNorm(1280)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, n)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # keep BatchNorm statistics frozen
        return self

    def features(self, x):
        if self.freeze:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    def head(self, fmap):
        t = fmap.flatten(2).transpose(1, 2)  # (B, 49, 1280)
        if self.pos is not None:
            t = t + self.pos
        if self.attention:
            t = self.norm(t + self.mha(t))
        if self.ffn is not None:
            t = self.norm2(t + self.ffn(t))
        return self.fc(self.drop(t.mean(1)))

    def forward(self, x):
        return self.head(self.features(x))


class MobileNetFrozen(nn.Module):
    """Original transfer-learning baseline: frozen MobileNetV2 + GAP + Dense(256) head."""

    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        self.backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1).features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc1 = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n))
        self.freeze = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x):
        with torch.no_grad():
            return self.backbone(x)

    def head(self, fmap):
        return self.fc1(fmap.mean((2, 3)))

    def forward(self, x):
        return self.head(self.features(x))


def build(name, **kw):
    if name == "cnn_scratch":
        return SmallCNN()
    if name == "mnv2_frozen":
        return MobileNetFrozen()
    if name == "agrivision":
        return AgriVision(heads=kw.get("heads", 8), pos=kw.get("pos", False), ffn=kw.get("ffn", False), freeze=not kw.get("unfreeze", False))
    if name == "agri_noattn":
        return AgriVision(attention=False)
    if name == "mnv2_ft":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        return m
    if name == "resnet18_ft":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, NUM_CLASSES)
        return m
    if name == "effb0_ft":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        return m
    if name == "vitb16_ft":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(768, NUM_CLASSES)
        return m
    raise ValueError(name)
