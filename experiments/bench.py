"""Re-measure efficiency for all trained models under identical, idle-GPU conditions."""
import json
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from common import RESULTS
from xai_eval import load_model


def timeit(fn, n, sync):
    for _ in range(10):
        fn()
    sync(); t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t) / n * 1000


def main(tags):
    out = {}
    torch.set_num_threads(8)
    for tag in tags:
        if not os.path.exists(os.path.join(RESULTS, "runs", tag, "model.pt")):
            continue
        m = load_model(tag).eval()
        params = sum(p.numel() for p in m.parameters()); trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        x1 = torch.randn(1, 3, 224, 224, device="cuda"); x64 = torch.randn(64, 3, 224, 224, device="cuda")
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                m(x1)
            gpu1 = timeit(lambda: m(x1), 200, torch.cuda.synchronize)
            with torch.autocast("cuda", dtype=torch.float16):
                gpu64 = timeit(lambda: m(x64), 20, torch.cuda.synchronize)
            mc = m.to("cpu"); xc = torch.randn(1, 3, 224, 224)
            cpu1 = timeit(lambda: mc(xc), 50, lambda: None)
        out[tag] = {"params_M": params / 1e6, "trainable_M": trainable / 1e6, "gflops": fc.get_total_flops() / 1e9, "size_mb": params * 4 / 1e6,
                    "gpu_ms_b1": gpu1, "gpu_imgs_per_s_b64_fp16": 64000 / gpu64, "cpu_ms_b1": cpu1}
        print(tag, json.dumps({k: round(v, 2) for k, v in out[tag].items()}), flush=True)
        del m, mc; torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(RESULTS, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or [d for d in sorted(os.listdir(os.path.join(RESULTS, "runs"))) if d.endswith("_s42")])
