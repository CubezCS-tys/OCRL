"""Keep the real-ink labels the current reader can vouch for.

A label cut from Azure's words is wrong when the cluster and the word slipped out of step; then the
reader's reading and the label have nothing in common. A reader mistake is different: one letter off.
So clusters where reader and label differ by at most one character are kept (the off-by-one cases are
the valuable ones: the label corrects the reader), and the rest are dropped as misaligned.

    python scripts/filter_realink.py --split train
"""
import argparse, glob
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ocrl.recog.decode import greedy
from ocrl.recog.model import load
from ocrl.recog.realink import load_shard, save_shard
from ocrl.text import strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train"); ap.add_argument("--ckpt", default="out/recognizer_v3.pt"); ap.add_argument("--max-edits", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(8)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, vocab, _ = load(a.ckpt, dev); model.eval()
out = Path(f"data/realink/{a.split}_f"); out.mkdir(parents=True, exist_ok=True)
c = Counter()
for f in sorted(glob.glob(f"data/realink/{a.split}/*.pt")):
    d = torch.load(f, map_location="cpu")
    pairs = load_shard(f)
    order = np.argsort([p[0].shape[2] for p in pairs]); preds = [None] * len(pairs)
    with torch.no_grad():
        for i in range(0, len(order), 512):
            idx = order[i : i + 512]; W = max(pairs[j][0].shape[2] for j in idx)
            x = torch.zeros(len(idx), 2, 48, W)
            for k, j in enumerate(idx):
                x[k, :, :, : pairs[j][0].shape[2]] = torch.from_numpy(pairs[j][0].copy()).float() / 255
            logp = model(x.to(dev)).cpu().numpy()
            for k, j in enumerate(idx):
                preds[j] = greedy(logp[: pairs[j][0].shape[2] // 2, k], vocab)
    keep = []
    for j, ((crop, label), pred) in enumerate(zip(pairs, preds)):
        e = Levenshtein.distance(strip_marks(pred), strip_marks(label))
        c["all"] += 1; c["exact"] += e == 0; c["one_off"] += e == 1
        if label and e <= a.max_edits:
            keep.append(j)
    save_shard(out / Path(f).name, [pairs[j][0] for j in keep], [pairs[j][1] for j in keep], [d["prov"][j] for j in keep])
    print(Path(f).name, len(pairs), "->", len(keep), flush=True)
print(f"{a.split}: {c['all']:,} clusters | reader agrees {100 * c['exact'] / c['all']:.1f}% | one letter off {100 * c['one_off'] / c['all']:.1f}% | kept {100 * (c['exact'] + c['one_off']) / c['all']:.1f}%")
