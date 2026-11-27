"""Score a line reader on the held-out journals' lines, against Azure's text (Azure-confident lines only).

Works for the owner's experiment-89 checkpoint and for ours, so the two are measured on the same lines the same way.
Vowel marks are dropped and Persian letters folded on both sides.
    python scripts/eval_line_reader.py out/reader_v7.pt out/line_reader_ocrl.pt --n 20000
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ocrl.recog.linereader import LineReader
from ocrl.recog.realink import fold
from ocrl.text import strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+"); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--out", default="out/line_reader_eval.json")
a = ap.parse_args()
norm = lambda t: " ".join(strip_marks(" ".join(fold(w) for w in (t or "").split())).split())
items = []
for f in sorted(glob.glob("data/reader/lines_test/*.pt")):
    d = torch.load(f, map_location="cpu")
    strip, off = d["strip"].numpy(), d["offsets"].tolist()
    items += [(strip[:, s:e], lab, p) for s, e, lab, p in zip(off, off[1:], d["labels"], d["prov"])]
pick = np.random.default_rng(0).permutation(len(items))[: a.n]  # a spread over all the held-out journals, not the first few
items = [items[i] for i in pick]
truth = [norm(l) for _, l, _ in items]
arabic = [any("؀" <= c <= "ۿ" for c in t) for t in truth]
res = json.load(open(a.out)) if Path(a.out).exists() else {}
print(f"{len(items):,} lines from {len({p[0][:4] for _, _, p in items})} held-out journals, {sum(map(len, truth)):,} characters")
for ck in a.ckpts:
    r = LineReader(ck)
    got = [norm(t) for t in r.read([im for im, _, _ in items])]
    e = [Levenshtein.distance(g, t) for g, t in zip(got, truth)]
    cer = 100 * sum(e) / sum(map(len, truth)); exact = 100 * sum(x == 0 for x in e) / len(e)
    cer_ar = 100 * sum(x for x, ar in zip(e, arabic) if ar) / max(sum(len(t) for t, ar in zip(truth, arabic) if ar), 1)
    print(f"   {ck:32s} characters differing from Azure {cer:5.2f}%  (lines with Arabic: {cer_ar:5.2f}%)  lines identical {exact:5.1f}%", flush=True)
    res[ck] = {"cer": cer, "cer_arabic": cer_ar, "exact": exact, "lines": len(items)}
    json.dump(res, open(a.out, "w"), indent=1)
