"""Settle what the checkpoint's shapes do not say about the owner's line reader, by reading known lines.

Tries each plausible setting on the blind test's control lines (where Azure and reader M agreed and people
confirmed the text) and reports how wrong each is. The right setting reads them almost perfectly.
"""
import itertools, json
from pathlib import Path

import cv2
import torch
from rapidfuzz.distance import Levenshtein

from ocrl.recog.linereader import LineReader
from ocrl.recog.realink import fold
from ocrl.text import strip_marks

torch.set_num_threads(2); cv2.setNumThreads(2)
X = Path("/home/yassine/data/inkscript_labelled/experiments/103_writer_as_judge/out")
CK = "/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
norm = lambda t: " ".join(strip_marks(" ".join(fold(w) for w in (t or "").split())).split())
A = {r["id"]: r for r in rd(X / "b/settle/A/readings.jsonl")}
ids = [r["id"] for r in rd(X / "blind/disputes.jsonl") if r["control"]][:40]
imgs = [cv2.imread(str(X / "blind/crops" / f"{i}.png"), 0) for i in ids]
truth = [norm(A[i]["printed"]) for i in ids]
rows = []
for wpool, invert, blank_first, late in itertools.product([(2, 2, 1, 1), (2, 1, 1, 1), (2, 2, 2, 1), (1, 2, 2, 1), (2, 1, 2, 1)], [True, False], [True, False], [True, False]):
    try:
        r = LineReader(CK, wpool, invert, blank_first, late, device="cpu")
        got = [norm(t) for t in r.read(imgs)]
    except Exception as e:
        continue
    for rev in (False, True):
        g = [t[::-1] if rev else t for t in got]
        e = sum(Levenshtein.distance(a, b) for a, b in zip(g, truth)); n = sum(map(len, truth))
        rows.append((100 * e / n, wpool, invert, blank_first, late, rev, g[0][:40]))
rows.sort(key=lambda r: r[0])
for r in rows[:6]:
    print(f"{r[0]:6.2f}% wrong | width pooling {r[1]} | ink as 1 {r[2]} | blank first {r[3]} | relu after add {r[4]} | reversed {r[5]} | {r[6]}")
print("truth:", truth[0][:40])
