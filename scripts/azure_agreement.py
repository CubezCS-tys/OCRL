"""How closely does a fully local reading agree with Azure's, over many documents it has not seen?

For each Azure line, the clusters of ours that sit inside it are read out and compared with Azure's text.
Azure is not ground truth; this is agreement, measured the same way for every reader so they can be compared.

    python scripts/azure_agreement.py --ckpt out/recognizer_v3.pt [--lines out/linefinder.pt] --docs 50
"""
import argparse, json, subprocess, tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ocrl.pipeline import OCR, _line_text
from ocrl.recog.realink import fold
from ocrl.text import CHARSET, strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--lines", default=""); ap.add_argument("--line-reader", default=""); ap.add_argument("--split", default="heldout")
ap.add_argument("--docs", type=int, default=50); ap.add_argument("--max-pages", type=int, default=6); ap.add_argument("--out", default="")
ap.add_argument("--ids", default="", help="a JSON list of document ids (default: data/realink/SPLIT_ids.json)")
ap.add_argument("--samples", default="/home/yassine/data/mandumah_s3", help="folder holding ID/ID.pdf and ID/ID.json")
a = ap.parse_args()
torch.set_num_threads(8); cv2.setNumThreads(4)
S3 = Path(a.samples)
ocr = OCR(a.ckpt, linefinder=a.lines or None, line_reader=a.line_reader or None)
OK = set(CHARSET + " ")
norm = lambda t: " ".join(strip_marks(" ".join(fold(w) for w in t.split())).split())
ids = json.load(open(a.ids or f"data/realink/{a.split}_ids.json"))[: a.docs]
tot, per_doc = Counter(), {}
for stem in ids:
    pdf, js = S3 / stem / f"{stem}.pdf", S3 / stem / f"{stem}.json"
    if not (pdf.exists() and js.exists()):
        continue
    az = json.loads(js.read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
    c = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        n = min(len(az["pages"]), a.max_pages)
        subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-l", str(n), str(pdf), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=600)
        for pi, f in enumerate(sorted(Path(tmp).glob("p-*.png"))[:n]):
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); pg = az["pages"][pi]
            sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
            items, whole = [], []  # clusters with their centres; or, with a line reader, whole lines with theirs
            for ri, (r, res) in enumerate(ocr.read_page(gray).regions):
                for li, ln in enumerate(res.lines):
                    if a.line_reader and ln.clusters:
                        bx = res.layout.boxes[[b for cl in ln.clusters for b in cl.blobs]]
                        whole.append((ln.text, (bx[:, 0].min() + bx[:, 2].max()) / 2 + r.x0 - 16, (bx[:, 1].min() + bx[:, 3].max()) / 2 + r.y0 - 16))
                    for cl in ln.clusters:
                        x0, y0, x1, y1 = res.layout.boxes[cl.body]
                        items.append((ri, li, cl, (x0 + x1) / 2 + r.x0 - 16, (y0 + y1) / 2 + r.y0 - 16))
            for ln in pg.get("lines", []):
                want = norm(ln["content"])
                if not want:
                    continue
                q = ln["polygon"]; bx = (min(q[0::2]) * sx - 6, min(q[1::2]) * sy - 6, max(q[0::2]) * sx + 6, max(q[1::2]) * sy + 6)
                groups = defaultdict(list)
                for ri, li, cl, cx, cy in items:
                    if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                        groups[(ri, li)].append(cl)
                got = norm(" ".join(_line_text(g) for g in groups.values()))
                if a.line_reader:  # our lines whose centre sits in this Azure line
                    got = norm(" ".join(t for t, cx, cy in whole if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]))
                arabic = all(ch in OK and not (ch.isascii() and ch.isalpha()) for ch in want)
                e = Levenshtein.distance(got, want)
                c["lines"] += 1; c["chars"] += len(want); c["edits"] += e; c["empty"] += not got
                if arabic:
                    c["ar_chars"] += len(want); c["ar_edits"] += e
    if c["chars"]:
        per_doc[stem] = 100 * c["edits"] / c["chars"]; tot += c
        print(f"{stem}: lines {c['lines']:4d}  disagreement {per_doc[stem]:5.1f}%  lines with nothing read {c['empty']}", flush=True)
v = sorted(per_doc.values())
print(f"\nALL {len(per_doc)} documents, {tot['lines']:,} Azure lines, {tot['chars']:,} characters")
print(f"   character disagreement with Azure: {100 * tot['edits'] / tot['chars']:.2f}%  (Arabic-only lines: {100 * tot['ar_edits'] / max(tot['ar_chars'], 1):.2f}%)")
print(f"   per document: median {v[len(v) // 2]:.1f}%, best quarter under {v[len(v) // 4]:.1f}%, worst quarter over {v[3 * len(v) // 4]:.1f}%")
print(f"   Azure lines where we read nothing: {100 * tot['empty'] / tot['lines']:.1f}%")
if a.out:
    json.dump({"per_doc": per_doc, "total": dict(tot)}, open(a.out, "w"))
