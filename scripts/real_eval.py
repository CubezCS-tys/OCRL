"""First test on a real scan: read each block of a parsed document and compare with its reference text.

Block positions come from the reference file (our own layout stage does not split columns yet),
so this measures reading, not page layout. The reference is another system's output, not ground truth.
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ocrl.blobs.layout import analyze
from ocrl.blobs.viz import overlay
from ocrl.pipeline import OCR, _line_text
from ocrl.text import CHARSET, strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="data/real/0013/pages")
ap.add_argument("--ref", default="data/real/parsed/model/v2_long6_from3/0013-002-001-005.master.json")
ap.add_argument("--ckpt", default="out/recognizer_v3.pt")
ap.add_argument("--out", default="out/real/0013")
ap.add_argument("--only-page", type=int, default=0)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--repair", action="store_true")
ap.add_argument("--own-layout", action="store_true", help="find columns and lines ourselves; the reference is used only to score")
args = ap.parse_args()
torch.set_num_threads(12)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)


from rapidfuzz.distance import Levenshtein

edit = Levenshtein.distance


OK = set(CHARSET + " ")
norm = lambda t: re.sub(r"\s+", " ", strip_marks(t)).strip()
inset = lambda t: re.sub(r"\s+", " ", "".join(c for c in norm(t) if c in OK)).strip()

ref = json.load(open(args.ref, encoding="utf-8"))
pages = {}
ocr = OCR(args.ckpt, repair=args.repair)
rows, notes, stats = [], Counter(), Counter()
def load_page(n):
    if n not in pages:
        img = cv2.imread(f"{args.pages}/p-{n:02d}.png", cv2.IMREAD_GRAYSCALE)
        pg = ref["pages"][n - 1]
        pages[n] = (img, img.shape[1] / pg["width"], img.shape[0] / pg["height"])
    return pages[n]


own = {}  # page -> [(region index, line index, cluster, centre x, centre y)] in our reading order


def read_own(n):
    if n not in own:
        img, _, _ = load_page(n)
        doc = ocr.read_page(img)
        items = []
        for ri, (r, res) in enumerate(doc.regions):
            notes.update(res.notes); stats["regions"] += 1
            for li, ln in enumerate(res.lines):
                for c in ln.clusters:
                    x0, y0, x1, y1 = res.layout.boxes[c.body]
                    items.append((ri, li, c, (x0 + x1) / 2 + r.x0 - 16, (y0 + y1) / 2 + r.y0 - 16))
                    stats["clusters"] += 1; stats["touching"] += c.touching; stats["flagged"] += bool(c.flags)
        own[n] = items
    return own[n]


for blk in ref["blocks"]:
    if args.only_page and not any(p["page"] == args.only_page for p in blk["parts"]):
        continue
    got = []
    for part in blk["parts"]:
        n = part["page"]
        img, sx, sy = load_page(n)
        for k, (x0, y0, x1, y1) in enumerate(part["regions"]):
            if args.own_layout:
                box = (x0 * sx - args.pad, y0 * sy - args.pad, x1 * sx + args.pad, y1 * sy + args.pad)
                lines = defaultdict(list)
                for ri, li, c, cx, cy in read_own(n):
                    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                        lines[(ri, li)].append(c)
                got.extend(_line_text(cl) for cl in lines.values())
                continue
            a, b = max(int(x0 * sx) - args.pad, 0), max(int(y0 * sy) - args.pad, 0)
            crop = img[b : int(y1 * sy) + args.pad, a : int(x1 * sx) + args.pad]
            if crop.size == 0 or crop.min() > 128:
                continue
            crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            lay = analyze(crop)
            res = ocr.read(crop, lay)
            got.append(" ".join(ln.text for ln in res.lines))
            notes.update(res.notes); stats["regions"] += 1
            for ln in res.lines:
                for c in ln.clusters:
                    stats["clusters"] += 1; stats["touching"] += c.touching; stats["flagged"] += bool(c.flags)
    ours, want = norm(" ".join(got)), norm(blk["text"])
    rows.append({"id": blk["id"], "type": blk["type"], "page": blk["parts"][0]["page"], "ref": want, "ours": ours,
                 "edits": edit(ours, want), "chars": len(want),
                 "edits_inset": edit(inset(ours), inset(want)), "chars_inset": len(inset(want))})

by = defaultdict(Counter)
for r in rows:
    for key in (r["type"], "ALL"):
        by[key].update(blocks=1, edits=r["edits"], chars=r["chars"], edits_inset=r["edits_inset"], chars_inset=r["chars_inset"])
print(f"{'block type':14s} blocks   chars  char error   (ignoring characters we cannot output)")
for key, c in sorted(by.items(), key=lambda kv: -kv[1]["chars"]):
    print(f"{key:14s} {c['blocks']:6d} {c['chars']:7d}  {100 * c['edits'] / max(c['chars'], 1):8.1f}%   {100 * c['edits_inset'] / max(c['chars_inset'], 1):8.1f}%")
print("regions", stats["regions"], "| clusters", stats["clusters"], "| touching", stats["touching"], "| flagged", stats["flagged"], "| page notes", dict(notes))
outside = Counter(c for r in rows for c in r["ref"] if c not in OK)
print("reference characters we cannot output:", sum(outside.values()), "of", sum(len(r["ref"]) for r in rows), "|", " ".join(f"{c}×{n}" for c, n in outside.most_common(25)))
json.dump(rows, open(out / ("blocks.json" if not args.only_page else f"blocks_p{args.only_page}.json"), "w"), ensure_ascii=False, indent=1)
if args.only_page:
    for r in rows[:6]:
        print(f"\n[{r['type']}] ref : {r['ref'][:150]}\n{' ' * (len(r['type']) + 3)}ours: {r['ours'][:150]}")
