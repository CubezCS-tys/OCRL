"""What do Azure's line boxes teach for the known faults? Counts over built pages, and overlays of chosen pages.

    python scripts/lf2_probe.py --data data --split val --s3 /home/yassine/data/reader_s3 --draw probe
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ocrl.blobs.linefinder import core_mask
import lf2_targets as T

DASH = re.compile(r"^[\s\-–—_\.…·]+$")
MARK = re.compile(r"^[\s\(\[]*([0-9٠-٩۰-۹]{1,3}|[a-zA-Zأ-ي])?[\s\)\]\.\-–:]*$|^[\s•●○■□▪◦\*\-–—]+$")
SPLIT = 1.2

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--split", default="val"); ap.add_argument("--s3"); ap.add_argument("--draw", default="probe")
ap.add_argument("--max", type=int, default=4000)
a = ap.parse_args()
meta = json.load(open(f"{a.data}/{a.split}_pages.json"))[: a.max]
c = Counter(); ex = {"dash": [], "mark": [], "side": [], "gap_table": [], "gap_text": [], "tall_dash": []}
cache = {}
for m in meta:
    name = m["page"]; doc, pn = name.rsplit("_p", 1); pi = int(pn) - 1
    if doc not in cache:
        cache.clear()
        az = json.loads((Path(a.s3) / doc / f"{doc}.json").read_text(encoding="utf-8")); cache[doc] = az.get("analyzeResult", az)
    L = cache[doc]["pages"][pi]["lines"]
    z = np.load(f"{a.data}/{a.split}/{name}.npz")
    boxes, ang = z["boxes"], z["angle"]
    if len(boxes) != len(L):
        c["mismatch"] += 1; continue
    h = boxes[:, 3] - boxes[:, 1]; med = np.median(h) if len(h) else 1
    segs = T.segments(z, SPLIT)
    tab = m["table"] >= 0.3
    c["pages"] += 1; c["lines"] += len(L); c["table_pages"] += tab; c["table_lines"] += len(L) * tab
    c["segments_added"] += len(segs) - len(boxes); c["segments_added_table"] += (len(segs) - len(boxes)) * tab
    for i, l in enumerate(L):
        t = l["content"].strip()
        if abs(abs(ang[i]) - 90) < 30:
            c["sideways"] += 1; ex["side"].append(name)
        if DASH.match(t):
            c["dash_only"] += 1; ex["dash"].append(name)
            if h[i] > 2.5 * med:
                c["dash_tall"] += 1; ex["tall_dash"].append(name)
        elif MARK.match(t) and len(t) <= 4:
            c["mark_only"] += 1; ex["mark"].append(name)
    # lines with a wide gap between words (Azure's one box spans what may be two cells or two columns)
    wl, wb = z["word_line"], z["words"]
    for i, b in enumerate(boxes):
        ws = wb[wl == i]
        if len(ws) < 2 or h[i] <= 0:
            continue
        ws = ws[np.argsort(ws[:, 0])]
        gaps = ws[1:, 0] - np.maximum.accumulate(ws[:-1, 2])
        if (gaps > SPLIT * h[i]).any():
            c["gap_lines"] += 1; c["gap_lines_table"] += tab
            ex["gap_table" if tab else "gap_text"].append(name)
print(dict(c))
for k in ("lines", "table_lines"):
    pass
print(f"share of lines with a word gap > {SPLIT} line heights: all {100 * c['gap_lines'] / c['lines']:.2f}%, table-heavy pages {100 * c['gap_lines_table'] / max(c['table_lines'], 1):.2f}%")
print(f"dash-only lines {c['dash_only']} (tall {c['dash_tall']}), mark-only lines {c['mark_only']}, sideways {c['sideways']}  of {c['lines']}")
out = Path(a.draw); out.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
for k, v in ex.items():
    v = sorted(set(v)); rng.shuffle(v)
    for name in v[:3]:
        img = cv2.imread(f"{a.data}/{a.split}/{name}.png", 0); z = np.load(f"{a.data}/{a.split}/{name}.npz")
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for M, col in ((core_mask(img.shape, z["boxes"]), (0, 200, 255)), (core_mask(img.shape, T.segments(z, SPLIT)), (0, 160, 0))):
            pass
        lay = rgb.copy()
        lay[core_mask(img.shape, z["boxes"]) > 0] = (0, 200, 255)          # current target: orange
        lay[core_mask(img.shape, T.segments(z, SPLIT)) == 0] = rgb[core_mask(img.shape, T.segments(z, SPLIT)) == 0]
        mm = (core_mask(img.shape, z["boxes"]) > 0) & (core_mask(img.shape, T.segments(z, SPLIT)) == 0)
        lay[mm] = (255, 0, 255)                                              # removed by the gap split: magenta
        rgb = cv2.addWeighted(lay, 0.45, rgb, 0.55, 0)
        for i, b in enumerate(z["boxes"]):
            cv2.rectangle(rgb, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (200, 0, 0) if i % 2 else (0, 0, 200), 1)
        cv2.imwrite(str(out / f"{k}_{name}.png"), rgb)
        print(k, name)
