"""Montage of units in a given miss category, to look at what the ink actually does."""
import sys
import cv2, numpy as np
from ocrl.blobs.layout import analyze
from ocrl.blobs.viz import overlay
from ocrl.eval.m0 import blob_units
from ocrl.render.page import compose_page
from ocrl.render.shaper import Font
from ocrl.rules.signature import satellite_range
from ocrl.text import load_corpus

name, px, want = sys.argv[1], int(sys.argv[2]), sys.argv[3]
corpus = load_corpus("data/corpus/sentences.txt")
page = compose_page(Font(f"data/fonts/{name}.ttf", px), " ".join(corpus[1000:1060]), width=px * 36, max_lines=30)
lay = analyze(page.image)
bu = blob_units(page, lay)
ub = {}
for b, us in bu.items():
    for u in us: ub.setdefault(u, set()).add(b)
touch = {u for us in bu.values() if len(us) > 1 for u in us}
ov = overlay(page.image, lay, scale=1)
tiles = []
for u, blobs in sorted(ub.items()):
    lo, hi = satellite_range(page.units[u].text)
    n = len(blobs) - 1
    kind = "touch" if u in touch else "more" if n > hi else "fewer" if n < lo else "ok"
    if kind != want: continue
    bx = lay.boxes[list(blobs)]
    x0, y0, x1, y1 = bx[:, 0].min() - 30, bx[:, 1].min() - 14, bx[:, 2].max() + 30, bx[:, 3].max() + 14
    x0, y0 = max(x0, 0), max(y0, 0)
    a = cv2.cvtColor(page.image[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    t = np.vstack([a, ov[y0:y1, x0:x1]])
    t = cv2.resize(t, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    tiles.append(t)
    if len(tiles) == 8: break
H = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
cv2.imwrite(sys.argv[4], np.hstack(tiles)); print(len(tiles))
