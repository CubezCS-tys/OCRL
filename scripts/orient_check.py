"""Check ocrl.blobs.orient on pages Azure found turned (|angle| > 45) and on upright pages of the same documents.

    uv run python scripts/orient_check.py
"""
import glob, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2

from lost_causes import page_image
from ocrl.blobs.orient import orientation

cv2.setNumThreads(4)
jobs = []
for f in sorted(glob.glob("/home/yassine/data/mandumah_s3/*/*.json") + glob.glob("/home/yassine/data/unseen_docs/*/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    d = d.get("analyzeResult", d)
    doc = Path(f).parent.name
    turned = [(i + 1, round(p.get("angle", 0) or 0)) for i, p in enumerate(d["pages"]) if abs(p.get("angle", 0) or 0) > 45 and len(p.get("lines", [])) >= 15]
    if not turned:
        continue
    up = [i + 1 for i, p in enumerate(d["pages"]) if abs(p.get("angle", 0) or 0) < 5 and len(p.get("lines", [])) >= 5]
    jobs += [(str(Path(f).parent.parent), doc, p, a) for p, a in turned]
    jobs += [(str(Path(f).parent.parent), doc, p, 0) for p in random.Random(0).sample(up, min(2, len(up)))]
tally = Counter()
for folder, doc, p, ang in jobs:
    g = page_image(Path(folder), doc, p)
    deg, conf = orientation(g)
    tally[(ang, deg)] += 1
    print(f"{doc} p{p}: azure {ang:4d}  ours {deg:3d} ({conf:.2f})", flush=True)
print(sorted(tally.items()))
