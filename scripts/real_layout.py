"""Does our whitespace layout put a real document's blocks in the right regions, in the right order?

The reference file's blocks are used as probes: each block's position is looked up in our regions.
"""
import argparse, json
from collections import Counter

import cv2

from ocrl.blobs.regions import find_regions

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="data/real/0013/pages")
ap.add_argument("--ref", default="data/real/parsed/model/v2_long6_from3/0013-002-001-005.master.json")
args = ap.parse_args()
ref = json.load(open(args.ref, encoding="utf-8"))
BODY = {"paragraph", "heading", "list_item", "abstract", "title", "author", "affiliation", "footnote", "reference", "table_caption"}
c, bad_pages = Counter(), []
for n, pg in enumerate(ref["pages"], 1):
    img = cv2.imread(f"{args.pages}/p-{n:02d}.png", cv2.IMREAD_GRAYSCALE)
    sx, sy = img.shape[1] / pg["width"], img.shape[0] / pg["height"]
    regions = find_regions(img)
    probes = []  # (reading order, region index, box)
    for b in ref["blocks"]:
        if b["type"] not in BODY or "order" not in b:
            continue
        for part in b["parts"]:
            if part["page"] != n:
                continue
            for x0, y0, x1, y1 in part["regions"]:
                box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hit = next((i for i, r in enumerate(regions) if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1), -1)
                probes.append((b["order"], hit, box, b["type"]))
    probes.sort(key=lambda p: p[0])
    c["pages"] += 1; c["blocks"] += len(probes); c["regions"] += len(regions)
    c["covered"] += sum(p[1] >= 0 for p in probes)
    seq = [p[1] for p in probes if p[1] >= 0]
    inversions = sum(a > b for a, b in zip(seq, seq[1:]))
    # Two blocks side by side (same height on the page, different columns) inside one region: columns were merged.
    merged = 0
    for i, p in enumerate(probes):
        for q in probes[i + 1 :]:
            if p[1] == q[1] >= 0:
                oy = min(p[2][3], q[2][3]) - max(p[2][1], q[2][1])
                ox = min(p[2][2], q[2][2]) - max(p[2][0], q[2][0])
                if oy > 0.5 * min(p[2][3] - p[2][1], q[2][3] - q[2][1]) and ox <= 0:
                    merged += 1
    c["order_breaks"] += inversions; c["merged_pairs"] += merged
    ok = inversions == 0 and merged == 0 and all(p[1] >= 0 for p in probes)
    c["pages_fully_right"] += ok
    if not ok:
        bad_pages.append((n, f"regions {len(regions)}, uncovered {sum(p[1] < 0 for p in probes)}, order breaks {inversions}, merged pairs {merged}"))
print(f"pages {c['pages']} | body blocks {c['blocks']} | our regions {c['regions']}")
print(f"blocks inside one of our regions: {c['covered']}/{c['blocks']}")
print(f"reading-order breaks: {c['order_breaks']} | side-by-side blocks merged into one region: {c['merged_pairs']}")
print(f"pages with every block placed and ordered correctly: {c['pages_fully_right']}/{c['pages']}")
for n, why in bad_pages:
    print(f"   page {n}: {why}")
