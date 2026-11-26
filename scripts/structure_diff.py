"""Side by side for one page: our blocks, and what the reference calls the lines in them."""
import json, sys
from collections import defaultdict, Counter
ours, ref, page = json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8")), int(sys.argv[3])
probes = [(r, b["id"], b["type"]) for b in ref["blocks"] for p in b["parts"] if p["page"] == page for r in p["regions"]]
def lookup(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hits = [p for p in probes if p[0][0] - .03 <= cx <= p[0][2] + .03 and p[0][1] - .03 <= cy <= p[0][3] + .03]
    return min(hits, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1])) if hits else None
for b in ours["blocks"]:
    for part in b["parts"]:
        if part["page"] != page: continue
        refs = Counter()
        for box in part["line_boxes"]:
            h = lookup(box); refs[f"{h[2]}:{h[1]}" if h else "-"] += 1
        mark = " " if all(k.split(":")[0] == b["type"] for k in refs) and len(refs) == 1 else "*"
        print(f"{mark} {b['id']:5s} {b['type']:12s} lines {len(part['line_boxes']):2d} y {part['regions'][0][1]:.2f}  ref {dict(refs)}  | {b['text'][:46]}")
