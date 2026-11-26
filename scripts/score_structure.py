"""Score a locally made master file against a reference one: block types, block cuts, reading order.

The unit is one of OUR lines. It is looked up in the reference by where it sits on the page.
"""
import argparse, json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--pages", default="", help="score only these pages: first-last")
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()
ours, ref = json.load(open(args.ours, encoding="utf-8")), json.load(open(args.ref, encoding="utf-8"))
lo, hi = (int(v) for v in args.pages.split("-")) if args.pages else (1, 10**6)

probes = defaultdict(list)  # page -> [(box, block id, type, order)]
for b in ref["blocks"]:
    for part in b["parts"]:
        for r in part["regions"]:
            probes[part["page"]].append((r, b["id"], b["type"], b.get("order")))


def lookup(page, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hits = [p for p in probes[page] if p[0][0] - 0.03 <= cx <= p[0][2] + 0.03 and p[0][1] - 0.03 <= cy <= p[0][3] + 0.03]
    return min(hits, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1])) if hits else None


c, conf = Counter(), Counter()
ref_sets, our_sets, seq = defaultdict(set), {}, []
n = 0
for b in ours["blocks"]:
    mine = set()
    for part in b["parts"]:
        if not lo <= part["page"] <= hi:
            continue
        for box in part["line_boxes"]:
            n += 1; mine.add(n)
            hit = lookup(part["page"], box)
            c["lines"] += 1
            if hit is None:
                c["lines_outside_reference"] += 1
                conf[("(nothing)", b["type"])] += 1
                continue
            ref_sets[hit[1]].add(n)
            c["type_right"] += hit[2] == b["type"]
            conf[(hit[2], b["type"])] += 1
            if hit[3] is not None and b.get("order") is not None:
                seq.append(hit[3])
    if mine:
        our_sets[b["id"]] = mine
ours_frozen = {frozenset(s) for s in our_sets.values()}
c["ref_blocks"] = len(ref_sets)
c["blocks_cut_the_same"] = sum(frozenset(s) in ours_frozen for s in ref_sets.values())
c["reading_steps"] = max(len(seq) - 1, 0)
c["reading_steps_right"] = sum(a <= b for a, b in zip(seq, seq[1:]))
pc = lambda a, b: f"{c[a]}/{c[b]} ({100 * c[a] / max(c[b], 1):.1f}%)"
print(f"lines given the reference's type: {pc('type_right', 'lines')} | blocks cut the same: {pc('blocks_cut_the_same', 'ref_blocks')} | reading steps in order: {pc('reading_steps_right', 'reading_steps')}")
if not args.quiet:
    per = defaultdict(Counter)
    for (r, o), k in conf.items():
        per[r]["n"] += k; per[r]["ok"] += k if r == o else 0
    for r, v in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        wrong = ", ".join(f"{o}×{k}" for (rr, o), k in conf.most_common() if rr == r and o != r)
        print(f"   {r:14s} {v['ok']:4d}/{v['n']:<4d} {100 * v['ok'] / v['n']:5.1f}%   {('read as: ' + wrong) if wrong else ''}")
json.dump(dict(c), open(args.ours.replace(".json", ".score.json"), "w"))
