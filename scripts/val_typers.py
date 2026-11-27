"""Score typer specs, rules and all (ocrl.structure.tag.Typer.tag, as the product runs it on given lines), on held-out
labelled pages of a pack (scripts/pack_structure_data.py): line types and blocks exact against the labeller's answer.
Blocks are cut where the typer starts one or the type changes (tables: one block per run), as e2e_judged.typer_on_azure.
    uv run python scripts/val_typers.py --pack out/structure_v6/pack6.json.gz --held out/structure_v6/v6_base.held_docs.json --set fresh SPEC...
"""
import argparse, gzip, json
from collections import Counter

import numpy as np
import torch
torch.set_num_threads(3)

from ocrl.structure.tag import ASIDE, TO_TYPE, Typer

ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--held", required=True); ap.add_argument("--set", default="fresh")
ap.add_argument("specs", nargs="+"); a = ap.parse_args()
held = set(json.load(open(a.held)))
pages = [p for p in json.load(gzip.open(a.pack, "rt", encoding="utf-8"))["haiku"] if p["doc"] in held and a.set in p]
print(f"{len(pages)} held-out pages from {len({p['doc'] for p in pages})} documents")
for spec in a.specs:
    ty = Typer(spec); ok = n = 0; B = Counter(); per = Counter(); pn = Counter()
    for p in pages:
        ls, W, H = p["lines"], p["W"], p["H"]
        if not ls: continue
        boxes = np.array([[l["box"][0] * W / 1000, l["box"][1] * H / 1000, l["box"][2] * W / 1000, l["box"][3] * H / 1000] for l in ls], np.float32)
        ty.open_item = False
        roles, starts, _ = ty.tag(boxes, [l["text"] for l in ls], W, H)
        ans = p[a.set]; want = {r: b["type"] for b in ans["blocks"] for r in b["lines"]}
        blocks, prev = [], None
        for l, r, st in zip(ls, roles, starts):
            t = TO_TYPE.get(r, "other")
            if l["id"] in want:
                n += 1; ok += t == want[l["id"]]; pn[want[l["id"]]] += 1; per[want[l["id"]]] += t == want[l["id"]]
            if blocks and t == prev and (st < 0.5 or t == "table") and t not in ASIDE:
                blocks[-1].append(l["id"])
            else:
                blocks.append([l["id"]])
            prev = t
        mine = {frozenset(b) for b in blocks}
        for b in ans["blocks"]:
            B["n"] += 1; B["ok"] += frozenset(b["lines"]) in mine
    weak = " ".join(f"{t} {100 * per[t] / pn[t]:.0f}" for t in ("title", "author", "affiliation", "list_item", "reference", "table", "figure") if pn[t])
    print(f"{spec}: types {100 * ok / n:.1f} | blocks exact {100 * B['ok'] / B['n']:.1f} | {weak}", flush=True)
