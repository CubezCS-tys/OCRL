"""A page for looking at the judged development pages: what the three judges, the cheap labellers and our own
structure models said for every line. Writes out/labeller/judged.html and judged_data.js (private scans: not tracked)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ocrl.structure.tag import TO_TYPE, Typer

torch.set_num_threads(2)
L = Path("out/labeller")
dev = json.loads(Path("out/census/dev_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
src = {f"{d['stem']}_p{d['page']}": d for d in json.load(open("out/census/dev_pages.json", encoding="utf-8"))}
WHO = [("opus", "Opus"), ("sonnet", "Sonnet"), ("argon", "Gemini 4 Argon"), ("haikuhigh", "Haiku"), ("flash", "Gemini Flash")]
v3 = Typer("out/structure_v3.pt")
pages = []
for p in dev["pages"]:
    ans = {}
    for w, _ in WHO:
        f = L / f"dev_{w}" / f"{p['id']}.json"
        if f.exists():
            a = json.load(open(f, encoding="utf-8")); ans[w] = ({r: b["type"] for b in a["blocks"] for r in b["lines"]}, {b["lines"][0] for b in a["blocks"] if b["lines"]})
    d = src[p["id"]]; W, H = d["width"], d["height"]
    boxes = np.array([l["box"] for l in d["lines"]]) * 300
    v3.open_item = False
    r3, _, _ = v3.tag(boxes, [l["text"] for l in d["lines"]], W * 300, H * 300)
    lines = []
    for i, l in enumerate(p["lines"]):
        rid = f"{p['page']}:{i}"
        votes = [ans[w][0].get(rid) for w in ("opus", "sonnet", "argon")]
        top, n = Counter(votes).most_common(1)[0]
        lines.append({"b": l["b"], "t": l["t"], "label": top if n >= 2 else None, "split": n < 3, "start": rid in ans["opus"][1],
                      "says": {w: ans[w][0].get(rid) for w, _ in WHO if w in ans}, "old": l["type"], "new": TO_TYPE.get(r3[i], "other")})
    pages.append({"id": p["id"], "kind": p["kind"], "img": "../census/" + p["img"], "lines": lines})
data = {"types": dev["types"], "who": WHO, "pages": pages}
(L / "judged_data.js").write_text("const J = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
n = sum(len(p["lines"]) for p in pages); ok = lambda k: sum(l[k] == l["label"] for p in pages for l in p["lines"] if l["label"])
print(f"{len(pages)} pages, {n} lines | judges split on {sum(l['split'] for p in pages for l in p['lines'])} | earlier model right on {ok('old')}, retrained on {ok('new')}")
