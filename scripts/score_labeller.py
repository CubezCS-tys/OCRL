"""Score a labelling model's answer for a hand-keyed development document with inkscript_v2's compare().

The answer is {"blocks": [{"type": ..., "lines": ["11:0", ...]}, ...]} in reading order, citing Azure line ids.
    uv run python scripts/score_labeller.py --labels out/labeller/haiku --name 0175-000-004-016
"""
import argparse, importlib.util, json, tempfile
from collections import Counter
from pathlib import Path

V2 = Path("/home/yassine/inkscript_v2")
ASIDE = {"running_head", "running_foot", "page_number", "noise", "handwriting"}
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--name", nargs="+", required=True)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("v2compare", V2 / "demo/compare.py")
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)
WANT = ("What each line is", "Blocks made of exactly the same lines", "Reading order: steps from one line to the next")
tot = Counter()
for name in a.name:
    doc = name.split("_p")[0]
    f = Path(a.labels) / f"{name}.json"
    if not f.exists():
        print(f"{name}: no answer file"); continue
    key = json.load(open(V2 / "out" / f"{name}.master.json", encoding="utf-8"))
    ans = json.load(open(f, encoding="utf-8"))
    given = {l["id"] for l in json.load(open(f"out/labeller/inputs/{name}/lines.json", encoding="utf-8"))}
    used = Counter(r for b in ans["blocks"] for r in b["lines"])
    blocks = []
    for i, b in enumerate(ans["blocks"]):
        parts = []
        for ref in b["lines"]:
            p = int(ref.split(":")[0])
            if parts and parts[-1]["page"] == p: parts[-1]["lines"].append(ref)
            else: parts.append({"page": p, "lines": [ref], "regions": [[0, 0, 0, 0]]})
        nb = {"id": f"b{i}", "type": b["type"], "text": "", "parts": parts}
        if b["type"] in ASIDE: nb["aside"] = True
        if b["type"] == "table": nb["table"] = {"header": [], "rows": [], "repeated_header_lines": list(b["lines"])}
        blocks.append(nb)
    model = {"id": doc, "made_by": {"text": "Azure", "structure": "labelling model"}, "document": {"type": "other", "type_label": "", "fields": {}},
             "pages": key["pages"], "blocks": blocks, "lists": [], "notes": [], "normalisations": [], "problems": [], "parts": []}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp); (out / "model/lab").mkdir(parents=True)
        (out / "model/lab" / f"{name}.master.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        compare.OUT = out
        r = compare.compare(name, "lab", compare.azure_lines()[doc], key_dir=V2 / "out")
    rows = {w: (x, y) for w, x, y in r["rows"] if w in WANT and y}
    for w, (x, y) in rows.items(): tot[w, 0] += x; tot[w, 1] += y
    K, M = compare.view(key), compare.view(model)
    miss = Counter((t, M["type"].get(i, "nothing")) for i, t in K["type"].items() if M["type"].get(i) != t)
    print(f"{name}: " + " | ".join(f"{w.split(':')[0][:22]} {x}/{y} ({100 * x / y:.0f}%)" for w, (x, y) in rows.items()) + f" | lines left out {len(given - set(used))}, used twice {sum(v > 1 for v in used.values())}")
    print("   key says / it says:", ", ".join(f"{k}/{m}×{n}" for (k, m), n in miss.most_common(6)))
print("ALL: " + " | ".join(f"{w.split(':')[0][:22]} {tot[w, 0]}/{tot[w, 1]} ({100 * tot[w, 0] / max(tot[w, 1], 1):.0f}%)" for w in WANT))
