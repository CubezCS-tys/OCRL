"""Score a locally made master file with inkscript_v2's own scorer, unchanged.

Azure's lines are the ruler, at scoring time only: each Azure line is given to the block of ours
that holds its centre, its id is written into that block, and inkscript_v2's compare() does the rest.
Azure lines that land in no block of ours are reported, not dropped: the scorer counts them as wrong.

    uv run python scripts/score_with_compare.py --name 0175-000-004-016 --ours out/parsed/0175-000-004-016/0175-000-004-016.local.master.json
"""
import argparse, importlib.util, json, shutil, sys, tempfile
from collections import Counter
from pathlib import Path

V2 = Path("/home/yassine/inkscript_v2")
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="the key's name, e.g. 0073-000-239-002_p11-12")
ap.add_argument("--ours", required=True)
ap.add_argument("--keys", default=str(V2 / "out"))
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()
doc = args.name.split("_p")[0]
ours = json.load(open(args.ours, encoding="utf-8"))
az = json.loads((V2 / f"data/samples/{doc}/{doc}.json").read_text(encoding="utf-8"))
az = az.get("analyzeResult", az)

# our lines, as (page, box, block index, position of the line in the document)
mine, n = {}, 0
for bi, b in enumerate(ours["blocks"]):
    for part in b["parts"]:
        for box in part["line_boxes"]:
            mine.setdefault(part["page"], []).append((box, bi, n)); n += 1
pages = {p["number"] for p in ours["pages"]}
given: dict[int, list] = {}
c = Counter()
for pg in az["pages"]:
    p = pg["pageNumber"]
    if p not in pages:
        continue
    for i, ln in enumerate(pg.get("lines", [])):
        q = ln["polygon"]; x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hits = [m for m in mine.get(p, []) if m[0][0] - .02 <= cx <= m[0][2] + .02 and m[0][1] - .02 <= cy <= m[0][3] + .02]
        c["azure_lines"] += 1
        if not hits:
            c["in_no_block"] += 1
            continue
        hit = min(hits, key=lambda m: (m[0][2] - m[0][0]) * (m[0][3] - m[0][1]))
        # other blocks of ours that also cover a good part of this Azure line: it straddles them
        over = {m[1] for m in mine.get(p, []) if min(m[0][2], x1) - max(m[0][0], x0) > 0.3 * (x1 - x0) and min(m[0][3], y1) - max(m[0][1], y0) > 0.5 * (y1 - y0)}
        c["straddles_two_blocks"] += len(over) > 1
        given.setdefault(hit[1], []).append((hit[2], -cx, p, f"{p}:{i}"))

blocks = []
for bi, b in enumerate(ours["blocks"]):
    ids = sorted(given.get(bi, []))
    if not ids:
        continue
    parts = []
    for _, _, p, ref in ids:
        if parts and parts[-1]["page"] == p:
            parts[-1]["lines"].append(ref)
        else:
            parts.append({"page": p, "lines": [ref], "regions": next((q["regions"] for q in b["parts"] if q["page"] == p), [[0, 0, 0, 0]])})
    nb = {"id": b["id"], "type": b["type"], "text": b["text"], "parts": parts}
    if "aside" in b:
        nb["aside"] = True
    elif "order" in b:
        nb["order"] = b["order"]
    if b["type"] == "table":  # we label a table but do not parse its cells: every line is given to the table as a whole
        nb["table"] = {"header": [], "rows": [], "repeated_header_lines": [r for p in parts for r in p["lines"]]}
        for p in parts:
            p["lines"] = []
    blocks.append(nb)
model = {"id": ours["id"], "made_by": ours["made_by"], "document": {"type": "other", "type_label": "", "fields": {}}, "pages": ours["pages"],
         "blocks": blocks, "lists": [], "notes": [], "normalisations": [], "problems": [], "parts": []}

spec = importlib.util.spec_from_file_location("v2compare", V2 / "demo/compare.py")
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)
texts = compare.azure_lines()
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp); (out / "model/ocrl").mkdir(parents=True)
    (out / "model/ocrl" / f"{args.name}.master.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    compare.OUT = out  # where it looks for the model's file; the key comes from key_dir
    r = compare.compare(args.name, "ocrl", texts[doc], key_dir=Path(args.keys))
print(f"{args.name}: Azure lines {c['azure_lines']}, in none of our blocks {c['in_no_block']}, straddling two {c['straddles_two_blocks']}")
STRUCTURE = ("What each line is", "In the text, beside it, or set aside", "Blocks made of exactly the same lines",
             "Reading order: steps from one line to the next", "Table cells: same lines in the same cell, same span", "Lists with exactly the same items")
for what, a, b in r["rows"]:
    if what in STRUCTURE and a is not None and b:
        print(f"   {what:52s} {a:4d} / {b:<4d} {100 * a / b:5.1f}%")
if not args.quiet:
    kinds = Counter(d[0] for d in r["diffs"])
    print("   differences by kind:", dict(kinds))
json.dump({"rows": r["rows"], "ruler": dict(c)}, open(args.ours.replace(".json", ".compare.json"), "w"), ensure_ascii=False)
