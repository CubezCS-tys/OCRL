"""The structure model on Azure's own lines: what it would give for a page already read by Azure, with no Gemini.

Writes a master file whose blocks cite Azure line ids directly, and scores it with inkscript_v2's compare().

    uv run python scripts/type_azure_lines.py --typer out/structure_big.pt --name 0175-000-004-016
"""
import argparse, importlib.util, json, tempfile
from pathlib import Path

import numpy as np

from ocrl.structure.tag import ASIDE, TO_TYPE, Typer

V2 = Path("/home/yassine/inkscript_v2")
ap = argparse.ArgumentParser()
ap.add_argument("--typer", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--keys", default=str(V2 / "out"))
ap.add_argument("--samples", default=str(V2 / "data/samples"))
ap.add_argument("--starts", default="model", choices=["model", "azure"], help="where a block begins: the structure model's call, or Azure's own paragraphs")
args = ap.parse_args()
doc = args.name.split("_p")[0]
key = json.load(open(Path(args.keys) / f"{args.name}.master.json", encoding="utf-8"))
az = json.loads((Path(args.samples) / doc / f"{doc}.json").read_text(encoding="utf-8"))
az = az.get("analyzeResult", az)
typer = Typer(args.typer)
want = {p["number"] for p in key["pages"]}
blocks, cur, last_flow = [], None, None
for pg in az["pages"]:
    p = pg["pageNumber"]
    if p not in want or not pg.get("lines"):
        continue
    dpi = 300.0  # the model saw pixel boxes at the scan's size; only ratios matter
    boxes = np.array([[min(l["polygon"][0::2]), min(l["polygon"][1::2]), max(l["polygon"][0::2]), max(l["polygon"][1::2])] for l in pg["lines"]]) * dpi
    ink = None
    img = next((f for f in Path("out/line_recall_pages", doc).glob("p-*.png") if int(f.stem.split("-")[1]) == p), None)
    if img is not None:
        import cv2
        from ocrl.structure.features import ink_features
        g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        ink = ink_features(g, boxes / dpi * np.array([g.shape[1] / pg["width"], g.shape[0] / pg["height"]] * 2))
    roles, starts, _ = typer.tag(boxes, [l["content"] for l in pg["lines"]], pg["width"] * dpi, pg["height"] * dpi, ink)
    if args.starts == "azure":
        first = {q["spans"][0]["offset"] for q in az.get("paragraphs", [])}
        starts = np.array([1.0 if l["spans"][0]["offset"] in first else 0.0 for l in pg["lines"]])
    cur = None
    for i, (role, start) in enumerate(zip(roles, starts)):
        kind, ref = TO_TYPE.get(role, "other"), f"{p}:{i}"
        if kind == "table":
            if cur is None or cur["type"] != "table" or cur["parts"][-1]["page"] != p:
                cur = {"id": f"b{len(blocks)}", "type": "table", "text": "", "parts": [{"page": p, "lines": [], "regions": [[0, 0, 0, 0]]}],
                       "table": {"header": [], "rows": [], "repeated_header_lines": []}}
                blocks.append(cur)
            cur["table"]["repeated_header_lines"].append(ref)
            continue
        if cur is not None and cur["type"] == kind and start < 0.5 and kind not in ASIDE and kind != "table" and cur["parts"][-1]["page"] == p:
            cur["parts"][-1]["lines"].append(ref)
            continue
        if start < 0.5 and kind not in ASIDE and last_flow is not None and last_flow["type"] == kind and cur is None:
            last_flow["parts"].append({"page": p, "lines": [ref], "regions": [[0, 0, 0, 0]]})  # carried over the page break
            cur = last_flow
            continue
        cur = {"id": f"b{len(blocks)}", "type": kind, "text": "", "parts": [{"page": p, "lines": [ref], "regions": [[0, 0, 0, 0]]}]}
        if kind in ASIDE:
            cur["aside"] = True
        blocks.append(cur)
        if kind not in ASIDE and kind not in ("footnote", "figure"):
            last_flow = cur
model = {"id": doc, "made_by": {"text": "Azure", "structure": "OCRL structure model"}, "document": {"type": "other", "type_label": "", "fields": {}},
         "pages": key["pages"], "blocks": blocks, "lists": [], "notes": [], "normalisations": [], "problems": [], "parts": []}
spec = importlib.util.spec_from_file_location("v2compare", V2 / "demo/compare.py")
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp); (out / "model/ocrl").mkdir(parents=True)
    (out / "model/ocrl" / f"{args.name}.master.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    compare.OUT = out
    r = compare.compare(args.name, "ocrl", compare.azure_lines()[doc], key_dir=Path(args.keys))
STRUCTURE = ("What each line is", "In the text, beside it, or set aside", "Blocks made of exactly the same lines", "Reading order: steps from one line to the next")
print(args.name + ":", " | ".join(f"{w.split(':')[0]} {a}/{b} ({100 * a / b:.0f}%)" for w, a, b in r["rows"] if w in STRUCTURE and b))
from collections import Counter
miss = Counter()
K, M = compare.view(key), compare.view(model)
for i, t in K["type"].items():
    if M["type"].get(i) != t:
        miss[(t, M["type"].get(i, "nothing"))] += 1
print("   key says / we say:", ", ".join(f"{a}/{b}×{n}" for (a, b), n in miss.most_common(10)))
