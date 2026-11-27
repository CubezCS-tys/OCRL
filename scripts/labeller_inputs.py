"""What a labelling model is given for a page: the scan with Azure's lines numbered on it, the lines as text, and the guide.

Used to test a model as labeller against the hand-keyed development documents (it never sees the keys).
    uv run python scripts/labeller_inputs.py
"""
import glob, json
from pathlib import Path

import cv2

V2 = Path("/home/yassine/inkscript_v2")
OUT = Path("out/labeller")
DOCS = {"0073-000-239-002_p11-12": ("0073-000-239-002", [11, 12]), "0175-000-004-016": ("0175-000-004-016", [1, 2]), "0499-026-119-022": ("0499-026-119-022", [1, 2])}
for name, (doc, pages) in DOCS.items():
    az = json.loads((V2 / f"data/samples/{doc}/{doc}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
    d = OUT / "inputs" / name; d.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in pages:
        f = next(f for f in glob.glob(f"out/line_recall_pages/{doc}/p-*.png") if int(Path(f).stem.split("-")[1]) == p)
        img = cv2.imread(f); pg = az["pages"][p - 1]
        s = 1150 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sx, sy = 1150 / pg["width"], img.shape[0] / pg["height"]
        for i, l in enumerate(pg["lines"]):
            q = l["polygon"]; x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
            cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
            cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
            lines.append({"id": f"{p}:{i}", "box": [round(1000 * x0 / pg["width"]), round(1000 * y0 / pg["height"]), round(1000 * x1 / pg["width"]), round(1000 * y1 / pg["height"])], "text": l["content"]})
        cv2.imwrite(str(d / f"page_{p}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    (d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False, indent=0), encoding="utf-8")
    print(name, len(lines), "lines")
guide = Path("docs/label_guide.md").read_text(encoding="utf-8").split("## Open questions")[0]
guide += """## Decisions taken for now

- A paragraph that opens with a dash, a bullet, or a number or letter with its mark is a `list_item`: every line of it, not only the first. Several such paragraphs in a row are several list items.
- A numbered heading is a `heading`.
- A label such as "ملخص" or "Abstract" standing above an abstract belongs to the `abstract`.
- Text on a magazine page: the short introduction between the title and the body is `standfirst`; large display text repeating a sentence of the body is `pull_quote`.
- Text printed inside a photograph, drawing or chart is `figure`.
- Marks that are not text (specks, stamps, ornaments, show-through read as words) are `noise`; writing added by hand is `handwriting`.
- Arabic reads right to left: on a page with columns the rightmost column comes first.
- A paragraph or list item that runs on into the next column or the next page is still one block.
"""
(OUT / "guide.md").write_text(guide, encoding="utf-8")
