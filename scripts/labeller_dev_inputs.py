"""Labeller inputs for the census's development pages: one folder per page, as for the keyed documents."""
import json
from pathlib import Path

import cv2

OUT = Path("out/labeller/dev_inputs")
pages = json.load(open("out/census/dev_pages.json", encoding="utf-8"))
names = []
for d in pages:
    name = f"{d['stem']}_p{d['page']}"; f = OUT / name; f.mkdir(parents=True, exist_ok=True)
    img = cv2.imread("out/census/dev/" + d["img"])
    s = 1150 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    sx, sy = 1150 / d["width"], img.shape[0] / d["height"]
    lines = []
    for i, l in enumerate(d["lines"]):
        x0, y0, x1, y1 = l["box"]
        cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
        cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
        lines.append({"id": f"{d['page']}:{i}", "box": [round(1000 * x0 / d["width"]), round(1000 * y0 / d["height"]), round(1000 * x1 / d["width"]), round(1000 * y1 / d["height"])], "text": l["text"]})
    cv2.imwrite(str(f / f"page_{d['page']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    (f / "lines.json").write_text(json.dumps(lines, ensure_ascii=False, indent=0), encoding="utf-8")
    names.append((name, d["img"].split("_")[1], len(lines)))
# batches of ten pages that each mix the kinds, for labellers that work a batch at a time
names.sort(key=lambda t: (int(Path(t[0]).name.__hash__() % 1), t[1]))
order = sorted(range(len(names)), key=lambda i: (i % 7, i))
batches = [[names[i][0] for i in order[k::7]] for k in range(7)]
json.dump(batches, open("out/labeller/dev_batches.json", "w"), indent=1)
print(len(names), "pages,", sum(n for _, _, n in names), "lines;", [len(b) for b in batches], "pages a batch")
