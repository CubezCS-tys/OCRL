"""Build a picture-first explanation of how real-ink labels are made, from one real page.

Output goes to out/explain/ (not tracked): it contains crops of a private scan.
    OMP_NUM_THREADS=2 uv run python scripts/explain_realink.py
"""
import base64, html, json
from pathlib import Path

import cv2
import numpy as np
import torch

from ocrl.blobs.clean import clean_page
from ocrl.blobs.layout import analyze
from ocrl.blobs.regions import find_regions
from ocrl.recog.crops import cluster_ids, crop
from ocrl.recog.decode import greedy
from ocrl.recog.model import load
from ocrl.recog.realink import fold, visual_units
from ocrl.text import strip_marks

torch.set_num_threads(2); cv2.setNumThreads(2)
import subprocess, tempfile
from rapidfuzz.distance import Levenshtein

DOC, PAGE = "0013-002-001-005", 2
model, vocab, _ = load("out/recognizer_v3.pt"); model.eval()
png = lambda a: "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", a)[1].tobytes()).decode()
PALETTE = [(200, 90, 20), (30, 130, 60), (160, 50, 150), (20, 110, 190), (190, 40, 60), (110, 110, 20), (90, 60, 200), (0, 140, 140)]


def page_words(img, pg):
    """Our clusters on the page, and for every Azure word: the clusters in its box and what the rules expect."""
    sx, sy = img.shape[1] / pg["width"], img.shape[0] / pg["height"]
    gray = clean_page(img).gray
    clusters = []
    for r in find_regions(gray):
        pad = np.pad(gray[r.y0 : r.y1, r.x0 : r.x1], 16, constant_values=255)
        lay = analyze(pad); ids = cluster_ids(lay)
        for ln in lay.lines:
            for b in ln.bodies:
                bx = lay.boxes[ids[b]]
                x0, y0, x1, y1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
                body = lay.boxes[b]
                arr = crop(pad, lay, ids[b], ln.baseline, ln.scale)
                with torch.no_grad():
                    logp = model(torch.from_numpy(arr).float()[None] / 255)[:, 0].numpy()
                off = np.array([r.x0 - 16, r.y0 - 16])
                clusters.append({"box": (x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1]), "mask": np.isin(lay.labels[y0:y1, x0:x1], ids[b]),
                                 "c": ((body[0] + body[2]) / 2 + off[0], (body[1] + body[3]) / 2 + off[1]), "read": greedy(logp, vocab)})
    words = []
    for wi, w in enumerate(pg["words"]):
        q = w["polygon"]; box = (min(q[0::2]) * sx, min(q[1::2]) * sy, max(q[0::2]) * sx, max(q[1::2]) * sy)
        text = fold(w["content"]); us = visual_units(text)
        inside = sorted([c for c in clusters if box[0] <= c["c"][0] <= box[2] and box[1] <= c["c"][1] <= box[3]], key=lambda c: -c["c"][0])
        verdict = "none" if not inside else "skip" if us is None else "ok" if len(inside) == len(us) else "few" if len(inside) < len(us) else "many"
        agree = [strip_marks(c["read"]) == strip_marks(u) for c, u in zip(inside, us)] if verdict == "ok" else None
        words.append({"i": wi, "text": text, "box": box, "units": us or [], "inside": inside, "verdict": verdict, "agree": agree, "gray": gray})
    return gray, clusters, words, (sx, sy)


def picture(gray, x0, y0, x1, y1, word_boxes=(), mine=(), zoom=3):
    x0, y0, x1, y1 = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, gray.shape[1])), int(min(y1, gray.shape[0]))
    out = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    for k, cl in enumerate(mine):
        bx0, by0, bx1, by1 = cl["box"]
        ya, yb, xa, xb = max(by0 - y0, 0), min(by1 - y0, out.shape[0]), max(bx0 - x0, 0), min(bx1 - x0, out.shape[1])
        if yb <= ya or xb <= xa:
            continue
        m = cl["mask"][ya - (by0 - y0) : yb - (by0 - y0), xa - (bx0 - x0) : xb - (bx0 - x0)]
        out[ya:yb, xa:xb][m] = np.array(PALETTE[k % len(PALETTE)][::-1], np.float32)
    out = cv2.resize(out, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    for (a, b, c, d) in word_boxes:
        cv2.rectangle(out, (int((a - x0) * zoom), int((b - y0) * zoom)), (int((c - x0) * zoom), int((d - y0) * zoom)), (200, 110, 20), 2)
    return png(out)


img = cv2.imread(f"data/real/0013/pages/p-{PAGE:02d}.png", 0)
az = json.loads(Path(f"/home/yassine/inkscript_v2/data/samples/{DOC}/{DOC}.json").read_text(encoding="utf-8"))
pg = az.get("analyzeResult", az)["pages"][PAGE - 1]
gray, clusters, words, (sx, sy) = page_words(img, pg)

# one line of the page, for the two overview pictures
line = pg["lines"][6]; q = line["polygon"]
lb = (min(q[0::2]) * sx - 10, min(q[1::2]) * sy - 22, max(q[0::2]) * sx + 10, max(q[1::2]) * sy + 22)
in_line = [w for w in words if lb[0] <= (w["box"][0] + w["box"][2]) / 2 <= lb[2] and lb[1] <= (w["box"][1] + w["box"][3]) / 2 <= lb[3]]
line_cl = [c for c in clusters if lb[0] <= c["c"][0] <= lb[2] and lb[1] <= c["c"][1] <= lb[3]]
overview = {"azure": picture(gray, *lb, word_boxes=[w["box"] for w in in_line], zoom=2), "ours": picture(gray, *lb, mine=line_cl, zoom=2),
            "n_words": len(in_line), "n_clusters": len(line_cl)}


def card(w):
    b = w["box"]; m = 26
    return {"text": w["text"], "units": w["units"], "n": len(w["inside"]), "verdict": w["verdict"],
            "img": picture(w["gray"], b[0] - m, b[1] - m, b[2] + m, b[3] + m, word_boxes=[b], mine=w["inside"], zoom=4),
            "reads": [c["read"] for c in w["inside"]], "agree": w["agree"]}


dist = lambda w: [Levenshtein.distance(strip_marks(c["read"]), strip_marks(u)) for c, u in zip(w["inside"], w["units"])]
multi = lambda w: len(w["units"]) >= 2 and len(w["text"]) >= 4
# a correction: the count matched and the reader is one letter off somewhere; a slip: nothing in common anywhere
is_fix = lambda w: w["verdict"] == "ok" and multi(w) and max(dist(w)) == 1 and all(ch not in "0123456789٠١٢٣٤٥٦٧٨٩" for ch in w["text"])
is_slip = lambda w: w["verdict"] == "ok" and len(w["units"]) >= 2 and max(len(u) for u in w["units"]) >= 2 and all(
    d >= max(len(strip_marks(c["read"])), len(u), 1) for d, c, u in zip(dist(w), w["inside"], w["units"]))
pick = lambda ws, f, n: [card(w) for w in [w for w in ws if f(w)][:n]]

# slipped labels are rare on a clean page; look for them on harder pages of the downloaded documents
slips, S3 = [], Path("/home/yassine/data/mandumah_s3")
for stem, page in (("0887-000-029-010", 4), ("0678-011-000-015", 2), ("1425-000-005-006", 10), ("2051-000-002-017", 6), ("0679-000-003,004-010", 10)):
    if len(slips) >= 3 or not (S3 / stem / f"{stem}.pdf").exists():
        continue
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-f", str(page), "-l", str(page), str(S3 / stem / f"{stem}.pdf"), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL)
        f = sorted(Path(tmp).glob("p-*.png"))
        if not f:
            continue
        a2 = json.loads((S3 / stem / f"{stem}.json").read_text(encoding="utf-8")); a2 = a2.get("analyzeResult", a2)
        _, _, w2, _ = page_words(cv2.imread(str(f[0]), 0), a2["pages"][page - 1])
        slips += pick(w2, is_slip, 3 - len(slips))

data = {
    "overview": overview,
    "good": pick(words, lambda w: w["verdict"] == "ok" and multi(w) and all(w["agree"]), 3),
    "fix": pick(words, is_fix, 3),
    "slipped": slips,
    "few": pick(words, lambda w: w["verdict"] == "few" and multi(w), 2),
    "many": pick(words, lambda w: w["verdict"] == "many" and len(w["text"]) >= 3, 2),
    "page_counts": {k: sum(w["verdict"] == k for w in words) for k in ("ok", "few", "many", "none", "skip")},
    "palette": ["#%02x%02x%02x" % p for p in PALETTE],
}
out = Path("out/explain"); out.mkdir(parents=True, exist_ok=True)
(out / "realink_data.js").write_text("window.RI = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8")
print({k: len(data[k]) for k in ("good", "fix", "slipped", "few", "many")}, data["page_counts"])
for c in data["fix"] + data["slipped"]: print("  ", c["text"], c["units"], c["reads"])
