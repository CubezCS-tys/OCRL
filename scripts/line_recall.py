"""How many of Azure's lines does our own line finder find? Azure is only the ruler here.

    uv run python scripts/line_recall.py --clean --docs 0499-026-119-022 0073-000-239-002 [--cache DIR] [--why]

A line is "found" if one of our lines holds its centre, and "found well" if that line of ours is
about its size (not several lines run together, not a whole table row for one cell).
"""
import argparse, json, subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

from ocrl.blobs.clean import clean_page
from ocrl.blobs.layout import analyze
from ocrl.blobs.regions import plan


finder = None  # set from --finder when run as a script


def our_lines(gray, rules=None):
    """Our lines' boxes on the page, in our reading order."""
    out = []
    found = finder.lines(gray) if finder is not None else None
    for r, crop, hints, rank in plan(gray, rules, found, pad=16):
        lay = analyze(crop, hints=hints, hint_rank=rank)
        sats = lay.satellites()
        for ln in lay.lines:
            ids = [b for body in ln.bodies for b in [body] + sats.get(body, [])]
            if not ids:
                continue
            bx = lay.boxes[ids]
            out.append((bx[:, 0].min() + r.x0 - 16, bx[:, 1].min() + r.y0 - 16, bx[:, 2].max() + r.x0 - 16, bx[:, 3].max() + r.y0 - 16))
    return out


def render(samples: str, doc: str, n: int, cache: str) -> list[Path]:
    """Page images at 300 dpi, by crop box: Azure measured each page by its crop box, which in
    many of these PDFs is smaller than the media box that pdftoppm draws by default."""
    d = Path(cache) / doc
    files = sorted(d.glob("p-*.png"))
    if len(files) < n:
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-l", str(n), str(Path(samples) / doc / f"{doc}.pdf"), str(d / "p")], check=True)
        files = sorted(d.glob("p-*.png"))
    return files[:n]


def score_page(job):
    path, pg, clean = job
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
    rules = None
    if clean:
        cleaned = clean_page(gray); gray, rules = cleaned.gray, cleaned.rules
    ours = our_lines(gray, rules)
    c = Counter(pages=1, our_lines=len(ours))
    for ln in pg.get("lines", []):
        p = ln["polygon"]
        x0, y0, x1, y1 = min(p[0::2]) * sx, min(p[1::2]) * sy, max(p[0::2]) * sx, max(p[1::2]) * sy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hits = [o for o in ours if o[0] - 4 <= cx <= o[2] + 4 and o[1] - 4 <= cy <= o[3] + 4]
        tall_ok = [o for o in hits if (o[3] - o[1]) < 2.2 * (y1 - y0)]
        good = any((o[2] - o[0]) < 1.6 * (x1 - x0) + 40 for o in tall_ok)
        c["azure_lines"] += 1; c["found"] += bool(hits); c["found_well"] += good
        if not hits:
            c["why_missing"] += 1
        elif not tall_ok:
            c["why_ours_taller"] += 1  # several lines run together, or a line inside a drawing
        elif not good:
            c["why_ours_wider"] += 1  # a table cell or half-line inside one longer line of ours
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/home/yassine/inkscript_v2/data/samples")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--finder", default="", help="a line-finder checkpoint")
    ap.add_argument("--max-pages", type=int, default=12)
    ap.add_argument("--cache", default="out/line_recall_pages")
    ap.add_argument("--why", action="store_true", help="say what went wrong with the lines not found well")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.finder:
        from ocrl.blobs.linefinder import LineFinder
        finder = LineFinder(args.finder)
    total = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for doc in args.docs:
            az = json.loads((Path(args.samples) / doc / f"{doc}.json").read_text(encoding="utf-8"))
            az = az.get("analyzeResult", az)
            n = min(len(az["pages"]), args.max_pages)
            files = render(args.samples, doc, n, args.cache)
            per = list(pool.map(score_page, [(f, az["pages"][i], args.clean) for i, f in enumerate(files)]))
            c = sum(per, Counter())
            total += c
            a = max(c["azure_lines"], 1)
            why = f"  | missing {c['why_missing']} taller {c['why_ours_taller']} wider {c['why_ours_wider']}" if args.why else ""
            worst = sorted(range(len(per)), key=lambda i: per[i]["found_well"] - per[i]["azure_lines"])[:3]
            pages = ("  worst pages " + " ".join(f"p{i + 1}:{per[i]['found_well']}/{per[i]['azure_lines']}" for i in worst)) if args.why else ""
            print(f"{doc}: pages {c['pages']:2d}  Azure lines {c['azure_lines']:4d}  found {100 * c['found'] / a:5.1f}%  found as a line of its own size {100 * c['found_well'] / a:5.1f}%  (our lines {c['our_lines']}){why}{pages}", flush=True)
    a = total["azure_lines"]
    print(f"ALL: pages {total['pages']}  Azure lines {a}  found {100 * total['found'] / a:.1f}%  found well {100 * total['found_well'] / a:.1f}%"
          + (f"  | missing {total['why_missing']} taller {total['why_ours_taller']} wider {total['why_ours_wider']}" if args.why else ""))
