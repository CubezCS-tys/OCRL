"""line_diag.py's cases for several finders at once, overall and on table-heavy pages, on CPU workers.

For every page (first --max-pages of each document, drawn at 300 dpi by the crop box and cleaned as the pipeline does):
  finder  the finder's own lines (as pipeline.read_page gets them, before any later step) against Azure's lines
  layout  line_diag.py's lines (finder + regions + layout, line_recall.our_lines) against Azure's lines

    python scripts/lf2_diag.py --samples held50 --docs ... --finders linefinder.pt out/lf2_a.pt --workers 48 --out res.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
import torch

torch.set_num_threads(1)
import line_recall as LR
from line_diag import CASES, cases
from lf2_targets import table_score
from ocrl.blobs.clean import clean_page
from ocrl.blobs.linefinder import LineFinder

_F = {}


def run_doc(job):
    doc, a = job
    for f in a["finders"]:
        if f not in _F:
            _F[f] = LineFinder(f, "cpu")
    azj = json.loads((Path(a["samples"]) / doc / f"{doc}.json").read_text(encoding="utf-8")); azj = azj.get("analyzeResult", azj)
    n = min(len(azj["pages"]), a["max_pages"])
    out = {f: {"finder": Counter(), "layout": Counter(), "finder_t": Counter(), "layout_t": Counter()} for f in a["finders"]}
    try:
        files = LR.render(a["samples"], doc, n, a["cache"])
    except Exception:
        return doc, out, 0
    pages = 0
    for pi, fpath in enumerate(files):
        pg = azj["pages"][pi]
        gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if gray is None or not pg.get("lines"):
            continue
        sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
        cleaned = clean_page(gray); g = cleaned.gray
        az = [(min(l["polygon"][0::2]) * sx, min(l["polygon"][1::2]) * sy, max(l["polygon"][0::2]) * sx, max(l["polygon"][1::2]) * sy) for l in pg["lines"]]
        tab = table_score(az) >= 0.3
        pages += 1
        for f in a["finders"]:
            LR.finder = _F[f]
            raw = [tuple(b) for b in _F[f].lines(gray if a["raw_feed"] else g)]
            for kind, ours in (("finder", raw), ("layout", LR.our_lines(g, cleaned.rules) if a["layout"] else None)):
                if ours is None:
                    continue
                cs = cases(ours, az)
                out[f][kind].update(cs)
                if tab:
                    out[f][kind + "_t"].update(cs)
                if a["draw"] and doc in a["draw_docs"] and kind == "finder":
                    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); lay = img.copy()
                    for o in ours:
                        cv2.rectangle(lay, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (255, 200, 120), -1)
                    img = cv2.addWeighted(lay, 0.35, img, 0.65, 0)
                    for o in ours:
                        cv2.rectangle(img, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (160, 90, 0), 2)
                    from line_diag import COL
                    for z, c in zip(az, cs):
                        if c != "one-to-one":
                            cv2.rectangle(img, (int(z[0]), int(z[1])), (int(z[2]), int(z[3])), COL[c], 4)
                    s = 1400 / img.shape[1]
                    Path(a["draw"]).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(a["draw"]) / f"{doc}_p{pi + 1:02d}_{Path(f).stem}.jpg"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return doc, out, pages


def fmt(c):
    t = max(sum(c.values()), 1)
    return f"{sum(c.values()):6d} lines | " + " | ".join(f"{k} {100 * c[k] / t:.2f}%" for k in CASES)


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples"); ap.add_argument("--docs", nargs="*"); ap.add_argument("--docs-from", help="json list, or ids.json:KEY")
    ap.add_argument("--finders", nargs="+"); ap.add_argument("--max-pages", type=int, default=12); ap.add_argument("--cache", default="cache300")
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--out"); ap.add_argument("--no-layout", action="store_true")
    ap.add_argument("--raw-feed", action="store_true", help="the finder sees the page before clean_page()")
    ap.add_argument("--draw", default=""); ap.add_argument("--draw-docs", nargs="*", default=[])
    a = vars(ap.parse_args())
    a["layout"] = not a.pop("no_layout")
    docs = a["docs"] or []
    if a["docs_from"]:
        p, _, key = a["docs_from"].partition(":")
        d = json.load(open(p)); docs += d[key] if key else d
    tot = {f: {"finder": Counter(), "layout": Counter(), "finder_t": Counter(), "layout_t": Counter()} for f in a["finders"]}
    pages = 0
    with mp.get_context("fork").Pool(a["workers"]) as pool:
        for k, (doc, out, np_) in enumerate(pool.imap_unordered(run_doc, [(d, a) for d in docs]), 1):
            pages += np_
            for f in out:
                for kk in out[f]:
                    tot[f][kk] += out[f][kk]
            if k % 50 == 0:
                print(k, "docs", flush=True)
    print(f"{len(docs)} documents, {pages} pages")
    for f in a["finders"]:
        for kk in ("finder", "finder_t", "layout", "layout_t"):
            if sum(tot[f][kk].values()):
                print(f"{f:28s} {kk:9s} {fmt(tot[f][kk])}")
    if a["out"]:
        json.dump({f: {k: dict(v) for k, v in d.items()} for f, d in tot.items()} | {"_docs": len(docs), "_pages": pages}, open(a["out"], "w"), indent=1)
