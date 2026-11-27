"""Why did layout drop a line the finder marked? For each Azure line a lost_causes run put down to "dropped by layout"
(or "read as empty"), replay the page's stages and say which rule lost it.

    uv run python scripts/drop_diag.py out/lost/head_all.json [--cause "dropped by layout"] [--max 40]

Reasons, first that applies:
  no region         the finder line's centre lies in no region of the page plan (regions are found from letter-sized ink)
  no letter         the hint holds no blob that took it as its line (none a third of its height; marks too few)
  speck             the line came out of layout but its ink is under 12 pixels both ways (pipeline.DROP_SPECKS)
  swallowed         the hint's letters went into a line that also holds another hint's (the line box then links elsewhere)
  elsewhere         a line of ours holds the ink, but its box does not link to Azure's (shape, or a neighbour's ink)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lost_causes as LC  # noqa: E402
from line_diag import cases  # noqa: E402


def link(o, z) -> bool:
    ocx, ocy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    zcx, zcy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
    oq, zq = 0.25 * (o[3] - o[1]), 0.25 * (z[3] - z[1])
    return (o[0] - 4 <= zcx <= o[2] + 4 and o[1] + oq <= zcy <= o[3] - oq) or (z[0] <= ocx <= z[2] and z[1] + zq <= ocy <= z[3] - zq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--cause", default="dropped by layout"); ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from ocrl.pipeline import OCR
    import ocrl.pipeline as PL
    from ocrl.blobs.clean import clean_page
    from ocrl.blobs.regions import plan, split_at_uprights, split_at_gaps
    from ocrl.blobs.layout import analyze

    R = json.load(open(a.run))
    pages = {(p["doc"], p["page"]): p for p in json.load(open(LC.ROOT / "out/lost/pages.json"))}
    ocr = OCR(str(LC.ROOT / "out/recognizer_v3.pt"), linefinder=str(LC.ROOT / "out/linefinder.pt"), line_reader=None)
    why, rows = Counter(), []
    n = 0
    for row in R["pages"]:
        lost = [l for l in row.get("lost_lines", []) if l["cause"] == a.cause]
        if not lost or n >= a.max:
            continue
        n += 1
        p = pages[(row["doc"], row["page"])]
        gray = LC.page_image(Path(p["dir"]), p["doc"], p["page"])
        # the turn, as read_page decided it, is not replayed here: pages that were turned are skipped
        if any("turned upright" in w for w in row.get("warnings", [])):
            continue
        cleaned = clean_page(gray)
        g = cleaned.gray
        lines = ocr.finder.lines(g)
        lines = split_at_uprights(lines, cleaned.uprights, g)
        lines = split_at_gaps(lines, g)
        if PL.CELL_INK:
            lines = PL.cell_ink(lines, cleaned.uprights, g)
        regs = plan(g, cleaned.rules, lines, pad=16)
        cx, cy = (lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2
        for l in lost:
            z = l["box"]
            fi = [i for i in range(len(lines)) if link(lines[i], z)]
            info = {"doc": p["doc"], "page": p["page"], "box": z, "text": l["text"], "finder": [[round(float(v)) for v in lines[i]] for i in fi]}
            if not fi:
                reason = "finder line not linked after cuts"
            else:
                reason = "no region"
                for r, crop, hints, rank in regs:
                    off = np.array([r.x0 - 16, r.y0 - 16] * 2, np.float32)
                    mine = [i for i in fi if r.x0 <= cx[i] <= r.x1 and r.y0 <= cy[i] <= r.y1]
                    if not mine or hints is None:
                        continue
                    # which hints of this region are ours
                    loc = lines[mine[0]] - off
                    k = int(np.argmin(np.abs(hints - loc).sum(1)))
                    lay = analyze(crop, hints=hints, hint_rank=rank)
                    # lines of the layout and their boxes
                    found_line = False
                    for li, ln in enumerate(lay.lines):
                        ids = np.flatnonzero(lay.line == li)
                        if not len(ids):
                            continue
                        bx = lay.boxes[ids]
                        box = (bx[:, 0].min() + off[0], bx[:, 1].min() + off[1], bx[:, 2].max() + off[0], bx[:, 3].max() + off[1])
                        bodies = lay.boxes[ln.bodies] if ln.bodies else np.zeros((0, 4))
                        h = hints[k]
                        bcx, bcy = (bodies[:, 0] + bodies[:, 2]) / 2, (bodies[:, 1] + bodies[:, 3]) / 2
                        inh = ((bcx >= h[0] - 1.5 * (h[3] - h[1])) & (bcx <= h[2] + 1.5 * (h[3] - h[1])) & (bcy >= h[1]) & (bcy <= h[3])).any() if len(bodies) else False
                        if inh:
                            found_line = True
                            w_, h_ = box[2] - box[0], box[3] - box[1]
                            others = sum(1 for kk, hh in enumerate(hints) if kk != k and any(
                                (hh[0] <= x <= hh[2]) and (hh[1] <= y <= hh[3]) for x, y in zip(bcx, bcy)))
                            if PL.DROP_SPECKS and max(w_, h_) < 12:
                                reason = "speck"
                            elif others:
                                reason = "swallowed"
                            else:
                                reason = "elsewhere"
                            info["ours"] = [round(float(v)) for v in box]
                            break
                    if not found_line:
                        # the blobs whose centre is in Azure's box: their heights against the hint's
                        zl = np.array(z, np.float32) - off
                        bc = (lay.boxes[:, :2] + lay.boxes[:, 2:]) / 2
                        inz = (bc[:, 0] >= zl[0]) & (bc[:, 0] <= zl[2]) & (bc[:, 1] >= zl[1]) & (bc[:, 1] <= zl[3]) & (lay.area >= 3)
                        inz[0] = False
                        hh = hints[k][3] - hints[k][1]
                        info["blobs"] = [[int(v) for v in lay.boxes[b]] + [int(lay.area[b]), int(lay.role[b]), int(lay.line[b])] for b in np.flatnonzero(inz)][:12]
                        info["hint"] = [round(float(v)) for v in hints[k] + off]
                        info["hint_h"] = round(float(hh))
                        reason = "no letter" if inz.any() else "no ink in region crop"
                    break
            info["reason"] = reason
            why[reason] += 1
            rows.append(info)
            print(f"{p['doc']} p{p['page']} {reason:30s} az {z} finder {info['finder'][:2]} ours {info.get('ours')} {l['text'][:30]!r}", flush=True)
    print(dict(why))
    if a.out:
        json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
