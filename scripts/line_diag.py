"""Why does our line finding lose some of Azure's lines? Sorts every Azure line into a case and draws the page.

Our lines and Azure's are linked when one's centre lies inside the middle band of the other. Each linked group is then one case:
  one-to-one        one line of ours for one of Azure's
  ours joins a row  one of ours holds several of Azure's that sit side by side (table cells, two columns, a wide gap)
  ours joins lines  one of ours holds several of Azure's stacked on top of each other
  ours splits       several of ours for one of Azure's
  tangled           several of ours and several of Azure's in one group
  missing           no line of ours at all

    uv run python scripts/line_diag.py --docs 0499-026-119-022 --draw out/diag [--finder out/linefinder.pt]
"""
import argparse, json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import line_recall as LR
from ocrl.blobs.clean import clean_page

CASES = ["one-to-one", "ours joins a row", "ours joins lines", "ours splits", "tangled", "missing"]
COL = {"one-to-one": (72, 160, 60), "ours joins a row": (0, 140, 255), "ours joins lines": (200, 60, 200), "ours splits": (200, 160, 0), "tangled": (40, 40, 220), "missing": (0, 0, 255)}


def cases(ours, az):
    """-> one case per Azure line, and the group id of every line of ours and of Azure's."""
    n, m = len(ours), len(az)
    par = list(range(n + m))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, o in enumerate(ours):
        ocx, ocy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
        for j, z in enumerate(az):
            zcx, zcy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
            # the middle half of a box's height: tightly set lines overlap at their edges (descenders
            # reaching into the next line's ascenders), and that must not count as a link
            oq, zq = 0.25 * (o[3] - o[1]), 0.25 * (z[3] - z[1])
            if (o[0] - 4 <= zcx <= o[2] + 4 and o[1] + oq <= zcy <= o[3] - oq) or (z[0] <= ocx <= z[2] and z[1] + zq <= ocy <= z[3] - zq):
                par[find(i)] = find(n + j)
    groups = {}
    for k in range(n + m):
        groups.setdefault(find(k), ([], []))[k >= n].append(k if k < n else k - n)
    out = [None] * m
    for os_, zs in groups.values():
        if not zs:
            continue
        if not os_: c = "missing"
        elif len(os_) == 1 and len(zs) == 1: c = "one-to-one"
        elif len(os_) == 1:
            # side by side if their vertical extents overlap; stacked otherwise
            zz = sorted((az[j] for j in zs), key=lambda z: z[1])
            stacked = any(b[1] > a[1] + 0.6 * (a[3] - a[1]) for a, b in zip(zz, zz[1:]))
            side = any(min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1]) for i, a in enumerate(zz) for b in zz[i + 1 :])
            c = "tangled" if stacked and side else "ours joins lines" if stacked else "ours joins a row"
        elif len(zs) == 1: c = "ours splits"
        else: c = "tangled"
        for j in zs: out[j] = c
    return out


_finder_path = None


def run_doc(job):
    """One document -> (name, cases per line, characters per case, order counts). Runs in its own process."""
    global _finder_path
    doc, a = job
    if a["finder"] and _finder_path != a["finder"]:
        from ocrl.blobs.linefinder import LineFinder
        LR.finder = LineFinder(a["finder"]); _finder_path = a["finder"]
    azj = json.loads((Path(a["samples"]) / doc / f"{doc}.json").read_text(encoding="utf-8")); azj = azj.get("analyzeResult", azj)
    n = min(len(azj["pages"]), a["max_pages"])
    per, chars, order = Counter(), Counter(), Counter()
    for pi, f in enumerate(LR.render(a["samples"], doc, n, a["cache"])):
        pg = azj["pages"][pi]
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
        cleaned = clean_page(gray); gray = cleaned.gray  # draw on what the line finding actually sees
        ours = LR.our_lines(gray, None if a["no_rule_cuts"] else cleaned.rules)
        az = [(min(l["polygon"][0::2]) * sx, min(l["polygon"][1::2]) * sy, max(l["polygon"][0::2]) * sx, max(l["polygon"][1::2]) * sy) for l in pg.get("lines", [])]
        cs = cases(ours, az)
        # Reading order: Azure lists its lines in reading order. Take the lines we matched one-to-one,
        # in Azure's order, and count how often our own order goes forward from one to the next.
        pos = []
        for z, c in zip(az, cs):
            if c == "one-to-one":
                zcx, zcy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
                k = next((i for i, o in enumerate(ours) if o[0] - 4 <= zcx <= o[2] + 4 and o[1] <= zcy <= o[3]), None)
                if k is not None: pos.append(k)
        order["steps"] += max(len(pos) - 1, 0); order["forward"] += sum(y > x for x, y in zip(pos, pos[1:]))
        for c, l in zip(cs, pg.get("lines", [])):
            per[c] += 1; chars[c] += len(l["content"])
        if a["draw"]:
            Path(a["draw"]).mkdir(parents=True, exist_ok=True)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); lay = img.copy()
            for o in ours:
                cv2.rectangle(lay, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (255, 200, 120), -1)
            img = cv2.addWeighted(lay, 0.35, img, 0.65, 0)
            for o in ours:
                cv2.rectangle(img, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (160, 90, 0), 2)
            for z, c in zip(az, cs):
                if c != "one-to-one":
                    cv2.rectangle(img, (int(z[0]), int(z[1])), (int(z[2]), int(z[3])), COL[c], 4)
            s = 1400 / img.shape[1]
            cv2.imwrite(str(Path(a["draw"]) / f"{doc}_p{pi + 1:02d}{'_finder' if a['finder'] else ''}.jpg"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return doc, per, chars, order


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/home/yassine/inkscript_v2/data/samples"); ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--finder", default=""); ap.add_argument("--max-pages", type=int, default=12); ap.add_argument("--cache", default="out/line_recall_pages")
    ap.add_argument("--draw", default="", help="folder for page pictures"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-rule-cuts", action="store_true", help="do not cut the page into regions at ruled lines")
    ap.add_argument("--workers", type=int, default=1)
    a = vars(ap.parse_args())
    jobs = [(d, a) for d in a["docs"]]
    total, chars, order = Counter(), Counter(), Counter()
    with mp.get_context("spawn").Pool(a["workers"]) as pool:
        for doc, per, ch, od in (pool.imap(run_doc, jobs) if a["workers"] > 1 else map(run_doc, jobs)):
            total += per; chars += ch; order += od
            if not a["quiet"]:
                t = max(sum(per.values()), 1)
                print(f"{doc}: {sum(per.values()):4d} Azure lines | order {100 * od['forward'] / max(od['steps'], 1):3.0f}% ({od['steps'] - od['forward']} steps back) | " + " | ".join(f"{c} {100 * per[c] / t:.0f}%" for c in CASES if per[c]), flush=True)
    t, tc = max(sum(total.values()), 1), max(sum(chars.values()), 1)
    print(f"reading order: {100 * order['forward'] / max(order['steps'], 1):.1f}% of {order['steps']} steps between matched lines go forward in our order too")
    print(f"ALL: {sum(total.values())} Azure lines | " + " | ".join(f"{c} {100 * total[c] / t:.1f}% of lines, {100 * chars[c] / tc:.1f}% of characters" for c in CASES if total[c]))
