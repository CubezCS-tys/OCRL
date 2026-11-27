"""Line-finder data at scale (second round): pick documents spread over journals, draw up to 12 pages of each
(spread through the document) by the crop box, and save per page the scaled image, Azure's line boxes and word
boxes (so targets can be rebuilt without drawing again). Masks are made by lf2_train.py from these.

    python scripts/lf2_build.py select --pool reader_ids.txt --exclude exclude.json --train 5000 --val 300 --out ids.json
    python scripts/lf2_build.py build --ids ids.json --split train --s3 /home/yassine/data/reader_s3 --out data --workers 96

Exclusions: documents in exclude.json["docs"]; journals in exclude.json["journals"] (the fresh test set) and the
journals of the 50 original held-out documents (so those stay unseen journals for the new finder, as they are for the
current one). Validation journals are also kept clear of the current finder's 350 training journals.
"""
import argparse, json, subprocess, tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocrl.blobs.linefinder import scale_page


def table_score(boxes):
    """Share of a page's Azure lines that sit in a row with two or more others (cells of a table row)."""
    if len(boxes) < 6:
        return 0.0
    b = np.asarray(boxes, np.float32)
    h = b[:, 3] - b[:, 1]
    lo, hi = np.maximum(b[:, None, 1], b[None, :, 1]), np.minimum(b[:, None, 3], b[None, :, 3])
    ov = (hi - lo) > 0.5 * np.minimum(h[:, None], h[None, :])
    apart = (b[:, None, 2] < b[None, :, 0]) | (b[None, :, 2] < b[:, None, 0])
    k = (ov & apart).sum(1)
    return float((k >= 2).mean())


def select(a):
    pool = [l.strip() for l in open(a.pool) if l.strip()[:1].isdigit()]
    ex = json.load(open(a.exclude))
    bad_doc = set(ex["docs"]) | set(ex["heldout50"])
    bad_j = set(ex["journals"]) | {d[:4] for d in ex["heldout50"]}
    old_train_j = {d[:4] for d in ex["train350"]}
    ok = sorted(d for d in pool if d not in bad_doc and d[:4] not in bad_j)
    by_j = defaultdict(list)
    for d in ok:
        by_j[d[:4]].append(d)
    rng = np.random.default_rng(11)
    js = sorted(by_j)
    rng.shuffle(js)
    # validation: whole journals, never in the current finder's training either, one or two documents each
    val, val_j = [], []
    for j in js:
        if len(val) >= a.val:
            break
        if j in old_train_j:
            continue
        docs = list(by_j[j]); rng.shuffle(docs)
        val += docs[:2]; val_j.append(j)
    train_js = [j for j in js if j not in set(val_j)]
    for j in train_js:
        rng.shuffle(by_j[j])
    train, r = [], 0
    while len(train) < a.train:  # round robin over journals: every journal before any journal's second document
        added = False
        for j in train_js:
            if r < len(by_j[j]) and len(train) < a.train:
                train.append(by_j[j][r]); added = True
        r += 1
        if not added:
            break
    out = {"train": train, "val": val, "val_journals": val_j}
    json.dump(out, open(a.out, "w"), indent=0)
    print(f"pool {len(pool)}, allowed {len(ok)} in {len(by_j)} journals | train {len(train)} docs in {len({d[:4] for d in train})} journals"
          f" | val {len(val)} docs in {len(val_j)} journals")


def pick_pages(n, k):
    if n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def one(args):
    stem, s3, dest, max_pages = args
    import cv2
    cv2.setNumThreads(1)
    pdf, js = Path(s3) / stem / f"{stem}.pdf", Path(s3) / stem / f"{stem}.json"
    st = {"docs": 1, "pages": 0, "lines": 0, "skip_aspect": 0, "skip_nolines": 0, "fail": 0}
    if not (pdf.exists() and js.exists()):
        st["fail"] = 1
        return st, []
    meta = []
    try:
        az = json.loads(js.read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        pages = az["pages"]
        with tempfile.TemporaryDirectory() as tmp:
            for pi in pick_pages(len(pages), max_pages):
                pg = pages[pi]
                if not pg.get("lines"):
                    st["skip_nolines"] += 1
                    continue
                pn = pg.get("pageNumber", pi + 1)
                subprocess.run(["pdftoppm", "-cropbox", "-r", "150", "-gray", "-png", "-f", str(pn), "-l", str(pn), "-singlefile", str(pdf), f"{tmp}/p"],
                               check=True, stderr=subprocess.DEVNULL, timeout=300)
                gray = cv2.imread(f"{tmp}/p.png", cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    st["fail"] += 1
                    continue
                # the crop-box trap: the drawn page must have Azure's page shape, or every box is misplaced
                if abs(gray.shape[1] / gray.shape[0] - pg["width"] / pg["height"]) > 0.02 * pg["width"] / pg["height"]:
                    st["skip_aspect"] += 1
                    continue
                small, _ = scale_page(gray)
                sx, sy = small.shape[1] / pg["width"], small.shape[0] / pg["height"]
                L = pg["lines"]
                boxes = np.array([[min(l["polygon"][0::2]) * sx, min(l["polygon"][1::2]) * sy, max(l["polygon"][0::2]) * sx, max(l["polygon"][1::2]) * sy] for l in L], np.float32)
                # a line's slant: angle of its top edge (polygon x0 y0 x1 y1 ...); sideways lines are about +-90 degrees
                ang = np.array([np.degrees(np.arctan2(l["polygon"][3] - l["polygon"][1], l["polygon"][2] - l["polygon"][0])) for l in L], np.float32)
                # words to lines by span offsets
                starts = [(l["spans"][0]["offset"], l["spans"][-1]["offset"] + l["spans"][-1]["length"], i) for i, l in enumerate(L) if l.get("spans")]
                starts.sort()
                wb, wl = [], []
                so = np.array([s[0] for s in starts]) if starts else np.zeros(0)
                for w in pg.get("words", []):
                    o = w["span"]["offset"]
                    k = int(np.searchsorted(so, o, side="right")) - 1
                    if k < 0 or o >= starts[k][1]:
                        continue
                    p = w["polygon"]
                    wb.append([min(p[0::2]) * sx, min(p[1::2]) * sy, max(p[0::2]) * sx, max(p[1::2]) * sy]); wl.append(starts[k][2])
                name = f"{stem}_p{pi + 1:03d}"
                cv2.imwrite(str(Path(dest) / f"{name}.png"), small)
                np.savez(Path(dest) / f"{name}.npz", boxes=boxes, angle=ang, words=np.array(wb, np.float32).reshape(-1, 4), word_line=np.array(wl, np.int32))
                meta.append({"page": name, "lines": len(L), "table": round(table_score(boxes), 3), "H": small.shape[0]})
                st["pages"] += 1; st["lines"] += len(L)
    except Exception as e:
        st["fail"] += 1
    return st, meta


def build(a):
    ids = json.load(open(a.ids))[a.split] if a.split in ("train", "val") else json.load(open(a.ids))
    dest = Path(a.out) / a.split
    dest.mkdir(parents=True, exist_ok=True)
    tot, meta = defaultdict(int), []
    with Pool(a.workers) as pool:
        for k, (st, m) in enumerate(pool.imap_unordered(one, [(s, a.s3, str(dest), a.max_pages) for s in ids], chunksize=1), 1):
            for kk, v in st.items():
                tot[kk] += v
            meta += m
            if k % 250 == 0:
                print(k, dict(tot), flush=True)
    json.dump(meta, open(Path(a.out) / f"{a.split}_pages.json", "w"))
    print(a.split, dict(tot), "journals", len({m['page'][:4] for m in meta}), "table-heavy pages (>=30% of lines in rows of 3+)", sum(m["table"] >= 0.3 for m in meta))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("select"); s.add_argument("--pool"); s.add_argument("--exclude"); s.add_argument("--train", type=int, default=5000)
s.add_argument("--val", type=int, default=300); s.add_argument("--out")
b = sub.add_parser("build"); b.add_argument("--ids"); b.add_argument("--split"); b.add_argument("--s3"); b.add_argument("--out")
b.add_argument("--workers", type=int, default=96); b.add_argument("--max-pages", type=int, default=12)
a = ap.parse_args()
select(a) if a.cmd == "select" else build(a)
