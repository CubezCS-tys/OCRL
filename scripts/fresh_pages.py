"""Training pages from journals the structure model has never seen: choose documents, then draw labeller inputs.

Runs where the documents are (PDF + Azure JSON per document folder). Excluded: every journal (first four digits)
of bulk_inputs, desk_inputs, the Desk database and the check set, and the dev, locked, keyed and check documents
(an exclusion file made on the laptop: {"journals": [...], "docs": [...]}). Each page of the first 40 of a document
gets the census kind its features are nearest (scripts/census.py page_row, out/census/kind_centres.npz), as
scripts/check_pages.py does. Documents are taken round-robin over journals, so as many journals as possible come
in; two pages a document, drawn with extra weight on the kinds the fresh test set shows weakest.
Inputs are drawn as the check pages were: crop box at 150 dpi, scaled to 1150 wide + 70 px strip, lines boxed and
numbered, lines.json in thousandths.

    python fresh_pages.py select --src /home/yassine/data/reader_s3 --exclude exclude.json --docs 3000 --out fresh_pages.json
    python fresh_pages.py draw --src /home/yassine/data/reader_s3 --pages fresh_pages.json --dest fresh_inputs --workers 32
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WEAK = {"Title and cover pages": 3.0, "Article first page": 3.0, "Two columns": 2.5, "Mixed: tables, charts, shaded cells": 2.5,
        "Small text block": 2.5, "Tables of numbers": 2.0}
SEED = 11


def kinds_of(args):
    src, stem = args
    from census import page_row
    z = np.load(HERE / "kind_centres.npz"); lo, hi, mean, std, C = z["lo"], z["hi"], z["mean"], z["std"], z["C"]
    try:
        d = json.loads((Path(src) / stem / f"{stem}.json").read_text(encoding="utf-8")); a = d.get("analyzeResult", d)
        hand = [(s["offset"], s["offset"] + s["length"]) for st in a.get("styles", []) if st.get("isHandwritten") for s in st.get("spans", [])]
        par = Counter(p["boundingRegions"][0]["pageNumber"] for p in a.get("paragraphs", []) if p.get("boundingRegions"))
        out = []
        for pg in a["pages"][:40]:
            n = len(pg.get("lines", []))
            if not (5 <= n <= 320) or abs(pg.get("angle") or 0) > 20:
                continue
            r = page_row(pg, len(a["pages"]), hand, par.get(pg["pageNumber"], 0))
            if not r:
                continue
            v = np.array(r, np.float32); v[0] = np.log1p(v[0]); v[1] = np.log1p(v[1]); v = (np.clip(v, lo, hi) - mean) / std
            out.append((pg["pageNumber"], n, int(((C - v) ** 2).sum(1).argmin())))
        return stem, out
    except Exception:
        return stem, []


def select(a):
    ex = json.load(open(a.exclude)); exj, exd = set(ex["journals"]), set(ex["docs"])
    names = {int(k): v for k, v in json.load(open(HERE / "kind_names.json")).items()}
    src = Path(a.src)
    docs = sorted(p.name for p in src.iterdir() if p.name[:4] not in exj and p.name not in exd and (p / f"{p.name}.pdf").exists() and (p / f"{p.name}.json").exists())
    print(f"{len(docs)} eligible documents in {len({d[:4] for d in docs})} journals", flush=True)
    with Pool(a.workers) as pool:
        rows = dict(pool.imap_unordered(kinds_of, [(a.src, d) for d in docs], chunksize=4))
    rng = np.random.default_rng(SEED)
    by_j = defaultdict(list)
    for d in docs:
        if len(rows.get(d, [])) >= 1:
            by_j[d[:4]].append(d)
    js = sorted(by_j); rng.shuffle(js)
    for j in js: rng.shuffle(by_j[j])
    chosen = []
    while len(chosen) < a.docs and any(by_j[j] for j in js):
        for j in js:
            if by_j[j] and len(chosen) < a.docs:
                chosen.append(by_j[j].pop())
    pages = []
    for d in chosen:
        opts = rows[d]
        w = np.array([WEAK.get(names[k], 1.0) for _, _, k in opts]); w /= w.sum()
        for i in rng.choice(len(opts), min(2, len(opts)), replace=False, p=w):
            no, n, k = opts[i]
            pages.append({"doc": d, "journal": d[:4], "number": int(no), "lines": int(n), "kind": names[k], "page": f"{d}_p{no}"})
    avail = Counter(names[k] for d in chosen for _, _, k in rows[d])
    json.dump({"seed": SEED, "documents": len(chosen), "journals": len({d[:4] for d in chosen}), "pages": pages,
               "kinds": dict(Counter(p["kind"] for p in pages).most_common()), "kinds_available_in_chosen_docs": dict(avail.most_common())},
              open(a.out, "w"), ensure_ascii=False, indent=0)
    print(f"{len(pages)} pages from {len(chosen)} documents in {len({d[:4] for d in chosen})} journals")
    for k, v in Counter(p["kind"] for p in pages).most_common():
        print(f"  {k:40s} {v:5d}   (pages of that kind in the chosen documents {avail[k]})")


def draw_one(args):
    src, dest, p = args
    import cv2
    cv2.setNumThreads(1)
    d = Path(dest) / p["page"]
    if (d / "lines.json").exists():
        return "have"
    try:
        az = json.loads((Path(src) / p["doc"] / f"{p['doc']}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        pg = next(q for q in az["pages"] if q["pageNumber"] == p["number"])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-cropbox", "-r", "150", "-gray", "-png", "-f", str(p["number"]), "-l", str(p["number"]),
                            str(Path(src) / p["doc"] / f"{p['doc']}.pdf"), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=300)
            img = cv2.imread(str(next(Path(tmp).glob("p-*.png"))))
        if abs(img.shape[0] / img.shape[1] - pg["height"] / pg["width"]) > 0.02 * pg["height"] / pg["width"]:
            return "wrong shape"
        s = 1150 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 70, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        sx, sy = 1150 / pg["width"], img.shape[0] / pg["height"]
        lines = []
        for i, l in enumerate(pg["lines"]):
            q = l["polygon"]; x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
            cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
            cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
            lines.append({"id": f"{p['number']}:{i}", "box": [round(1000 * x0 / pg["width"]), round(1000 * y0 / pg["height"]),
                          round(1000 * x1 / pg["width"]), round(1000 * y1 / pg["height"])], "text": l["content"]})
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"page_{p['number']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        (d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return "ok"
    except Exception as e:
        return f"failed: {type(e).__name__}"


def draw(a):
    pages = json.load(open(a.pages))["pages"]
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, maxtasksperchild=50) as pool:
        st = Counter(pool.imap_unordered(draw_one, [(a.src, a.dest, p) for p in pages], chunksize=4))
    print(dict(st))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "draw"])
    ap.add_argument("--src", default="/home/yassine/data/reader_s3"); ap.add_argument("--exclude", default="exclude.json")
    ap.add_argument("--docs", type=int, default=3000); ap.add_argument("--out", default="fresh_pages.json")
    ap.add_argument("--pages", default="fresh_pages.json"); ap.add_argument("--dest", default="fresh_inputs"); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    select(a) if a.cmd == "select" else draw(a)
