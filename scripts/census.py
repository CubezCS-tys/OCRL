"""A census of page geometry from Azure's boxes alone: no images, no labels, no reading.

Every page becomes a short row of numbers (how many lines, how tall, how many columns, how table-like,
which script, ...). The rows are then grouped into kinds of page, and a few pages from each kind are
drawn small so the kinds can be looked at.

    python scripts/census.py features --samples /home/yassine/data/reader_s3 --ids data/reader/train_ids.json data/reader/test_ids.json --workers 48
    python scripts/census.py kinds --k 14 --show 10 --dev 5
"""
import argparse, json, re, subprocess, tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("out/census")
NAMES = ["lines", "words_per_line", "line_height", "height_spread", "ink_share", "top_margin", "bottom_margin", "side_margin",
         "columns", "in_rows", "short_lines", "centred", "latin", "digits", "marked", "low_conf", "handwritten", "lines_per_par",
         "first_page", "tall_page", "big_type_lines", "ragged"]
_MARK = re.compile(r"^\s*(?:[-–—−ـ•*·▪■●◦]+\s*\S|\(?\s?[0-9٠-٩]{1,3}\s?[-–—ـ.)]\s*\S)")


def page_row(pg, n_pages, hand_spans, par_count):
    W, H = pg["width"], pg["height"]
    ls = pg.get("lines", [])
    if len(ls) < 3 or not W or not H:
        return None
    b = np.array([[min(l["polygon"][0::2]), min(l["polygon"][1::2]), max(l["polygon"][0::2]), max(l["polygon"][1::2])] for l in ls])
    w, h = b[:, 2] - b[:, 0], np.maximum(b[:, 3] - b[:, 1], 1e-3)
    hm = float(np.median(h)); x0, x1 = float(b[:, 0].min()), float(b[:, 2].max()); tw = max(x1 - x0, 1e-3)
    text = [l["content"] for l in ls]; chars = "".join(text); n = max(len(chars), 1)
    words = pg.get("words", []); conf = np.array([w_["confidence"] for w_ in words]) if words else np.ones(1)
    # columns: clear vertical strips between whole lines, at least four lines on each side
    cols, edges = 1, np.zeros(int(W * 20) + 2, int)
    for a, c in zip((b[:, 0] * 20).astype(int), (b[:, 2] * 20).astype(int)):
        edges[max(a, 0) : max(c, a + 1)] += 1
    span = edges[int(x0 * 20) : int(x1 * 20)]
    free = np.flatnonzero(span == 0)
    if len(free):
        for run in np.split(free, np.flatnonzero(np.diff(free) > 1) + 1):
            mid = x0 + (run[0] + run[-1]) / 40
            if len(run) / 20 >= 0.6 * hm and (b[:, 2] <= mid).sum() >= 4 and (b[:, 0] >= mid).sum() >= 4:
                cols += 1
    cy = (b[:, 1] + b[:, 3]) / 2; order = np.argsort(cy)
    level = np.zeros(len(ls), bool)  # a line with another line level with it
    for i, j in zip(order, order[1:]):
        if abs(cy[i] - cy[j]) < 0.4 * hm and (b[i, 2] < b[j, 0] or b[j, 2] < b[i, 0]):
            level[i] = level[j] = True
    colw = tw / cols
    centre = np.abs((b[:, 0] + b[:, 2]) / 2 - (x0 + x1) / 2) < 0.04 * tw
    s0 = pg["spans"][0]["offset"] if pg.get("spans") else 0; s1 = s0 + (pg["spans"][0]["length"] if pg.get("spans") else 0)
    hand = sum(max(0, min(e, s1) - max(a, s0)) for a, e in hand_spans) / max(s1 - s0, 1)
    return [len(ls), len(words) / len(ls), hm / H, float(np.percentile(h, 90)) / hm, float((w * h).sum()) / (W * H), float(b[:, 1].min()) / H,
            1 - float(b[:, 3].max()) / H, min(x0, W - x1) / W, cols, float((level & (w < 0.35 * tw)).mean()), float((w < 0.5 * colw).mean()),
            float((centre & (w < 0.7 * tw)).mean()), sum("a" <= c.lower() <= "z" for c in chars) / n, sum(c.isdigit() for c in chars) / n,
            float(np.mean([bool(_MARK.match(t)) for t in text])), float((conf < 0.8).mean()), hand, len(ls) / max(par_count, 1),
            float(pg["pageNumber"] == 1), H / W, float((h > 1.6 * hm).mean()), float(np.std(b[:, 0]) / tw)]


def doc_rows(args):
    samples, stem, max_pages = args
    try:
        d = json.loads((Path(samples) / stem / f"{stem}.json").read_text(encoding="utf-8")); a = d.get("analyzeResult", d)
        hand = [(s["offset"], s["offset"] + s["length"]) for st in a.get("styles", []) if st.get("isHandwritten") for s in st.get("spans", [])]
        par = Counter(p["boundingRegions"][0]["pageNumber"] for p in a.get("paragraphs", []) if p.get("boundingRegions"))
        out = []
        for pg in a["pages"][:max_pages]:
            r = page_row(pg, len(a["pages"]), hand, par.get(pg["pageNumber"], 0))
            if r:
                out.append((stem, pg["pageNumber"], samples, r))
        return out
    except Exception:
        return []


def kmeans(X, k, rng, rounds=40):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(rounds):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab, C


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["features", "kinds"])
    ap.add_argument("--samples", nargs="+", default=["/home/yassine/data/reader_s3"]); ap.add_argument("--ids", nargs="+", default=[])
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--k", type=int, default=14); ap.add_argument("--show", type=int, default=10); ap.add_argument("--dev", type=int, default=5)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "features":
        jobs = []
        for samples in a.samples:
            have = {p.name for p in Path(samples).iterdir()}
            want = [i for f in a.ids for i in json.load(open(f))] or sorted(have)
            jobs += [(samples, s, a.max_pages) for s in want if s in have]
        with Pool(a.workers) as pool:
            rows = [r for rs in pool.imap_unordered(doc_rows, jobs, chunksize=4) for r in rs]
        np.savez_compressed(OUT / "pages.npz", stem=np.array([r[0] for r in rows]), page=np.array([r[1] for r in rows]),
                            samples=np.array([r[2] for r in rows]), X=np.array([r[3] for r in rows], np.float32), names=np.array(NAMES))
        print(f"{len(rows):,} pages from {len({r[0] for r in rows}):,} documents and {len({r[0][:4] for r in rows}):,} journals")
    else:
        d = np.load(OUT / "pages.npz"); X, stem, page, samples = d["X"], d["stem"], d["page"], d["samples"]
        Z = X.copy(); Z[:, 0] = np.log1p(Z[:, 0]); Z[:, 1] = np.log1p(Z[:, 1])
        lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
        Z = (np.clip(Z, lo, hi) - Z.mean(0)) / (Z.std(0) + 1e-6)
        rng = np.random.default_rng(0)
        best = min((kmeans(Z, a.k, rng) for _ in range(4)), key=lambda r: ((Z - r[1][r[0]]) ** 2).sum())
        lab, C = best
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); xy = (Z - Z.mean(0)) @ Vt[:2].T
        kinds, picks = [], []
        for j in np.argsort(-np.bincount(lab, minlength=a.k)):
            idx = np.flatnonzero(lab == j)
            if len(idx) == 0:
                continue
            dist = ((Z[idx] - C[j]) ** 2).sum(1)
            # examples: the most typical pages, one per journal, so a kind is not just one journal's house style
            seen, ex = set(), []
            for i in idx[np.argsort(dist)]:
                if stem[i][:4] not in seen:
                    seen.add(stem[i][:4]); ex.append(int(i))
                if len(ex) == a.show:
                    break
            top = np.argsort(-np.abs(C[j]))[:5]
            kinds.append({"kind": int(j), "pages": int(len(idx)), "share": float(len(idx) / len(lab)), "journals": int(len({s[:4] for s in stem[idx]})),
                          "traits": [[NAMES[t], float(C[j][t]), float(np.median(X[idx, t])), float(np.median(X[:, t]))] for t in top],
                          "medians": {n: float(np.median(X[idx, t])) for t, n in enumerate(NAMES)},
                          "examples": [{"stem": str(stem[i]), "page": int(page[i]), "img": f"k{j:02d}_{n}.jpg"} for n, i in enumerate(ex)]})
            picks += [(str(samples[i]), str(stem[i]), int(page[i]), f"k{j:02d}_{n}.jpg", 55, 46) for n, i in enumerate(ex)]
            picks += [(str(samples[i]), str(stem[i]), int(page[i]), f"dev_k{j:02d}_{n}.png", 150, None) for n, i in enumerate(ex[: a.dev])]
        (OUT / "img").mkdir(exist_ok=True); (OUT / "dev").mkdir(exist_ok=True)
        import cv2
        dev = []
        for smp, st, pg, name, dpi, q in picks:
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["pdftoppm", "-cropbox", "-r", str(dpi), "-gray", "-png", "-f", str(pg), "-l", str(pg), f"{smp}/{st}/{st}.pdf", f"{tmp}/p"], stderr=subprocess.DEVNULL, timeout=120)
                f = next(iter(Path(tmp).glob("p-*.png")), None)
                if f is None:
                    continue
                g = cv2.imread(str(f), 0)
                if q:
                    cv2.imwrite(str(OUT / "img" / name), g, [cv2.IMWRITE_JPEG_QUALITY, q])
                else:
                    cv2.imwrite(str(OUT / "dev" / name), g)
                    azd = json.loads((Path(smp) / st / f"{st}.json").read_text(encoding="utf-8")); az = azd.get("analyzeResult", azd); p = az["pages"][pg - 1]
                    first = {q_["spans"][0]["offset"] for q_ in az.get("paragraphs", [])}
                    dev.append({"stem": st, "page": pg, "img": name, "width": p["width"], "height": p["height"],
                                "lines": [{"box": [min(l["polygon"][0::2]), min(l["polygon"][1::2]), max(l["polygon"][0::2]), max(l["polygon"][1::2])], "text": l["content"],
                                           "par_start": l["spans"][0]["offset"] in first} for l in p["lines"]]})
        sub = rng.choice(len(lab), min(6000, len(lab)), replace=False)
        json.dump({"names": NAMES, "pages": int(len(lab)), "documents": int(len(set(stem))), "journals": int(len({s[:4] for s in stem})), "kinds": kinds,
                   "map": [[round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2), int(lab[i])] for i in sub]}, open(OUT / "kinds.json", "w"), ensure_ascii=False)
        json.dump(dev, open(OUT / "dev_pages.json", "w"), ensure_ascii=False)
        for k in kinds:
            print(f"kind {k['kind']:2d}: {k['pages']:6,} pages ({100 * k['share']:4.1f}%), {k['journals']:4d} journals | " + ", ".join(f"{n} {m:.2f} (all {al:.2f})" for n, _, m, al in k["traits"][:4]))
        print("development pages drawn:", len(dev))
