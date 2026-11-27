"""Line images with Azure's text, for training a line reader the way the owner's own was trained.

Every Azure line Azure itself is confident about becomes one training line: cut from the page with a
little margin, scaled to 48 pixels high. Lines Azure doubts are left out.

    python scripts/build_reader_lines.py --split train --workers 64
"""
import argparse, json, subprocess, tempfile, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SRC = Path("/home/yassine/data/reader_s3")
OUT = Path("data/reader")
H = 48


def one(args):
    stem, max_pages, min_conf = args
    import cv2
    cv2.setNumThreads(1)
    from ocrl.recog.realink import fold
    from ocrl.text import TATWEEL
    pdf, js = SRC / stem / f"{stem}.pdf", SRC / stem / f"{stem}.json"
    stats, crops, labels, prov = Counter(), [], [], []
    if not (pdf.exists() and js.exists()):
        return Counter(missing=1), crops, labels, prov
    try:
        az = json.loads(js.read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        n = min(len(az["pages"]), max_pages)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-cropbox", "-r", "200", "-gray", "-png", "-l", str(n), str(pdf), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=300)
            for pi, f in enumerate(sorted(Path(tmp).glob("p-*.png"))[:n]):
                pg = az["pages"][pi]
                if abs(pg.get("angle") or 0) > 20 or not pg.get("lines") or len(pg["lines"]) > 400:
                    stats["page_skipped"] += 1
                    continue
                gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if gray is None or gray.size > 30_000_000:
                    stats["page_skipped"] += 1
                    continue
                sx, sy = gray.shape[1] / pg["width"], gray.shape[0] / pg["height"]
                words = sorted((w["span"]["offset"], w["confidence"]) for w in pg.get("words", []))
                offs = np.array([w[0] for w in words]); conf = np.array([w[1] for w in words])
                for li, ln in enumerate(pg["lines"]):
                    stats["lines"] += 1
                    sp = ln["spans"][0]
                    c = conf[(offs >= sp["offset"]) & (offs < sp["offset"] + sp["length"])]
                    if len(c) == 0 or c.min() < min_conf:
                        stats["doubted_by_azure"] += 1
                        continue
                    text = " ".join(fold(w) for w in ln["content"].replace(TATWEEL, "").split())
                    q = ln["polygon"]; x0, y0, x1, y1 = min(q[0::2]) * sx, min(q[1::2]) * sy, max(q[0::2]) * sx, max(q[1::2]) * sy
                    h = y1 - y0
                    if not text or h < 8 or x1 - x0 < 10 or h > 1.8 * (x1 - x0):
                        stats["odd_shape"] += 1
                        continue
                    a, b = max(int(y0 - 0.18 * h), 0), min(int(y1 + 0.18 * h) + 1, gray.shape[0])
                    l, r = max(int(x0 - 0.3 * h), 0), min(int(x1 + 0.3 * h) + 1, gray.shape[1])
                    cut = gray[a:b, l:r]
                    w = int(np.clip(round(cut.shape[1] * H / cut.shape[0]), 8, 2200))
                    crops.append(cv2.resize(cut, (w, H), interpolation=cv2.INTER_AREA)); labels.append(text); prov.append((stem, pi + 1, li))
                    stats["kept"] += 1
                stats["pages"] += 1
        stats["documents"] += 1
    except Exception:
        stats["failed"] += 1
    return stats, crops, labels, prov


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-pages", type=int, default=16); ap.add_argument("--min-conf", type=float, default=0.85)
    a = ap.parse_args()
    ids = json.load(open(OUT / f"{a.split}_ids.json"))
    out = OUT / f"lines_{a.split}"; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.pt"):
        f.unlink()
    total, C, L, P, shard, t0 = Counter(), [], [], [], 0, time.time()

    def flush():
        global C, L, P, shard
        if not C:
            return
        offsets = np.concatenate([[0], np.cumsum([c.shape[1] for c in C])]).astype(np.int64)
        torch.save({"strip": torch.from_numpy(np.concatenate(C, axis=1)), "offsets": torch.from_numpy(offsets), "labels": L, "prov": P}, out / f"shard_{shard:03d}.pt")
        shard += 1; C, L, P = [], [], []

    def guard():  # one odd document once made a worker grow to 100 GB: cap each worker (and its pdftoppm) at 16 GB
        import resource
        import cv2  # noqa: F401  (load the libraries first: the cap is on growth, not on what they reserve)
        vm = int(next(l for l in open("/proc/self/status") if l.startswith("VmSize")).split()[1]) << 10
        resource.setrlimit(resource.RLIMIT_AS, (vm + (16 << 30), vm + (16 << 30)))

    with Pool(a.workers, initializer=guard, maxtasksperchild=20) as pool:
        for k, (st, c, l, p) in enumerate(pool.imap_unordered(one, [(s, a.max_pages, a.min_conf) for s in ids], chunksize=1), 1):
            total += st; C += c; L += l; P += p
            if len(C) >= 40000:
                flush()
            if k % 250 == 0:
                print(f"  {k}/{len(ids)} documents, {total['kept']:,} lines kept, {time.time() - t0:.0f} s", flush=True)
    flush()
    print(f"{a.split}: " + ", ".join(f"{k} {v:,}" for k, v in total.most_common()))
