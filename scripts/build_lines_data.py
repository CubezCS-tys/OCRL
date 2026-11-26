"""Training pages for the line finder: each page scaled to a fixed width, with Azure's line cores as the target.

    python scripts/build_lines_data.py --split train --workers 48
"""
import argparse, json, subprocess, tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

S3 = Path("/home/yassine/data/mandumah_s3")
OUT = Path("data/linefinder")


def one(args):
    stem, split, max_pages = args
    import cv2
    cv2.setNumThreads(1)
    from ocrl.blobs.linefinder import core_mask, scale_page
    pdf, js = S3 / stem / f"{stem}.pdf", S3 / stem / f"{stem}.json"
    if not (pdf.exists() and js.exists()):
        return 0, 0
    try:
        az = json.loads(js.read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
        n = min(len(az["pages"]), max_pages); pages = lines = 0
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-r", "150", "-gray", "-png", "-l", str(n), str(pdf), f"{tmp}/p"], check=True, stderr=subprocess.DEVNULL, timeout=600)
            for pi, f in enumerate(sorted(Path(tmp).glob("p-*.png"))[:n]):
                pg = az["pages"][pi]
                if not pg.get("lines"):
                    continue
                gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                small, _ = scale_page(gray)
                sx, sy = small.shape[1] / pg["width"], small.shape[0] / pg["height"]
                boxes = np.array([[min(l["polygon"][0::2]) * sx, min(l["polygon"][1::2]) * sy, max(l["polygon"][0::2]) * sx, max(l["polygon"][1::2]) * sy] for l in pg["lines"]])
                cv2.imwrite(str(OUT / split / f"{stem}_p{pi + 1:03d}.png"), small)
                cv2.imwrite(str(OUT / split / f"{stem}_p{pi + 1:03d}.mask.png"), core_mask(small.shape, boxes) * 255)
                np.save(OUT / split / f"{stem}_p{pi + 1:03d}.boxes.npy", boxes.astype(np.float32))
                pages += 1; lines += len(boxes)
        return pages, lines
    except Exception as e:
        return 0, 0


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-pages", type=int, default=14)
a = ap.parse_args()
(OUT / a.split).mkdir(parents=True, exist_ok=True)
ids = json.load(open(f"data/realink/{a.split}_ids.json"))
with Pool(a.workers) as pool:
    res = pool.map(one, [(s, a.split, a.max_pages) for s in ids], chunksize=2)
print(a.split, "documents", len(ids), "pages", sum(r[0] for r in res), "lines", sum(r[1] for r in res))
