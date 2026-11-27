"""Write NAME.cleanup.png (lf2_targets.clean_small) for every page of a split.

    python scripts/lf2_cleanall.py --data data --split train --workers 96 [--only-with-clean]
"""
import argparse, glob
from multiprocessing import Pool
from pathlib import Path

import cv2


def one(f):
    cv2.setNumThreads(1)
    from lf2_targets import clean_small
    out = f[:-4] + ".cleanup.png"
    if Path(out).exists():
        return 0
    try:
        cv2.imwrite(out, clean_small(cv2.imread(f, 0)))
        return 1
    except Exception:
        return 0


ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--split"); ap.add_argument("--workers", type=int, default=96); ap.add_argument("--only-with-clean", action="store_true")
a = ap.parse_args()
fs = sorted(f for f in glob.glob(f"{a.data}/{a.split}/*.png") if f.count(".") == 1)
import random; random.Random(3).shuffle(fs)  # spread over journals, so a part-finished run is a fair sample
if a.only_with_clean:
    fs = [f for f in fs if Path(f[:-4] + ".clean.png").exists()]
with Pool(a.workers) as pool:
    n = 0
    for k, r in enumerate(pool.imap_unordered(one, fs, chunksize=8), 1):
        n += r
        if k % 5000 == 0:
            print(k, flush=True)
print("cleaned", n, "of", len(fs))
