"""Download documents from a list of short-lived links (id <TAB> extension <TAB> url), many at a time.

The links are made on the owner's machine; this runs wherever the files are wanted. No credentials are needed here.
    python scripts/fetch_links.py links.tsv /home/yassine/data/reader_s3 --workers 48
"""
import argparse, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("links"); ap.add_argument("dest"); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
rows = [l.rstrip("\n").split("\t") for l in open(a.links) if l.strip()]


def get(row):
    doc, ext, url = row
    out = Path(a.dest) / doc / f"{doc}.{ext}"
    if out.exists() and out.stat().st_size > 0:
        return 0, True
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as f:
                n = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk); n += len(chunk)
            return n, True
        except Exception:
            out.unlink(missing_ok=True)
    return 0, False


done = size = failed = 0
with ThreadPoolExecutor(a.workers) as pool:
    for n, ok in pool.map(get, rows):
        done += 1; size += n; failed += not ok
        if done % 1000 == 0:
            print(f"{done}/{len(rows)} files, {size / 1e9:.1f} GB, {failed} failed", flush=True)
print(f"finished: {done} files, {size / 1e9:.1f} GB, {failed} failed")
