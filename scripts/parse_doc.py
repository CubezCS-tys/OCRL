"""Parse a scanned document locally, start to finish: page images in, a master file and a viewer out.

    uv run python scripts/parse_doc.py --pdf data/real/0013/0013-002-001-005.pdf --out out/parsed/0013-002-001-005
"""
import argparse, json, subprocess, time
from pathlib import Path

import cv2
import torch

from ocrl.doc.parse import parse, to_master
from ocrl.doc.viewer import write_viewer
from ocrl.pipeline import OCR

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="out/recognizer_v3.pt")
ap.add_argument("--pages", default="", help="first-last, e.g. 1-3")
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--typer", default="", help="a structure model checkpoint; without it the hand-written rules type the blocks")
args = ap.parse_args()
torch.set_num_threads(12)
out = Path(args.out); (out / "pages").mkdir(parents=True, exist_ok=True)
doc_id = Path(args.pdf).stem
rng = ["-f", args.pages.split("-")[0], "-l", args.pages.split("-")[-1]] if args.pages else []
subprocess.run(["pdftoppm", "-r", str(args.dpi), "-gray", "-png", *rng, args.pdf, str(out / "pages" / "p")], check=True)
files = sorted((out / "pages").glob("p-*.png"))
images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
t = time.time()
typer = None
if args.typer:
    from ocrl.structure.tag import Typer
    typer = Typer(args.typer)
infos, blocks = parse(OCR(args.ckpt), images, int(args.pages.split("-")[0]) if args.pages else 1, typer)
how = f"OCRL structure model ({Path(args.typer).stem}), trained on the Desk database, local. Not checked by a person." if typer else ""
master = to_master(doc_id, infos, blocks, args.dpi, Path(args.ckpt).stem, how)
secs = time.time() - t
(out / f"{doc_id}.local.master.json").write_text(json.dumps(master, ensure_ascii=False, indent=1), encoding="utf-8")
write_viewer(master, images, out / f"{doc_id}.html", args.dpi, secs)
from collections import Counter
print(f"{len(images)} pages in {secs:.0f} s ({secs / len(images):.1f} s a page) | {len(master['blocks'])} blocks:", dict(Counter(b["type"] for b in master["blocks"])))
print("wrote", out / f"{doc_id}.html")
