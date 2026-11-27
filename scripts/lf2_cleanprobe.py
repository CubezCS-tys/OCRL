"""Does the finder see something different at work than in training? It is trained on raw pages but run on
clean_page()'s output (paper evened, ruled lines and pictures painted out). Render pages at 300 dpi, clean, scale;
save the cleaned page beside the raw one (NAME.clean.png), and score a finder on both.

    python scripts/lf2_cleanprobe.py make --data data --split val --s3 /home/yassine/data/reader_s3 --n 400 --workers 16
    python scripts/lf2_cleanprobe.py score --data data --split val --finder linefinder.pt [--only-cleaned]
"""
import argparse, json, subprocess, tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def make_one(args):
    name, data, split, s3, raw300 = args
    cv2.setNumThreads(1)
    from ocrl.blobs.clean import clean_page
    from ocrl.blobs.linefinder import scale_page
    doc, pn = name.rsplit("_p", 1)
    az = json.loads((Path(s3) / doc / f"{doc}.json").read_text(encoding="utf-8")); az = az.get("analyzeResult", az)
    p = az["pages"][int(pn) - 1].get("pageNumber", int(pn))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-f", str(p), "-l", str(p), "-singlefile", str(Path(s3) / doc / f"{doc}.pdf"), f"{tmp}/p"],
                       check=True, stderr=subprocess.DEVNULL, timeout=300)
        g = cv2.imread(f"{tmp}/p.png", 0)
    ref = cv2.imread(f"{data}/{split}/{name}.png", 0)
    c = scale_page(clean_page(g).gray)[0]
    c = cv2.resize(c, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA) if c.shape != ref.shape else c
    cv2.imwrite(f"{data}/{split}/{name}.clean.png", c)
    if raw300:
        r = scale_page(g)[0]
        cv2.imwrite(f"{data}/{split}/{name}.raw300.png", cv2.resize(r, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA) if r.shape != ref.shape else r)
    return name


def score(a):
    import torch
    from ocrl.blobs.linefinder import CORE, LineFinder
    torch.cuda.set_per_process_memory_fraction(0.3)
    meta = {m["page"]: m["table"] for m in json.load(open(f"{a.data}/{a.split}_pages.json"))}
    names = sorted(n for n in meta if Path(f"{a.data}/{a.split}/{n}.clean.png").exists())
    res = {}
    for fp in a.finder:
        lf = LineFinder(fp, "cuda")
        for kind in ("png", "raw300.png", "clean.png", "cleanup.png"):
            r = {"all": [0, 0], "table": [0, 0]}
            for n in names:
                f = f"{a.data}/{a.split}/{n}.{kind}"
                if not Path(f).exists():
                    continue
                img = cv2.imread(f, 0)
                boxes = np.load(f"{a.data}/{a.split}/{n}.npz")["boxes"]
                H, W = img.shape; ph, pw = (-H) % 8, (-W) % 8
                x = torch.from_numpy(np.pad(img, ((0, ph), (0, pw)), constant_values=255)).float()[None, None].cuda() / 255
                with torch.no_grad():
                    prob = torch.sigmoid(lf.net(x))[0, 0, :H, :W].cpu().numpy()
                _, lab = cv2.connectedComponents((prob > 0.5).astype(np.uint8), connectivity=8)
                used, found = set(), 0
                for x0, y0, x1, y1 in boxes:
                    h = y1 - y0
                    ids = np.unique(lab[int(y0 + CORE[0] * h) : max(int(y0 + CORE[1] * h), int(y0 + CORE[0] * h) + 1), int(x0) : max(int(x1), int(x0) + 1)])
                    ids = ids[ids > 0]
                    found += len(ids) == 1 and ids[0] not in used
                    used.update(ids.tolist())
                for k in ("all", "table") if meta[n] >= 0.3 else ("all",):
                    r[k][0] += found; r[k][1] += len(boxes)
            print(f"{fp} on {kind}: one-to-one {100 * r['all'][0] / max(r['all'][1], 1):.2f}% of {r['all'][1]} | table-heavy {100 * r['table'][0] / max(r['table'][1], 1):.2f}% of {r['table'][1]}", flush=True)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
m = sub.add_parser("make"); m.add_argument("--data"); m.add_argument("--split"); m.add_argument("--s3"); m.add_argument("--n", type=int, default=400)
m.add_argument("--workers", type=int, default=16); m.add_argument("--raw300", action="store_true"); m.add_argument("--table-first", action="store_true")
s = sub.add_parser("score"); s.add_argument("--data"); s.add_argument("--split"); s.add_argument("--finder", nargs="+")
a = ap.parse_args()
if a.cmd == "make":
    meta = json.load(open(f"{a.data}/{a.split}_pages.json"))
    rng = np.random.default_rng(5)
    idx = rng.permutation(len(meta))
    pick = [meta[i]["page"] for i in idx][: a.n] if a.n > 0 else [m["page"] for m in meta]
    if a.table_first:  # every table-heavy page, then the rest
        pick = sorted(set([m["page"] for m in meta if m["table"] >= 0.3] + pick))
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(make_one, [(n, a.data, a.split, a.s3, a.raw300) for n in pick]), 1):
            if k % 100 == 0:
                print(k, flush=True)
    print("made", len(pick))
else:
    score(a)
