"""Parse the three hand-keyed documents once and type them with several structure models, scored with the keys.

The pages (rendered by parse_doc.py, out/parsed_C/DOC/pages) are read once (cached in OUT/DOC.infos.pkl); every
typer spec then runs the rest of ocrl.doc.parse.parse on a fresh copy of those lines, and the master file is
scored by scripts/score_with_compare.py (inkscript_v2's scorer, unchanged).
    uv run python scripts/keyed_typers.py --specs "out/structure_v2.pt+out/structure_v3.pt" "out/structure_v2.pt+out/structure_v5.pt"
"""
import argparse, copy, json, pickle, re, subprocess, sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent))
import e2e_judged as E
from ocrl.doc import parse as P
from ocrl.structure.tag import Typer

ap = argparse.ArgumentParser(); ap.add_argument("--specs", nargs="+", required=True)
ap.add_argument("--docs", nargs="*", default=["0073-000-239-002", "0175-000-004-016", "0499-026-119-022"])
ap.add_argument("--out", default="out/structure_v5/keyed"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); cv2.setNumThreads(2)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ocr = None
for doc in a.docs:
    files = sorted(Path(f"out/parsed_C/{doc}/pages").glob("p-*.png"), key=lambda f: int(f.stem[2:]))
    nums = [int(f.stem[2:]) for f in files]
    images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    cache = out / f"{doc}.infos.pkl"
    if cache.exists():
        read = pickle.loads(cache.read_bytes())
    else:
        if ocr is None:
            from ocrl.pipeline import OCR
            ocr = OCR(E.CKPT, linefinder=E.LINEFINDER, line_reader=E.READER)
        read = {n: P.read_page(ocr, g, n) for n, g in zip(nums, images)}
        cache.write_bytes(pickle.dumps(read))
    key = f"{doc}_p{nums[0]}-{nums[-1]}" if len(nums) > 1 else f"{doc}_p{nums[0]}"
    for spec in a.specs:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", spec.replace("out/", "").replace(".pt", ""))
        real = P.read_page
        P.read_page = lambda o, g, n: copy.deepcopy(read[n])
        try:
            infos, blocks = P.parse(None, images, nums[0], Typer(spec))
        finally:
            P.read_page = real
        master = P.to_master(doc, infos, blocks, 300, "v7 line reader", f"structure {spec}")
        f = out / f"{doc}.{tag}.master.json"
        f.write_text(json.dumps(master, ensure_ascii=False, indent=1), encoding="utf-8")
        r = subprocess.run([sys.executable, "scripts/score_with_compare.py", "--name", key, "--ours", str(f), "--quiet"], capture_output=True, text=True)
        print(f"== {doc} | {spec} | blocks by type: {dict(__import__('collections').Counter(b['type'] for b in master['blocks']))}")
        print(r.stdout.strip() or r.stderr.strip()[-800:], flush=True)
