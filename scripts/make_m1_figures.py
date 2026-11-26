"""Collect real pictures and numbers from the pipeline for docs/milestone1.html -> docs/m1_data.js"""
import argparse, base64, json, re
from pathlib import Path

import cv2
import numpy as np
import torch

from ocrl.blobs.layout import analyze
from ocrl.blobs.viz import overlay
from ocrl.pipeline import EXTRA, MISSING, OCR, check
from ocrl.recog.crops import cluster_ids, cluster_labels, crop
from ocrl.recog.data import TextSource, font_paths, make_page
from ocrl.recog.decode import greedy, nbest
from ocrl.recog.model import VOCAB, Recognizer
from ocrl.rules.signature import satellite_range
from ocrl.text import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/gpu/out/recognizer_gpu.pt")
ap.add_argument("--log", default="out/gpu/train.log")
ap.add_argument("--m1", default="out/m1.json")
args = ap.parse_args()


def png(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


ocr = OCR(args.ckpt, repair=True)
model = ocr.model
rng = np.random.default_rng(2026)
corpus = load_corpus("data/corpus/sentences.txt")
src = TextSource(corpus[int(len(corpus) * 0.95):], rng)
fonts, cache = font_paths("held_out"), {}

# --- one page for the "from page to clusters" strip ------------------------------------------
page = make_page(fonts, src, rng, cache, lines=3, strength=1.0)
lay = analyze(page.image)
res = ocr.read(page.image, lay)
strip = {
    "page": png(page.image),
    "overlay": png(overlay(page.image, lay, scale=1)),
    "font": page.font, "px": page.px,
    "truth": page.lines, "read": [ln.text for ln in res.lines],
    "blobs": int((lay.role > 0).sum()), "clusters": sum(len(ln.bodies) for ln in lay.lines),
}

# --- clusters to watch the net read -----------------------------------------------------------
def describe(pg, lay, ln, ids, label, cl, notes):
    """Everything the page shows about one cluster, taken from a real pipeline run (`cl`)."""
    arr = crop(pg.image, lay, ids, ln.baseline, ln.scale)
    with torch.no_grad():
        logp = model(torch.from_numpy(arr).float()[None] / 255)[:, 0].numpy()
    p = np.exp(logp)
    n_sat = len(ids) - 1
    active = {MISSING, EXTRA} - ({MISSING} if "dots-fusing" in notes else set())
    doubts = lambda t: [x for x in check(t, n_sat) if x in active]
    keep = [0] + [i for i in range(1, len(VOCAB)) if p[:, i].max() > 0.04]
    own, other = arr[0][:, ::-1].astype(np.float32), arr[1][:, ::-1].astype(np.float32)  # back to natural orientation
    rgb = 255 - own[..., None] - other[..., None] * np.array([0.12, 0.42, 0.62], np.float32)  # neighbours in a warm tint (BGR)
    rgb = cv2.resize(np.clip(rgb, 0, 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return {
        "img": png(rgb), "W": int(arr.shape[2]), "T": int(logp.shape[0]),
        "symbols": ["blank"] + [VOCAB[i] for i in keep[1:]],
        "probs": [[round(float(v), 3) for v in p[:, i]] for i in keep],
        "path": [VOCAB[i] if i else "" for i in p.argmax(1)],
        "first": cl.first, "final": cl.text, "label": label, "satellites": n_sat,
        "first_flags": doubts(cl.first), "first_range": list(satellite_range(cl.first)),
        "touching": cl.touching, "dots_fusing": "dots-fusing" in notes,
        "alts": [[a, round(float(s), 2), doubts(a)] for a, s in nbest(logp)[:4]],
    }


want = {"long": None, "repeat": None, "single": None, "digit": None, "repaired": None, "touching": None, "wrong": None}
for _ in range(80):
    pg = make_page(fonts, src, rng, cache, lines=8, strength=1.0)
    l2 = analyze(pg.image)
    r2 = ocr.read(pg.image, l2)
    labels, ids = cluster_labels(pg, l2), cluster_ids(l2)
    for li, ln in enumerate(r2.lines):
        for cl in ln.clusters:
            label = labels.get(cl.body, "")
            if not label:
                continue
            right, kind = cl.text == label, None
            if right and not cl.flags and not cl.touching:
                if want["long"] is None and 5 <= len(label) <= 7: kind = "long"
                elif want["repeat"] is None and re.search(r"(.)\1", label) and len(label) >= 3: kind = "repeat"
                elif want["single"] is None and label in "اورد": kind = "single"
                elif want["digit"] is None and label.isdigit(): kind = "digit"
            elif right and cl.flags == ["repaired"] and want["repaired"] is None: kind = "repaired"
            elif right and cl.touching and not cl.flags and want["touching"] is None and label.isalpha(): kind = "touching"
            elif not right and len(label) >= 2 and want["wrong"] is None: kind = "wrong"
            if kind:
                want[kind] = describe(pg, l2, l2.lines[li], ids[cl.body], label, cl, r2.notes)
    if all(v is not None for v in want.values()):
        break
titles = {"long": "A long sub-word", "repeat": "A doubled letter", "single": "A single letter", "digit": "A digit",
          "repaired": "A misread the rules caught", "touching": "Two units touching", "wrong": "A misread that got through"}
examples = [dict(v, kind=k, title=titles[k]) for k, v in want.items() if v is not None]

# --- training curve from the log ----------------------------------------------------------------
steps, evals, step = [], [], 0
for line in Path(args.log).read_text().splitlines():
    if m := re.match(r"step\s+(\d+)\s+loss ([\d.]+)\s+([\d.]+) it/s", line):
        step = int(m[1]); steps.append([step, float(m[2]), float(m[3])])
    elif m := re.match(r"\s+eval (seen|unseen) fonts\s+cluster-exact ([\d.]+)%\s+CER ([\d.]+)%", line):
        at = step if step % 4000 == 0 else step + (4000 - step % 4000) if step else 4000
        if not evals or evals[-1]["step"] != at:
            evals.append({"step": at})
        evals[-1][m[1]] = float(m[2]); evals[-1][m[1] + "_cer"] = float(m[3])

params = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("cnn", "proj", "rnn", "out")}
spend = [json.loads(l) for l in Path("out/gpu/gpu_spend.jsonl").read_text().splitlines()] if Path("out/gpu/gpu_spend.jsonl").exists() else []
data = {
    "strip": strip, "examples": examples, "loss": steps, "evals": evals, "params": params, "vocab": len(VOCAB),
    "m1": json.load(open(args.m1)) if Path(args.m1).exists() else {}, "spend": spend,
    "train_fonts": [Path(p).stem for p in font_paths("train")], "held_out": [Path(p).stem for p in font_paths("held_out")],
}
Path("docs/m1_data.js").write_text("window.M1 = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print("examples:", [e["kind"] for e in examples], "| evals:", len(evals), "| params:", params, "| vocab:", len(VOCAB))
print("size", Path("docs/m1_data.js").stat().st_size // 1024, "KB")
