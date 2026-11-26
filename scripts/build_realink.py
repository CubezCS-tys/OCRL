"""Build the real-ink cluster sets from downloaded documents (PDF + Azure JSON), and check their labels.

    uv run python scripts/build_realink.py build --split train
    uv run python scripts/build_realink.py build --split heldout
    uv run python scripts/build_realink.py check --split train --n 24000

Re-runnable: `build` rewrites the split's shards from scratch, so it can be repeated after the line finder changes.
The documents are private; nothing here leaves the machine.
"""
import argparse, json, subprocess, tempfile, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

S3 = Path("/home/yassine/data/mandumah_s3")
OUT = Path("data/realink")


def one_document(args):
    stem, max_pages, per_page, seed = args
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    from ocrl.recog.realink import align_page
    pdf, js = S3 / stem / f"{stem}.pdf", S3 / stem / f"{stem}.json"
    stats, crops, labels, prov = Counter(), [], [], []
    if not (pdf.exists() and js.exists()):
        return stem, Counter(missing_file=1), crops, labels, prov
    try:
        az = json.loads(js.read_text(encoding="utf-8"))
        az = az.get("analyzeResult", az)
        rng = np.random.default_rng([seed, abs(hash(stem)) % (2**31)])
        with tempfile.TemporaryDirectory() as tmp:
            n = min(len(az["pages"]), max_pages)
            subprocess.run(["pdftoppm", "-r", "300", "-gray", "-png", "-l", str(n), str(pdf), f"{tmp}/p"], check=True,
                           stderr=subprocess.DEVNULL, timeout=600)
            for pi, f in enumerate(sorted(Path(tmp).glob("p-*.png"))[:n]):
                gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                pg = az["pages"][pi]
                if gray is None or not pg.get("words"):
                    stats["pages_without_words"] += 1
                    continue
                samples, st = align_page(gray, pg["words"], pg["width"], pg["height"])
                stats.update(st); stats["pages"] += 1
                if len(samples) > per_page:  # a few from every page beats everything from a few pages
                    samples = [samples[i] for i in sorted(rng.choice(len(samples), per_page, replace=False))]
                for s in samples:
                    crops.append(s.crop); labels.append(s.label); prov.append((stem, pi + 1, s.word, s.rank))
    except Exception as e:  # one unreadable PDF must not stop the build
        stats["documents_failed"] += 1
        stats[f"error:{type(e).__name__}"] += 1
    stats["documents"] += 1
    return stem, stats, crops, labels, prov


def build(a):
    from ocrl.recog.realink import save_shard
    ids = json.load(open(OUT / f"{a.split}_ids.json"))
    dest = OUT / a.split
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("shard_*.pt"):
        old.unlink()
    total, crops, labels, prov, n_shard, kept, t = Counter(), [], [], [], 0, 0, time.time()

    def flush():
        nonlocal crops, labels, prov, n_shard
        if crops:
            save_shard(dest / f"shard_{n_shard:03d}.pt", crops, labels, prov)
            n_shard += 1
            crops, labels, prov = [], [], []

    with Pool(a.workers) as pool:
        for k, (stem, stats, c, l, p) in enumerate(pool.imap_unordered(one_document, [(s, a.max_pages, a.per_page, 7) for s in ids]), 1):
            total.update(stats); crops += c; labels += l; prov += p; kept += len(c)
            if len(crops) >= 25000:
                flush()
            if k % 25 == 0:
                print(f"  {k}/{len(ids)} documents, {kept:,} clusters kept, {time.time() - t:.0f} s", flush=True)
    flush()
    total["clusters_kept"] = kept
    json.dump(dict(total), open(dest / "stats.json", "w"), indent=1)
    w = max(total["words"], 1)
    print(f"{a.split}: documents {total['documents']} ({total['documents_failed']} failed, {total['missing_file']} missing) | pages {total['pages']} | "
          f"words {total['words']:,} | aligned {total['aligned']:,} ({100 * total['aligned'] / w:.1f}%) | clusters {total['clusters']:,}, kept {kept:,} in {n_shard} shards")
    for k in ("low_confidence", "not_our_characters", "no_clusters", "too_few_clusters", "too_many_clusters"):
        print(f"   skipped, {k.replace('_', ' ')}: {total[k]:,} ({100 * total[k] / w:.1f}%)")


def check(a):
    import cv2, torch
    from rapidfuzz.distance import Levenshtein
    from ocrl.recog.decode import greedy
    from ocrl.recog.model import load
    from ocrl.recog.realink import load_shard
    from ocrl.text import strip_marks
    torch.set_num_threads(6)
    model, vocab, _ = load(a.ckpt)
    model.eval()
    rng = np.random.default_rng(3)
    samples, prov = [], []
    for f in sorted((OUT / a.split).glob("shard_*.pt")):
        d = torch.load(f, map_location="cpu")
        s = load_shard(f)
        samples += s; prov += d["prov"]
    pick = rng.choice(len(samples), min(a.n, len(samples)), replace=False)
    pick = sorted(pick, key=lambda i: samples[i][0].shape[2])
    c, wrong = Counter(), []
    with torch.no_grad():
        for j in range(0, len(pick), 512):
            idx = pick[j : j + 512]
            W = max(samples[i][0].shape[2] for i in idx)
            x = torch.zeros(len(idx), 2, 48, W)
            for k, i in enumerate(idx):
                x[k, :, :, : samples[i][0].shape[2]] = torch.from_numpy(samples[i][0].copy()).float() / 255
            logp = model(x).numpy()
            for k, i in enumerate(idx):
                pred = greedy(logp[: samples[i][0].shape[2] // 2, k], vocab)
                lab = samples[i][1]
                c["n"] += 1; c["exact"] += pred == lab
                c["exact_bare"] += strip_marks(pred) == strip_marks(lab)
                c["edits"] += Levenshtein.distance(pred, lab); c["chars"] += len(lab)
                c["edits_bare"] += Levenshtein.distance(strip_marks(pred), strip_marks(lab)); c["chars_bare"] += len(strip_marks(lab))
                if strip_marks(pred) != strip_marks(lab):
                    wrong.append((i, pred))
    print(f"{a.split}: {c['n']:,} clusters | reader agrees exactly {100 * c['exact'] / c['n']:.2f}% (ignoring vowel marks {100 * c['exact_bare'] / c['n']:.2f}%) | "
          f"character disagreement {100 * c['edits'] / c['chars']:.2f}% (ignoring marks {100 * c['edits_bare'] / max(c['chars_bare'], 1):.2f}%)")
    # 40 disagreements as pictures, with the label and the reading listed beside their numbers
    out = Path(a.show); out.mkdir(parents=True, exist_ok=True)
    chosen = [wrong[i] for i in rng.choice(len(wrong), min(40, len(wrong)), replace=False)]
    lines = []
    for sheet in range(0, len(chosen), 10):
        tiles = []
        for n, (i, pred) in enumerate(chosen[sheet : sheet + 10], sheet + 1):
            img = samples[i][0][:, :, ::-1]  # back to how it looks on the page
            rgb = 255 - img[0].astype(np.float32)[..., None] - img[1].astype(np.float32)[..., None] * np.array([0.12, 0.42, 0.62], np.float32)
            rgb = cv2.resize(np.clip(rgb, 0, 255).astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.copyMakeBorder(rgb, 34, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(rgb, str(n), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
            tiles.append(rgb)
            lines.append(f"{n:2d}  label «{samples[i][1]}»  reader «{pred}»  {prov[i][0]} p{prov[i][1]} word {prov[i][2]} rank {prov[i][3]}")
        H = max(t.shape[0] for t in tiles)
        row = np.hstack([cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
        cv2.imwrite(str(out / f"disagree_{sheet // 10}.png"), row)
    (out / "disagree.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
b = sub.add_parser("build"); b.add_argument("--split", default="train"); b.add_argument("--workers", type=int, default=6)
b.add_argument("--max-pages", type=int, default=12); b.add_argument("--per-page", type=int, default=160)
c = sub.add_parser("check"); c.add_argument("--split", default="train"); c.add_argument("--n", type=int, default=24000)
c.add_argument("--ckpt", default="out/recognizer_v3.pt"); c.add_argument("--show", default="data/realink/check")
a = ap.parse_args()
build(a) if a.cmd == "build" else check(a)
