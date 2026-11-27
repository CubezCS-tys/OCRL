"""Train the cluster recogniser on endless synthetic pages."""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ocrl.recog.data import ClusterStream, TextSource, collate, font_paths, latin_paths, make_page, page_samples
from ocrl.recog.decode import greedy
from ocrl.recog.model import VOCAB, Recognizer, load, save
from ocrl.text import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--out", default="out/recognizer.pt")
ap.add_argument("--resume", default=None)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--real", default="", help="folder of real-ink shards to mix in")
ap.add_argument("--real-every", type=int, default=2, help="every n-th batch is real ink")
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--hidden", type=int, default=192)
args = ap.parse_args()

CORPUS = "data/corpus/sentences.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(max(1, torch.get_num_threads() - args.workers))


def fixed_set(split: str, pages: int, seed: int):
    """A frozen validation set: test-only sentences, same degradations, no crop jitter."""
    rng = np.random.default_rng(seed)
    corpus = load_corpus(CORPUS)
    src = TextSource(corpus[int(len(corpus) * 0.95):], rng)
    fonts, cache, out = font_paths(split), {}, []
    for _ in range(pages):
        out.extend(page_samples(make_page(fonts, src, rng, cache, latin_fonts=latin_paths())))
    return out


def evaluate(model, samples):
    model.eval()
    right = edits = chars = 0
    samples = sorted(samples, key=lambda s: s[0].shape[2])
    with torch.no_grad():
        for i in range(0, len(samples), 256):
            chunk = samples[i : i + 256]
            x, widths, _, _ = collate(chunk)
            logp = model(x.to(device).float() / 255).cpu().numpy()
            for j, (_, label) in enumerate(chunk):
                pred = greedy(logp[: int(widths[j]) // 2, j])
                right += pred == label
                edits += _edit(pred, label)
                chars += len(label)
    model.train()
    return right / len(samples), edits / max(chars, 1)


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


val = {"seen fonts": fixed_set("train", 40, 101), "unseen fonts": fixed_set("held_out", 40, 102)}
print({k: len(v) for k, v in val.items()}, "validation clusters; device", device, flush=True)

model, start = Recognizer(args.width, args.hidden).to(device), 0
if args.resume:
    model, vocab, start = load(args.resume, device)
    assert vocab == VOCAB, "checkpoint was trained with a different character set"
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
loader = DataLoader(ClusterStream(CORPUS, batch=args.batch, seed=start), batch_size=None, num_workers=args.workers, prefetch_factor=8)

real = []
if args.real:
    import glob
    from ocrl.recog.realink import load_shard
    for f in sorted(glob.glob(f"{args.real}/*.pt")):
        real.extend((c, l) for c, l in load_shard(f) if l and all(ch in VOCAB for ch in l) and c.shape[2] >= 8)
    print(len(real), "real-ink clusters loaded", flush=True)
real_rng = np.random.default_rng(start + 7)


def real_batch():
    """Real clusters of similar width, so little is wasted on padding."""
    pick = real_rng.integers(0, len(real), args.batch * 3)
    pick = sorted(pick, key=lambda i: real[i][0].shape[2])
    a = int(real_rng.integers(0, len(pick) - args.batch + 1))
    return collate([real[i] for i in pick[a : a + args.batch]])


t0, run = time.time(), 0.0
for step, (x, widths, flat, lengths) in enumerate(loader, 1):
    if real and step % args.real_every == 0:
        x, widths, flat, lengths = real_batch()
    logp = model(x.to(device).float() / 255)
    loss = F.ctc_loss(logp, flat.to(device), Recognizer.frames(widths), lengths, zero_infinity=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
    if step % 200 == 0:
        print(f"step {step:6d}  loss {run:.4f}  {step / (time.time() - t0):.1f} it/s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save(model, args.out, start + step)
        for name, samples in val.items():
            acc, cer = evaluate(model, samples)
            print(f"   eval {name:13s} cluster-exact {100 * acc:.2f}%  CER {100 * cer:.2f}%", flush=True)
    if step >= args.steps:
        break
