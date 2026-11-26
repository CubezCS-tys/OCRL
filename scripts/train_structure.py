"""Train the structure model on the Desk lines. Held out: a random 300-article sample, and one whole journal."""
import argparse, collections, time

import numpy as np
import torch
import torch.nn.functional as F

from ocrl.structure.features import ROLES
from ocrl.structure.model import LineTagger, save

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--out", default="out/structure.pt")
ap.add_argument("--unseen-journal", default="j0097")
ap.add_argument("--d", type=int, default=192)
ap.add_argument("--layers", type=int, default=4)
args = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.set_num_threads(4)  # a machine reporting hundreds of threads makes small tensor operations crawl
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = torch.load("data/structure/desk_lines.pt")
off, meta, vocab = D["offsets"].tolist(), D["meta"], D["vocab"]
stems = sorted({m[0] for m in meta})
holdout_doc = {s for s in stems if rng.random() < 0.03}
split = {"train": [], "dev": [], "unseen documents": [], "unseen journal": []}
for i, (stem, page, run) in enumerate(meta):
    key = "unseen documents" if run.startswith("v33_300b") else "unseen journal" if args.unseen_journal in run else "dev" if stem in holdout_doc else "train"
    split[key].append(i)
print({k: len(v) for k, v in split.items()}, "pages | device", dev, flush=True)
MAXL = 320


def batch(idx, train=False):
    L = min(max(off[i + 1] - off[i] for i in idx), MAXL)
    g = torch.zeros(len(idx), L, D["geom"].shape[1]); t = torch.zeros(len(idx), L, D["text"].shape[1], dtype=torch.long)
    r = torch.full((len(idx), L), -100, dtype=torch.long); s = torch.zeros(len(idx), L); pad = torch.ones(len(idx), L, dtype=torch.bool)
    for k, i in enumerate(idx):
        a, b = off[i], min(off[i + 1], off[i] + L)
        sel = np.arange(a, b)
        if train and b - a > 6 and rng.random() < 0.3:  # our own line finder will miss some lines
            sel = sel[rng.random(b - a) > 0.04]
        n = len(sel)
        g[k, :n] = D["geom"][sel]; t[k, :n] = D["text"][sel].long(); r[k, :n] = D["role"][sel].long(); s[k, :n] = D["start"][sel].float(); pad[k, :n] = False
    if train:
        g += torch.randn_like(g) * 0.006  # line boxes from another finder sit slightly differently
        noise = (torch.rand(t.shape) < 0.06) & (t > 0)  # and its reader misreads some letters
        t = torch.where(noise, torch.randint(2, len(vocab) + 2, t.shape), t)
    return g.to(dev), t.to(dev), r.to(dev), s.to(dev), pad.to(dev)


def evaluate(model, idx):
    model.eval(); c = collections.Counter(); per = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for j in range(0, len(idx), 64):
            g, t, r, s, pad = batch(idx[j : j + 64])
            lr_, ls_ = model(g, t, pad)
            pr, ps = lr_.argmax(-1), ls_ > 0
            ok = ~pad
            c["lines"] += int(ok.sum()); c["role_ok"] += int(((pr == r) & ok).sum())
            c["start_tp"] += int((ps & (s > 0) & ok).sum()); c["start_fp"] += int((ps & (s == 0) & ok).sum()); c["start_fn"] += int((~ps & (s > 0) & ok).sum())
            for role in range(len(ROLES)):
                m = (r == role) & ok
                per[role]["n"] += int(m.sum()); per[role]["ok"] += int(((pr == r) & m).sum())
            # blocks: a run of lines from one predicted start to the next, compared with the true runs
            for k in range(g.shape[0]):
                n = int(ok[k].sum())
                true = torch.cumsum(s[k, :n], 0); pred = torch.cumsum(ps[k, :n].float(), 0)
                same_role = pr[k, :n] == r[k, :n]
                for b in true.unique():
                    m = true == b
                    c["blocks"] += 1
                    first = int(m.nonzero()[0]); last = int(m.nonzero()[-1])
                    whole = pred[m].unique().numel() == 1 and bool(ps[k, first]) and (last + 1 >= n or bool(ps[k, last + 1])) if first > 0 else pred[m].unique().numel() == 1 and (last + 1 >= n or bool(ps[k, last + 1]))
                    c["blocks_ok"] += bool(whole and same_role[m].all())
    model.train()
    f1 = 2 * c["start_tp"] / max(2 * c["start_tp"] + c["start_fp"] + c["start_fn"], 1)
    return c["role_ok"] / c["lines"], f1, c["blocks_ok"] / max(c["blocks"], 1), per


model = LineTagger(len(vocab), args.d, args.layers, 4 if args.d < 256 else 8).to(dev)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
steps = args.epochs * (len(split["train"]) // args.batch)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.06)
t0, step = time.time(), 0
for ep in range(args.epochs):
    order = rng.permutation(split["train"])
    # pages of similar length together, so little is wasted on padding
    order = sorted(order, key=lambda i: (off[i + 1] - off[i]) // 12 + rng.random())
    batches = [order[i : i + args.batch] for i in range(0, len(order) - args.batch + 1, args.batch)]
    for bi in rng.permutation(len(batches)):
        g, t, r, s, pad = batch(batches[bi], train=True)
        lr_, ls_ = model(g, t, pad)
        loss = F.cross_entropy(lr_[~pad], r[~pad]) + F.binary_cross_entropy_with_logits(ls_[~pad], s[~pad])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {step / (time.time() - t0):.1f} it/s", flush=True)
    acc, f1, blk, _ = evaluate(model, split["dev"])
    print(f"== epoch {ep}: dev line roles {100 * acc:.2f}%  block starts F1 {100 * f1:.2f}%  blocks exact {100 * blk:.2f}%", flush=True)
    save(model, vocab, args.out)
for name in ("dev", "unseen documents", "unseen journal"):
    acc, f1, blk, per = evaluate(model, split[name])
    print(f"\n{name}: {len(split[name])} pages | line roles {100 * acc:.2f}% | block starts F1 {100 * f1:.2f}% | blocks exact {100 * blk:.2f}%")
    print("   " + "  ".join(f"{ROLES[k]} {100 * v['ok'] / v['n']:.0f}% ({v['n']})" for k, v in sorted(per.items(), key=lambda kv: -kv[1]['n']) if v["n"]))
