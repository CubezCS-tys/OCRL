"""Train a line reader from scratch on Azure-labelled lines: the owner's recipe (inkscript experiment 89), our data.

Same network as theirs, so the two can be compared like for like. The image is mirrored so Arabic runs left
to right for the network; numbers and Latin runs are reversed in the label to match (ocrl.recog.linereader.to_logical
is its own inverse). The checkpoint is written in the same form as theirs, so LineReader loads either.
"""
import argparse, glob, time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

from ocrl.recog.linereader import LineNet, to_logical

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--budget", type=int, default=60000, help="total line width per batch, pixels")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", default="out/line_reader.pt"); ap.add_argument("--min-char", type=int, default=30)
ap.add_argument("--workers", type=int, default=12, help="processes that prepare batches, so the GPU never waits")
ap.add_argument("--shards", type=int, default=0, help="use only this many training shards (0 = all); for a quick speed test")
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = run all epochs); for a quick speed test")
ap.add_argument("--init", default="", help="start from this checkpoint (its character list is kept) instead of from scratch")
ap.add_argument("--pct-start", type=float, default=0.05, help="share of the steps spent warming the learning rate up")
ap.add_argument("--aug-old", type=float, default=0.0, help="chance that a training line is also made to look like old, faded print")
a = ap.parse_args()
torch.set_num_threads(8); cv2.setNumThreads(2)
dev = "cuda"; rng = np.random.default_rng(0)


def load(split):
    items = []
    files = sorted(glob.glob(f"data/reader/lines_{split}/*.pt"))
    for f in files[: a.shards] if a.shards and split == "train" else files:
        d = torch.load(f, map_location="cpu", mmap=True)  # file-backed: the images can be dropped from memory and read back
        strip, off = d["strip"].numpy(), d["offsets"].tolist()
        items += [(strip[:, s:e], lab) for s, e, lab in zip(off, off[1:], d["labels"])]
    return items


train, test = load("train"), load("test") if not a.max_steps else []
init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
count = Counter(ch for _, l in train for ch in l)
chars = init["meta"]["chars"] if init else sorted(ch for ch, n in count.items() if n >= a.min_char)
idx = {ch: i + 1 for i, ch in enumerate(chars)}
keep = lambda items: [(im, l) for im, l in items if l and all(ch in idx for ch in l) and im.shape[1] >= 2 * len(l) + 4]
n0 = len(train); train, test = keep(train), keep(test)
print(f"{len(train):,} training lines ({n0 - len(train):,} dropped for rare characters or too little width), {len(test):,} test lines, {len(chars)} characters", flush=True)
enc = lambda text: [idx[ch] for ch in to_logical(text)]  # the order the mirrored image shows it


def augment(im):
    im = im.astype(np.float32)
    if rng.random() < 0.5:  # the cut is never placed quite the same: shift and stretch a little
        h = im.shape[0]; s = rng.uniform(0.86, 1.12); dy = rng.uniform(-3, 3)
        M = np.float32([[rng.uniform(0.92, 1.08), 0, 0], [0, s, dy + (1 - s) * h / 2]])
        im = cv2.warpAffine(im, M, (max(int(im.shape[1] * M[0, 0]), 8), h), borderValue=255)
    if rng.random() < 0.3: im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.4: im = 255 - (255 - im) * rng.uniform(0.6, 1.3)
    if rng.random() < 0.3: im = im + rng.normal(0, rng.uniform(2, 10), im.shape)
    return np.clip(im, 0, 255)


def age(im):
    """Old print: ink that has faded or broken up, letters that bled or thinned, yellowed low-contrast paper,
    speckle, and a scan made at lower resolution or saved as a rough JPEG. im is float, 0 ink .. 255 paper."""
    h, w = im.shape
    if rng.random() < 0.4:  # strokes thicken (ink spread) or thin (worn type)
        k = np.ones((2, 2), np.uint8)
        im = cv2.erode(im, k) if rng.random() < 0.5 else cv2.dilate(im, k)
    if rng.random() < 0.5:  # ink faded unevenly: a smooth random field lifts parts of the strokes toward paper
        f = cv2.resize(rng.random((max(h // 8, 2), max(w // 8, 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        ink = 255 - im
        im = 255 - ink * np.clip(1 - rng.uniform(0.3, 0.75) * f, 0.15, 1)
    if rng.random() < 0.3:  # broken strokes: small specks of ink missing
        im = np.where(rng.random(im.shape) < rng.uniform(0.03, 0.12), np.maximum(im, rng.uniform(120, 230)), im)
    if rng.random() < 0.6:  # grey paper, weak contrast
        paper, ink_lvl = rng.uniform(150, 235), rng.uniform(20, 110)
        im = ink_lvl + (im / 255) * (paper - ink_lvl)
    if rng.random() < 0.3:  # dirt and speckle
        m = rng.random(im.shape) < rng.uniform(0.002, 0.015)
        im = np.where(m, rng.uniform(30, 120), im)
    if rng.random() < 0.4:  # low-resolution scan
        s = rng.uniform(0.45, 0.8)
        small = cv2.resize(im, (max(int(w * s), 4), max(int(h * s), 4)), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:  # rough JPEG
        ok, buf = cv2.imencode(".jpg", np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(15, 50))])
        if ok: im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return im


def batch(group, train_mode):
    ims = [augment(age(im.astype(np.float32)) if a.aug_old and rng.random() < a.aug_old else im) if train_mode else im.astype(np.float32) for im, _ in group]
    W = -(-max(i.shape[1] for i in ims) // 64) * 64  # few distinct widths, so the GPU can reuse its fastest plan
    x = torch.zeros(len(ims), 1, 48, W)
    for k, im in enumerate(ims):
        x[k, 0, :, : im.shape[1]] = torch.from_numpy(1 - im[:, ::-1].copy() / 255)  # mirrored, ink as 1
    widths = torch.tensor([i.shape[1] for i in ims])
    return x, widths, [l for _, l in group]


def batches(items, shuffle):
    order = sorted(range(len(items)), key=lambda i: items[i][0].shape[1] + (rng.uniform(0, 40) if shuffle else 0))
    out, cur, w = [], [], 0
    for i in order:
        wi = items[i][0].shape[1]
        if cur and (len(cur) + 1) * max(w, wi) > a.budget:
            out.append(cur); cur, w = [], 0
        cur.append(i); w = max(w, wi)
    if cur: out.append(cur)
    if shuffle: rng.shuffle(out)
    return out


def decode(ids):
    keep = ids[np.insert(ids[1:] != ids[:-1], 0, True)]
    return to_logical("".join(chars[c - 1] for c in keep if c))


@torch.no_grad()
def evaluate(net, items):
    net.eval(); e = n = exact = 0
    for b in batches(items, False):
        x, widths, labels = batch([items[i] for i in b], False)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x.to(dev))
        ids = out.float().argmax(-1).cpu().numpy()
        for k, lab in enumerate(labels):
            got = decode(ids[: max(int(widths[k]) // 4, 1), k])
            e += Levenshtein.distance(got, lab); n += len(lab); exact += got == lab
    net.train()
    return 100 * e / max(n, 1), 100 * exact / max(len(items), 1)


class Batches(torch.utils.data.Dataset):
    """One item is one whole prepared batch, so the augmentation runs in the worker processes."""

    def __init__(self, groups): self.groups = groups
    def __len__(self): return len(self.groups)
    def __getitem__(self, k):
        x, widths, labels = batch([train[i] for i in self.groups[k]], True)
        tg = [enc(l) for l in labels]
        return x, widths, torch.tensor([c for t in tg for c in t]), torch.tensor([len(t) for t in tg])


def seed_worker(k):
    global rng
    rng = np.random.default_rng(1000 * int(torch.initial_seed() % 100000) + k); cv2.setNumThreads(1); torch.set_num_threads(1)


torch.backends.cudnn.benchmark = False  # batch size and width both vary, so there is no single plan to find
net = LineNet(len(chars) + 1)
if init: net.load_state_dict(init["model"])
net = net.to(dev)
per_epoch = len(batches(train, False))
steps = a.max_steps or a.epochs * per_epoch
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * per_epoch + 10, pct_start=a.pct_start)
scaler = torch.amp.GradScaler("cuda")
print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M weights, {per_epoch} steps an epoch, {steps} steps in all", flush=True)
t0, step, run, lines_seen = time.time(), 0, 0.0, 0
for ep in range(a.epochs):
    loader = torch.utils.data.DataLoader(Batches(batches(train, True)), batch_size=None, shuffle=False, num_workers=a.workers, worker_init_fn=seed_worker, prefetch_factor=4, pin_memory=True)
    for x, widths, targets, lengths in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x.to(dev, non_blocking=True))
        loss = F.ctc_loss(out.float(), targets, torch.clamp(widths // 4, min=1), lengths, zero_infinity=True)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); scaler.step(opt); scaler.update()
        if step < a.epochs * per_epoch + 9: sched.step()  # shuffling changes the batch count a little from pass to pass
        step += 1; lines_seen += len(lengths); run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0 or step == a.max_steps:
            el = time.time() - t0
            print(f"ep {ep} step {step}/{steps} loss {run:.4f} {step / el:.1f} it/s, {lines_seen / el:.0f} lines/s", flush=True)
        if step == a.max_steps:
            raise SystemExit
    cer, exact = evaluate(net, test[:20000])
    print(f"== epoch {ep}: unseen journals, against Azure's text: characters differing {cer:.2f}%, lines identical {exact:.1f}%", flush=True)
    ck = {"model": net.state_dict(), "meta": {"chars": chars, "H": 48, "hid": 384, "feat": 384, "arch": "lstm", "made_by": "OCRL scripts/train_line_reader.py"}}
    torch.save(ck, a.out); torch.save(ck, a.out.replace(".pt", f"_e{ep}.pt"))  # the last, and a copy of every epoch
