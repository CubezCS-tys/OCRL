"""Train the line finder on Azure's line boxes; test on held-out documents by counting lines found one-to-one."""
import argparse, glob, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ocrl.blobs.linefinder import CORE, UNet

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--crop", type=int, default=640)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", default="out/linefinder.pt")
a = ap.parse_args()
torch.set_num_threads(8); cv2.setNumThreads(2)
rng = np.random.default_rng(0); dev = "cuda"
load = lambda split: [(cv2.imread(f, 0), cv2.imread(f.replace(".png", ".mask.png"), 0), f) for f in sorted(glob.glob(f"data/linefinder/{split}/*.png")) if not f.endswith(".mask.png")]
train, held = load("train"), load("heldout")
print(len(train), "training pages,", len(held), "held-out pages", flush=True)


def sample():
    xs, ys = [], []
    for _ in range(a.batch):
        img, mask, _ = train[int(rng.integers(len(train)))]
        H, W = img.shape; c = a.crop
        y, x = int(rng.integers(0, max(H - c, 1))), int(rng.integers(0, max(W - c, 1)))
        i, m = img[y : y + c, x : x + c].astype(np.float32), mask[y : y + c, x : x + c]
        if i.shape != (c, c):
            i = np.pad(i, ((0, c - i.shape[0]), (0, c - i.shape[1])), constant_values=255); m = np.pad(m, ((0, c - m.shape[0]), (0, c - m.shape[1])))
        # scans differ in exposure, sharpness and grain
        if rng.random() < 0.5: i = 255 - (255 - i) * rng.uniform(0.6, 1.3)
        if rng.random() < 0.3: i = cv2.GaussianBlur(i, (0, 0), rng.uniform(0.4, 1.2))
        if rng.random() < 0.4: i = i + rng.normal(0, rng.uniform(2, 12), i.shape)
        if rng.random() < 0.3: i = i - rng.uniform(0, 60)  # tinted paper
        xs.append(np.clip(i, 0, 255) / 255); ys.append(m > 0)
    return torch.tensor(np.stack(xs), dtype=torch.float32)[:, None].to(dev), torch.tensor(np.stack(ys), dtype=torch.float32)[:, None].to(dev)


@torch.no_grad()
def evaluate(net, pages):
    net.eval(); found = total = extra = 0
    for img, _, f in pages:
        boxes = np.load(f.replace(".png", ".boxes.npy"))
        H, W = img.shape; ph, pw = (-H) % 8, (-W) % 8
        x = torch.from_numpy(np.pad(img, ((0, ph), (0, pw)), constant_values=255)).float()[None, None].to(dev) / 255
        prob = torch.sigmoid(net(x))[0, 0, :H, :W].cpu().numpy()
        n, lab = cv2.connectedComponents((prob > 0.5).astype(np.uint8), connectivity=8)
        used = set()
        for x0, y0, x1, y1 in boxes:
            h = y1 - y0
            ids = np.unique(lab[int(y0 + CORE[0] * h) : max(int(y0 + CORE[1] * h), int(y0 + CORE[0] * h) + 1), int(x0) : max(int(x1), int(x0) + 1)])
            ids = ids[ids > 0]
            total += 1
            # found one-to-one: exactly one predicted line covers this line's core, and no other line claimed it
            if len(ids) == 1 and ids[0] not in used:
                found += 1
            used.update(ids.tolist())
        extra += (n - 1) - len(used)
    net.train()
    return found / max(total, 1), extra / max(total, 1), total


net = UNet().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
scaler = torch.amp.GradScaler("cuda")
print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters", flush=True)
t0 = time.time()
for step in range(1, a.steps + 1):
    x, y = sample()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = net(x)
    logit = logit.float()
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    loss = F.binary_cross_entropy_with_logits(logit, y) + dice
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 250 == 0:
        print(f"step {step}/{a.steps} loss {loss.item():.4f} {step / (time.time() - t0):.1f} it/s", flush=True)
    if step % 1500 == 0 or step == a.steps:
        torch.save({"model": net.state_dict(), **net.config}, a.out)
        r, e, n = evaluate(net, held[:200])
        print(f"== step {step}: held-out lines found one-to-one {100 * r:.2f}% of {n}; predicted lines matching nothing {100 * e:.2f}%", flush=True)
