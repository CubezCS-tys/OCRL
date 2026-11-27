"""Train the line finder on the large set built by lf2_build.py. Same network, loss and augmentation as
scripts/train_linefinder.py; pages are read from disk by loader workers (55k pages do not belong in memory).

    python scripts/lf2_train.py --data data --steps 24000 --out out/lf2_a.pt [--base 24] [--split-gap 0] [--rule-cut]

Targets (core_mask) are built on the fly from the saved Azure line boxes:
  default       exactly the current recipe: the middle band (CORE) of each Azure line box, its full width
  --split-gap G each line's core is cut where two neighbouring words of the line are more than G line heights apart
  --rule-cut    cores are cut where an upright ruled line crosses them
"""
import argparse, glob, json, os, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ocrl.blobs.linefinder import CORE, UNet
from lf2_targets import target

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--crop", type=int, default=640); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--base", type=int, default=24)
ap.add_argument("--out", required=True); ap.add_argument("--split-gap", type=float, default=0.0); ap.add_argument("--rule-cut", action="store_true")
ap.add_argument("--eval-every", type=int, default=3000); ap.add_argument("--eval-pages", type=int, default=800); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--gpu-gb", type=float, default=13.5)
ap.add_argument("--only-cleaned", action="store_true", help="train only on pages that have NAME.cleanup.png")
ap.add_argument("--input", default="raw", choices=["raw", "clean"], help="clean: train and score on the page as clean_page() leaves it (NAME.cleanup.png)")
a = ap.parse_args()
SUF = ".cleanup.png" if a.input == "clean" else ".png"


class Pages(torch.utils.data.Dataset):
    def __init__(self, names, n):
        self.names, self.n = names, n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        cv2.setNumThreads(1)
        rng = np.random.default_rng((os.getpid() * 1000003 + k * 7919 + int(time.time() * 1e3)) % 2**32)
        f = self.names[int(rng.integers(len(self.names)))]
        img = cv2.imread(f + SUF, 0)
        if img is None:
            img = cv2.imread(f + ".png", 0)
        z = np.load(f + ".npz")
        mask = target(img, z, a.split_gap, a.rule_cut)
        c = a.crop
        H, W = img.shape
        y, x = int(rng.integers(0, max(H - c, 1))), int(rng.integers(0, max(W - c, 1)))
        i, m = img[y : y + c, x : x + c], mask[y : y + c, x : x + c]
        if rng.random() < 0.3:
            i = cv2.GaussianBlur(i, (0, 0), rng.uniform(0.4, 1.2))
        xs, ys = np.full((c, c), 255, np.uint8), np.zeros((c, c), np.uint8)
        xs[: i.shape[0], : i.shape[1]] = i; ys[: m.shape[0], : m.shape[1]] = m
        return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def evaluate(net, names):
    """The current recipe's measure: an Azure line is found one-to-one if exactly one predicted core covers its core band
    and no other Azure line claimed that core. Reported over all pages and over table-heavy ones."""
    net.eval(); r = {"all": [0, 0, 0], "table": [0, 0, 0]}
    for f, tab in names:
        img = cv2.imread(f + SUF, 0); boxes = np.load(f + ".npz")["boxes"]
        H, W = img.shape; ph, pw = (-H) % 8, (-W) % 8
        x = torch.from_numpy(np.pad(img, ((0, ph), (0, pw)), constant_values=255)).float()[None, None].to(dev) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prob = torch.sigmoid(net(x).float())[0, 0, :H, :W].cpu().numpy()
        n, lab = cv2.connectedComponents((prob > 0.5).astype(np.uint8), connectivity=8)
        used, found = set(), 0
        for x0, y0, x1, y1 in boxes:
            h = y1 - y0
            ids = np.unique(lab[int(y0 + CORE[0] * h) : max(int(y0 + CORE[1] * h), int(y0 + CORE[0] * h) + 1), int(x0) : max(int(x1), int(x0) + 1)])
            ids = ids[ids > 0]
            if len(ids) == 1 and ids[0] not in used:
                found += 1
            used.update(ids.tolist())
        for key in ("all", "table") if tab else ("all",):
            r[key][0] += found; r[key][1] += len(boxes); r[key][2] += (n - 1) - len(used)
    net.train()
    return {k: (100 * v[0] / max(v[1], 1), 100 * v[2] / max(v[1], 1), v[1]) for k, v in r.items()}


if __name__ == "__main__":
    torch.set_num_threads(4); cv2.setNumThreads(2)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(min(a.gpu_gb / (torch.cuda.get_device_properties(0).total_memory / 2**30), 0.95))
    train = sorted(f[:-4] for f in glob.glob(f"{a.data}/train/*.png") if f.count(".") == 1)
    if a.only_cleaned:
        train = [f for f in train if os.path.exists(f + ".cleanup.png")]
    meta = {m["page"]: m["table"] for m in json.load(open(f"{a.data}/val_pages.json"))}
    rng = np.random.default_rng(0)
    vnames = sorted(meta); rng.shuffle(vnames)
    val = [(f"{a.data}/val/{p}", meta[p] >= 0.3) for p in vnames[: a.eval_pages]]
    print(len(train), "training pages,", len(meta), "validation pages (", len(val), "scored during training,", sum(t for _, t in val), "table-heavy)", flush=True)
    dl = torch.utils.data.DataLoader(Pages(train, a.steps * a.batch), batch_size=a.batch, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    net = UNet(a.base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, base {a.base}, split_gap {a.split_gap}, rule_cut {a.rule_cut}", flush=True)
    t0 = time.time(); best = -1
    for step, (xs, ys) in enumerate(dl, 1):
        x = xs.to(dev, non_blocking=True).float()[:, None]; y = (ys.to(dev) > 0).float()[:, None]
        B = (x.shape[0], 1, 1, 1); r = lambda: torch.rand(B, device=dev)
        x = torch.where(r() < 0.5, 255 - (255 - x) * (0.6 + 0.7 * r()), x)
        x = torch.where(r() < 0.4, x + torch.randn_like(x) * (2 + 10 * r()), x)
        x = torch.where(r() < 0.3, x - 60 * r(), x)
        x = x.clamp(0, 255) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x)
        logit = logit.float(); p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(logit, y) + dice
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0:
            print(f"step {step}/{a.steps} loss {loss.item():.4f} {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ck = {"model": net.state_dict(), **net.config, "step": step, "split_gap": a.split_gap, "rule_cut": a.rule_cut, "input": a.input}
            torch.save(ck, a.out.replace(".pt", f"_s{step}.pt"))
            torch.save(ck, a.out)
            ev = evaluate(net, val)
            print(f"== step {step}: unseen-journal validation one-to-one {ev['all'][0]:.2f}% of {ev['all'][2]} lines, extra {ev['all'][1]:.2f}% | "
                  f"table-heavy {ev['table'][0]:.2f}% of {ev['table'][2]}, extra {ev['table'][1]:.2f}%", flush=True)
        if step >= a.steps:
            break
