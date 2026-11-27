"""Retrain the structure model on labels made by a labelling model, and score it on the judged development pages.

Starts from an existing structure model and gives it a new output for each of the owner's block types (each
started from the Desk roles that map to it). Training pages are labelled pages of Azure's lines; the test is the
70 census pages where three independent models (Opus, Sonnet, Gemini 4 Argon) were asked, taking the type at
least two of them chose.

    uv run python scripts/finetune_structure.py --labels out/labeller/bulk_haiku --epochs 16 --out out/structure_v3.pt
"""
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cv2

from ocrl.structure.features import N_GEOM, N_INK, ROLES, geometry, ink_features, line_crops, text_ids
from ocrl.structure.model import LineTagger, load, save
from ocrl.structure.tag import TO_TYPE, TYPES, list_items

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="out/structure_v2.pt"); ap.add_argument("--labels", default="out/labeller/bulk_haiku"); ap.add_argument("--inputs", default="out/labeller/bulk_inputs")
ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--batch", type=int, default=24); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--out", default="out/structure_v3.pt"); ap.add_argument("--only-score", action="store_true"); ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--fraction", type=float, default=1.0, help="use only this share of the training documents (for a learning curve)")
ap.add_argument("--ink", action="store_true", help="also give the model each line's ink measurements (stroke weight, density, boldness for size)")
ap.add_argument("--visual", action="store_true", help="give the model a small picture of every line")
ap.add_argument("--d", type=int, default=0, help="width of a bigger model trained from nothing (0: keep the base model's size and weights)")
ap.add_argument("--layers", type=int, default=0); ap.add_argument("--limit", type=int, default=0, help="load only this many labelled pages (to try the script)")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
L = Path("out/labeller"); TI = {t: i for i, t in enumerate(TYPES)}
base, vocab = load(a.base); base.to(dev_)


def page(folder: Path, answer: dict | None):
    """Features of one page in Azure's line order; with an answer, also each line's type and whether it opens a block."""
    lines = json.load(open(folder / "lines.json", encoding="utf-8"))
    img = cv2.imread(str(next(folder.glob("page_*.jpg"))))[:, :-70]  # without the strip added for the line numbers
    H, W = img.shape[:2]
    # the scan is grey; whatever is coloured is the outline or number drawn for the labellers: paint it out
    gray = np.where(img.max(2).astype(int) - img.min(2) > 45, 255, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.uint8)
    boxes = np.array([[l["box"][0] * W / 1000, l["box"][1] * H / 1000, l["box"][2] * W / 1000, l["box"][3] * H / 1000] for l in lines], np.float32)
    texts = [l["text"] for l in lines]
    out = {"geom": torch.from_numpy(geometry(boxes, texts, W, H, ink_features(gray, boxes) if a.ink else None)), "text": torch.from_numpy(np.stack([text_ids(t, vocab) for t in texts])).long(),
           "boxes": boxes, "texts": texts, "W": W, "H": H, "ids": [l["id"] for l in lines]}
    if a.visual:
        out["crops"] = torch.from_numpy(line_crops(gray, boxes))
    if answer is not None:
        block = {r: k for k, b in enumerate(answer["blocks"]) for r in b["lines"]}; typ = {r: b["type"] for b in answer["blocks"] for r in b["lines"]}
        out["type"] = torch.tensor([TI.get(typ.get(i, "other"), TI["other"]) for i in out["ids"]])
        bl = [block.get(i, -1) for i in out["ids"]]
        out["start"] = torch.tensor([1.0] + [float(x != y) for x, y in zip(bl[1:], bl[:-1])])
    return out


def judged():
    """The test pages: for each line, the type at least two of the three judges gave (lines with no majority are left out)."""
    pages = []
    for folder in sorted((L / "dev_inputs").iterdir()):
        J = [json.load(open(L / f"dev_{w}" / f"{folder.name}.json", encoding="utf-8")) for w in ("opus", "sonnet", "argon")]
        T = [{r: b["type"] for b in j["blocks"] for r in b["lines"]} for j in J]
        p = page(folder, J[0])  # block starts from Opus, the judge closest to the other two on blocks
        maj = []
        for i in p["ids"]:
            t, n = Counter(x.get(i) for x in T).most_common(1)[0]
            maj.append(TI[t] if n >= 2 and t in TI else -100)
        p["type"] = torch.tensor(maj); p["name"] = folder.name; pages.append(p)
    return pages


def batch(ps, train=False, cols=None):
    n = min(max(len(p["ids"]) for p in ps), 320)
    g = torch.zeros(len(ps), n, cols or ps[0]["geom"].shape[1]); t = torch.zeros(len(ps), n, ps[0]["text"].shape[1], dtype=torch.long)
    y = torch.full((len(ps), n), -100, dtype=torch.long); s = torch.zeros(len(ps), n); pad = torch.ones(len(ps), n, dtype=torch.bool)
    for k, p in enumerate(ps):
        m = min(len(p["ids"]), n)
        g[k, :m] = p["geom"][:m, : g.shape[2]]; t[k, :m] = p["text"][:m]; y[k, :m] = p["type"][:m]; s[k, :m] = p["start"][:m]; pad[k, :m] = False
    if train:
        g += torch.randn_like(g) * 0.006
        noise = (torch.rand(t.shape) < 0.05) & (t > 0); t = torch.where(noise, torch.randint(2, len(vocab) + 2, t.shape), t)
    c = None
    if a.visual and "crops" in ps[0]:
        c = torch.full((len(ps), n, *ps[0]["crops"].shape[1:]), 255, dtype=torch.uint8)
        for k, p in enumerate(ps):
            m = min(len(p["ids"]), n); c[k, :m] = p["crops"][:m]
        c = c.to(dev_)
    return g.to(dev_), t.to(dev_), y.to(dev_), s.to(dev_), pad.to(dev_), c


@torch.no_grad()
def score(model, pages, labels, rule: bool):
    """Line types right against the judges, overall and for the worst types; with the list rule on top or not."""
    model.eval(); ok = n = 0; miss = Counter(); per = defaultdict(Counter)
    for p in pages:
        g, t, y, s, pad, c = batch([p], cols=model.config["n_geom"])
        lr, ls = model(g, t, pad, c) if model.config.get("visual") else model(g, t, pad)
        roles = [labels[i] for i in lr[0].argmax(-1).tolist()]; starts = torch.sigmoid(ls[0]).float().cpu().numpy()
        if rule:
            roles, starts, _ = list_items(roles, starts, p["texts"], p["boxes"])
        for k, r in enumerate(roles[: len(p["ids"])]):
            want = int(p["type"][k])
            if want < 0: continue
            got = TO_TYPE.get(r, "other"); n += 1; ok += got == TYPES[want]; per[TYPES[want]]["n"] += 1; per[TYPES[want]]["ok"] += got == TYPES[want]
            if got != TYPES[want]: miss[TYPES[want], got] += 1
    model.train()
    return ok / max(n, 1), miss, per


test = judged()
acc, miss, _ = score(base, test, base.labels, False); acc_r, miss_r, _ = score(base, test, base.labels, True)
print(f"before, on the 70 judged pages: {100 * acc:.1f}% of line types right; with the list rule {100 * acc_r:.1f}% ({sum(len(p['ids']) for p in test)} lines)", flush=True)
if a.only_score: raise SystemExit
folders = sorted(p for p in Path(a.inputs).iterdir() if (Path(a.labels) / f"{p.name}.json").exists())
if a.limit: folders = folders[: a.limit]
data = []
for f in folders:
    try: data.append((f.name.rsplit("_p", 1)[0], page(f, json.load(open(Path(a.labels) / f"{f.name}.json", encoding="utf-8")))))
    except Exception: pass
docs = sorted({d for d, _ in data}); held = set(rng.choice(docs, max(len(docs) // 12, 1), replace=False))
keep_docs = set(rng.permutation([d for d in docs if d not in held])[: max(int(a.fraction * (len(docs) - len(held))), 1)])
train = [p for d, p in data if d in keep_docs]; dev = [p for d, p in data if d in held]
print(f"{len(train)} training pages, {len(dev)} held-out pages from {len(held)} documents, {sum(len(p['ids']) for p in train):,} training lines", flush=True)
print("labels in training:", ", ".join(f"{TYPES[i]} {n}" for i, n in Counter(int(x) for p in train for x in p["type"]).most_common(12)), flush=True)
n_geom = N_GEOM + (N_INK if a.ink else 0)
fresh = bool(a.d and (a.d != base.config["d"] or a.layers != base.config["layers"]))
model = LineTagger(base.config["n_vocab"], a.d or base.config["d"], a.layers or base.config["layers"], (8 if (a.d or 0) >= 256 else base.config["heads"]), len(TYPES), n_geom, a.visual)
if not fresh:
    skip = ("role.", "geom.0.weight", "inp.weight")
    model.load_state_dict({k: v for k, v in base.state_dict().items() if not k.startswith(skip)}, strict=False)
    with torch.no_grad():  # what the old model knew keeps its weights; the new inputs (ink numbers, the picture) start silent
        model.geom[0].weight.zero_(); model.geom[0].weight[:, :N_GEOM] = base.geom[0].weight
        model.inp.weight.zero_(); model.inp.weight[:, : base.inp.weight.shape[1]] = base.inp.weight
model.to(dev_); base.to(dev_)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M weights | looks at line pictures: {a.visual} | starts from the earlier model: {not fresh} | device {dev_}", flush=True)
with torch.no_grad():  # each block type starts from the Desk roles that were mapped to it
    for i, t in enumerate(TYPES if not fresh else []):
        src = [j for j, r in enumerate(ROLES) if TO_TYPE.get(r) == t]
        if src: model.role.weight[i] = base.role.weight[src].mean(0); model.role.bias[i] = base.role.bias[src].mean()
        else: model.role.weight[i].zero_(); model.role.bias[i] = base.role.bias.min()
steps = a.epochs * (len(train) // a.batch)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
count = Counter(int(x) for p in train for x in p["type"]); w = torch.tensor([min((max(count.values()) / max(count[i], 1)) ** 0.3, 6.0) for i in range(len(TYPES))]).to(dev_)
t0, best = time.time(), (0.0, None)
for ep in range(a.epochs):
    order = sorted(rng.permutation(len(train)), key=lambda i: len(train[i]["ids"]) // 12 + rng.random())
    bs = [order[i : i + a.batch] for i in range(0, len(order) - a.batch + 1, a.batch)]
    for bi in rng.permutation(len(bs)):
        g, t, y, s, pad, c = batch([train[i] for i in bs[bi]], train=True)
        lr, ls = model(g, t, pad, c) if a.visual else model(g, t, pad)
        loss = F.cross_entropy(lr[~pad], y[~pad], weight=w) + F.binary_cross_entropy_with_logits(ls[~pad], s[~pad])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    d_acc, _, _ = score(model, dev, TYPES, False); j_acc, j_miss, _ = score(model, test, TYPES, False)
    print(f"epoch {ep + 1}/{a.epochs}: loss {loss.item():.3f} | held-out labelled pages {100 * d_acc:.1f}% | judged pages {100 * j_acc:.1f}% | {time.time() - t0:.0f} s", flush=True)
    if d_acc > best[0]:  # chosen on the held-out labelled pages, never on the judged ones
        best = (d_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
model.load_state_dict(best[1]); model.to(dev_); save(model.cpu(), vocab, a.out, labels=TYPES); model.to(dev_)
acc, miss, per = score(model, test, TYPES, False); acc_r, _, _ = score(model, test, TYPES, True)
print(f"after, on the 70 judged pages: {100 * acc:.1f}% of line types right; with the list rule {100 * acc_r:.1f}%")
print("still wrong most (judges say / it says):", ", ".join(f"{k}/{m} ×{n}" for (k, m), n in miss.most_common(8)))
print("by type:", ", ".join(f"{t} {100 * c['ok'] / c['n']:.0f}% of {c['n']}" for t, c in sorted(per.items(), key=lambda kv: -kv[1]['n'])[:14]))
