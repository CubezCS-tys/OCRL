"""Train the structure model on the Haiku-labelled pages (bulk and desk sets), optionally after pre-training on the
Gemini-labelled Desk pages with their labels cleaned, and measure it on held-out labelled documents.

Input: a pack made by scripts/pack_structure_data.py (Azure's lines, page size and labels; no pictures), and for
pre-training data/structure/desk_lines.pt (scripts/build_structure_data.py). The v3 recipe is the default:
start from v2, a new output per block type started from the Desk roles mapped to it, 16 epochs, batch 24, lr 4e-4,
type weights (most common / count) ** 0.3 capped at 6, start loss weight 1, best epoch by held-out line types.

Held out: about 400 pages, whole documents, drawn only from documents the bulk set (v3's training) never saw, so
the deployed pair can be scored on them fairly. They are also kept out of pre-training, as are the 70 judged
pages' documents and the keyed documents.

    python scripts/train_structure_v5.py --name v5a --out out/v5a.pt                       # v3 recipe, all pages
    python scripts/train_structure_v5.py --name gem --pretrain-only --epochs 4 --out out/gem.pt   # Gemini, cleaned
    python scripts/train_structure_v5.py --name v5g --base out/gem.pt --out out/v5g.pt     # then the Haiku pages
"""
import argparse, gzip, json, os, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ocrl.structure.features import FIRST, N_GEOM, ROLES, geometry, text_ids
from ocrl.structure.model import LineTagger, load, save
from ocrl.structure.tag import TO_TYPE, TYPES, _ITEM, _as_types
from ocrl.text import strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--out", default="")
ap.add_argument("--pack", default="data/pack.json.gz"); ap.add_argument("--gemini", default="data/desk_lines.pt")
ap.add_argument("--base", default="data/structure_v2.pt"); ap.add_argument("--v3", default="data/structure_v3.pt")
ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--batch", type=int, default=24); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--fraction", type=float, default=1.0, help="share of the training documents (learning curve)")
ap.add_argument("--sets", default="bulk,desk", help="which Haiku sets to train on")
ap.add_argument("--start-weight", type=float, default=1.0); ap.add_argument("--type-power", type=float, default=0.3); ap.add_argument("--type-cap", type=float, default=6.0)
ap.add_argument("--select", default="types", choices=["types", "both"], help="best epoch by held-out line types, or by types + exact blocks")
ap.add_argument("--pretrain-only", action="store_true", help="train on the cleaned Gemini pages only (no Haiku pages)")
ap.add_argument("--raw-gemini", action="store_true", help="pre-train on Gemini labels only mapped, not cleaned")
ap.add_argument("--mix-gemini", type=float, default=0.0, help="also train on this share of Gemini pages per epoch, next to the Haiku pages")
ap.add_argument("--haiku-start-w", type=float, default=1.0, help="weight of Haiku pages' block starts (with --mix-gemini: 0 = starts from Gemini only)")
ap.add_argument("--gem-type-w", type=float, default=1.0, help="weight of mixed-in Gemini pages' line types")
ap.add_argument("--d", type=int, default=0); ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--baseline", action="store_true", help="also score the deployed pair (v2 starts, 0.3/0.7 types) and v3 on the held-out pages")
ap.add_argument("--gpu-frac", type=float, default=0.22); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
if dev_ == "cuda":
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
TI = {t: i for i, t in enumerate(TYPES)}
WEAK = ["title", "author", "affiliation", "list_item", "reference", "heading", "footnote", "table", "figure", "verse", "page_number", "running_head"]
base, vocab = load(a.base); base.to(dev_)
pack = json.load(gzip.open(a.pack, "rt", encoding="utf-8"))
OUT = Path("out"); OUT.mkdir(exist_ok=True)
log = open(f"logs/{a.name}.log", "a")


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()


def features(p):
    lines = p["lines"]; W, H = p["W"], p["H"]
    boxes = np.array([[l["box"][0] * W / 1000, l["box"][1] * H / 1000, l["box"][2] * W / 1000, l["box"][3] * H / 1000] for l in lines], np.float32).reshape(-1, 4)
    texts = [l["text"] for l in lines]
    return {"geom": torch.from_numpy(geometry(boxes, texts, W, H)), "text": torch.from_numpy(np.stack([text_ids(t, vocab) for t in texts])).long(),
            "ids": [l["id"] for l in lines], "texts": texts, "name": p["page"], "doc": p["doc"]}


def with_answer(f, ans):
    block = {r: k for k, b in enumerate(ans["blocks"]) for r in b["lines"]}; typ = {r: b["type"] for b in ans["blocks"] for r in b["lines"]}
    f = dict(f)
    f["type"] = torch.tensor([TI.get(typ.get(i, "other"), TI["other"]) for i in f["ids"]])
    bl = [block.get(i, -1) for i in f["ids"]]
    f["start"] = torch.tensor([1.0] + [float(x != y) for x, y in zip(bl[1:], bl[:-1])])
    f["block"] = bl
    return f


# --- the Haiku pages: held-out documents only from those the bulk set never had
pages = [p for p in pack["haiku"] if len(p["lines"])]
bulk_docs = {p["doc"] for p in pages if "bulk" in p}
cand = sorted({p["doc"] for p in pages} - bulk_docs)
per_doc = Counter(p["doc"] for p in pages)
held, n = set(), 0
for d in np.random.default_rng(7).permutation(cand):
    if n >= 400: break
    held.add(str(d)); n += per_doc[d]
sets = a.sets.split(",")
train_all, val = [], []
for p in pages:
    f = features(p)
    if p["doc"] in held:
        val.append(with_answer(f, p["desk"]))
    else:
        ans = next((p[s] for s in sets if s in p), None)  # a page in both sets: the bulk answer (as v3 saw it)
        if ans is not None: train_all.append(with_answer(f, ans))
tdocs = sorted({f["doc"] for f in train_all})
keep = set(rng.permutation(tdocs)[: max(int(round(a.fraction * len(tdocs))), 1)])
train = [f for f in train_all if f["doc"] in keep] if not a.pretrain_only else []
json.dump(sorted(held), open(f"out/{a.name}.held_docs.json", "w"))


def judged():
    out = []
    for p in pack["dev"]:
        f = with_answer(features(p), p["opus"])
        T = [{r: b["type"] for b in p[w]["blocks"] for r in b["lines"]} for w in ("opus", "sonnet", "argon")]
        maj = []
        for i in f["ids"]:
            t, k = Counter(x.get(i) for x in T).most_common(1)[0]
            maj.append(TI[t] if k >= 2 and t in TI else -100)
        f["type"] = torch.tensor(maj); out.append(f)
    return out


test = judged()
say(f"[{a.name}] {len(train)} training pages ({len(keep)} of {len(tdocs)} documents), {len(val)} held-out pages from {len(held)} documents, "
    f"{sum(len(f['ids']) for f in train):,} training lines; sets {sets}")


# --- the Gemini pages, cleaned
def gemini(exclude):
    g = torch.load(a.gemini, weights_only=False)
    inv = {i: c for c, i in g["vocab"].items()}
    off = g["offsets"].tolist(); role = g["role"].numpy(); start = g["start"].numpy().astype(np.float32); text = g["text"].numpy()
    num_only = re.compile(r"^[\s0-9٠-٩۰-۹\-–—()/.\[\]|«»]+$")
    kept, skipped, changed = [], Counter(), Counter()
    for k, (stem, page, run) in enumerate(g["meta"]):
        if stem in exclude:
            skipped["excluded"] += 1; continue
        a0, a1 = off[k], off[k + 1]
        r = [ROLES[x] for x in role[a0:a1]]; s = start[a0:a1].copy()
        firsts = ["".join(inv.get(int(c), "?") for c in row[:FIRST] if c > 0) for row in text[a0:a1]]
        typ = []
        for i, x in enumerate(r):
            t = TO_TYPE.get(x, "other")
            if not a.raw_gemini:
                if x == "quote": t = "paragraph"; changed["quote"] += 1
                elif x == "front-matter": t = None; changed["front-matter ignored"] += 1
                elif x == "caption":
                    near = set(r[max(i - 4, 0): i + 5])
                    t = "table_caption" if "table-cell" in near else "figure_caption" if "pictureText" in near else None
                    changed[f"caption->{t}"] += 1
                elif x in ("pageHeader", "pageFooter") and num_only.match(firsts[i]) and any(ch.isdigit() for ch in firsts[i]):
                    t = "page_number"; changed["number-only furniture"] += 1
            typ.append(TI[t] if t is not None else -100)
        if not a.raw_gemini:
            for i in range(1, len(r)):  # one block per table, poem, figure
                if r[i] in ("table-cell", "verse-group", "pictureText") and r[i - 1] == r[i] and s[i] == 1:
                    s[i] = 0; changed[f"merged {r[i]}"] += 1
            cuts = [i for i in range(len(r)) if s[i] >= 0.5] + [len(r)]
            for b0, b1 in zip(cuts, cuts[1:]):  # a paragraph whose block opens with an item mark is a list item
                if r[b0] == "paragraph" and _ITEM.match(strip_marks(firsts[b0])):
                    for i in range(b0, b1):
                        if r[i] == "paragraph": typ[i] = TI["list_item"]
                    changed["item-mark paragraph blocks"] += 1
        kept.append({"geom": g["geom"][a0:a1], "text": g["text"][a0:a1].long(), "type": torch.tensor(typ), "start": torch.from_numpy(s),
                     "ids": list(range(a1 - a0)), "doc": stem, "gem": True, "block": (np.cumsum(s >= 0.5) - 1).tolist()})
    say(f"Gemini pages: {len(kept)} kept, {dict(skipped)}; label changes: {dict(changed)}")
    return kept


gem = []
if a.pretrain_only or a.mix_gemini:
    exclude = held | {f["doc"] for f in test} | {"0013-002-001-005", "0073-000-239-002", "0175-000-004-016", "0499-026-119-022"}
    gem = gemini(exclude)
    if a.pretrain_only: train = gem


def batch(ps, trn=False):
    n = min(max(len(p["ids"]) for p in ps), 320)
    g = torch.zeros(len(ps), n, N_GEOM); t = torch.zeros(len(ps), n, ps[0]["text"].shape[1], dtype=torch.long)
    y = torch.full((len(ps), n), -100, dtype=torch.long); s = torch.zeros(len(ps), n); pad = torch.ones(len(ps), n, dtype=torch.bool)
    for k, p in enumerate(ps):
        m = min(len(p["ids"]), n)
        g[k, :m] = p["geom"][:m, :N_GEOM]; t[k, :m] = p["text"][:m]; y[k, :m] = p["type"][:m]; s[k, :m] = p["start"][:m]; pad[k, :m] = False
    if trn:
        g += torch.randn_like(g) * 0.006
        noise = (torch.rand(t.shape) < 0.05) & (t > 0); t = torch.where(noise, torch.randint(2, len(vocab) + 2, t.shape), t)
    return g.to(dev_), t.to(dev_), y.to(dev_), s.to(dev_), pad.to(dev_)


def blocks_score(ref_blocks, cut_starts):
    """Reference blocks (lists of line positions) reproduced exactly / split / merged, given predicted starts."""
    pred = np.cumsum(cut_starts)
    ok = split = merged = 0
    size = Counter(pred.tolist())
    for b in ref_blocks:
        ids = {int(pred[i]) for i in b}
        if len(ids) > 1: split += 1
        elif size[next(iter(ids))] > len(b): merged += 1
        else: ok += 1
    return ok, split, merged


@torch.no_grad()
def outputs(model, f):
    g, t, y, s, pad = batch([f])
    lr, ls = model(g, t, pad)
    return lr[0].softmax(-1).float().cpu().numpy(), torch.sigmoid(ls[0]).float().cpu().numpy()


@torch.no_grad()
def evaluate(get, pages, judged_set=False):
    """`get(f)` -> (P over TYPES [n, 27], P(start) [n]). Line types, weak types, blocks against the page's answer."""
    ok = n = 0; per = defaultdict(Counter); B = Counter(); loss = []
    for f in pages:
        q, st = get(f)
        m = min(len(f["ids"]), q.shape[0])
        got = q[:m].argmax(1); want = f["type"][:m].numpy()
        for k in range(m):
            if want[k] < 0: continue
            n += 1; ok += got[k] == want[k]; per[TYPES[want[k]]]["n"] += 1; per[TYPES[want[k]]]["ok"] += got[k] == want[k]
            per[TYPES[got[k]]]["pred"] += 1
        if not judged_set:
            cut = (st[:m] >= 0.5).astype(int); cut[0] = 1
            bl = f["block"][:m]; groups = defaultdict(list)
            for i, b in enumerate(bl): groups[b].append(i)
            refs = [v for k2, v in groups.items() if k2 >= 0 and v == list(range(v[0], v[-1] + 1))]  # contiguous reference blocks
            o, sp, me = blocks_score(refs, cut); B["ok"] += o; B["split"] += sp; B["merged"] += me; B["n"] += len(refs)
            ws = want >= 0
            loss.append(float(F.nll_loss(torch.log(torch.from_numpy(q[:m][ws]).clamp_min(1e-8)), torch.from_numpy(want[ws]).long(), reduction="sum")) / max(ws.sum(), 1))
    r = {"types": 100 * ok / max(n, 1), "lines": n, "weak": {t: round(100 * per[t]["ok"] / per[t]["n"], 1) for t in WEAK if per[t]["n"]},
         "weak_n": {t: per[t]["n"] for t in WEAK if per[t]["n"]}}
    if not judged_set:
        r.update({"blocks_exact": 100 * B["ok"] / max(B["n"], 1), "split": 100 * B["split"] / max(B["n"], 1), "merged": 100 * B["merged"] / max(B["n"], 1),
                  "type_nll": float(np.mean(loss))})
    return r


def as_types(model):
    def get(f):
        q, st = outputs(model, f)
        return (q if model.labels == TYPES else _as_types(q, model.labels)), st
    return get


def fmt(r):
    s = f"types {r['types']:.1f}"
    if "blocks_exact" in r: s += f" | blocks exact {r['blocks_exact']:.1f} split {r['split']:.1f} merged {r['merged']:.1f} | type nll {r['type_nll']:.3f}"
    return s + " | " + " ".join(f"{k} {v:.0f}" for k, v in r["weak"].items())


if a.baseline:
    v3, _ = load(a.v3); v3.to(dev_); g2, g3 = as_types(base), as_types(v3)
    pair = lambda f: (lambda A, B_: (0.3 * A[0] + 0.7 * B_[0], A[1]))(g2(f), g3(f))
    res = {"pair": evaluate(pair, val), "v3": evaluate(g3, val), "v2": evaluate(g2, val),
           "pair_judged": evaluate(pair, test, True), "v3_judged": evaluate(g3, test, True), "v2_judged": evaluate(g2, test, True)}
    for k, r in res.items(): say(f"baseline {k}: {fmt(r)}")
    say("held-out weak-type line counts:", res["pair"]["weak_n"])
    json.dump(res, open(f"out/{a.name}.baseline.json", "w"), indent=1)
    if a.epochs == 0: raise SystemExit

# --- the model
fresh = bool(a.d and (a.d != base.config["d"] or a.layers != base.config["layers"]))
model = LineTagger(base.config["n_vocab"], a.d or base.config["d"], a.layers or base.config["layers"], 8, len(TYPES), N_GEOM)
if not fresh:
    if base.labels == TYPES:
        model.load_state_dict(base.state_dict())
    else:
        model.load_state_dict({k: v for k, v in base.state_dict().items() if not k.startswith("role.")}, strict=False)
        with torch.no_grad():
            for i, t in enumerate(TYPES):
                src = [j for j, r in enumerate(ROLES) if TO_TYPE.get(r) == t]
                if src: model.role.weight[i] = base.role.weight[src].mean(0); model.role.bias[i] = base.role.bias[src].mean()
                else: model.role.weight[i].zero_(); model.role.bias[i] = base.role.bias.min()
model.to(dev_); model.labels = TYPES
say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M weights, d {model.config['d']} layers {model.config['layers']} | from {'nothing' if fresh else a.base} | device {dev_}")
count = Counter(int(x) for p in train for x in p["type"] if x >= 0)
w = torch.tensor([min((max(count.values()) / max(count[i], 1)) ** a.type_power, a.type_cap) for i in range(len(TYPES))]).to(dev_)
say("labels in training:", ", ".join(f"{TYPES[i]} {k}" for i, k in count.most_common(14)))
n_mix = int(a.mix_gemini * len(train)) if gem and not a.pretrain_only else 0
steps = a.epochs * ((len(train) + n_mix) // a.batch)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
t0, best, hist = time.time(), (-1.0, None, None), []
for ep in range(a.epochs):
    pool = train + ([gem[i] for i in rng.choice(len(gem), n_mix, replace=False)] if n_mix else [])
    order = sorted(rng.permutation(len(pool)), key=lambda i: len(pool[i]["ids"]) // 12 + rng.random())
    bs = [order[i: i + a.batch] for i in range(0, len(order) - a.batch + 1, a.batch)]
    tl = []
    for bi in rng.permutation(len(bs)):
        g, t, y, s, pad = batch([pool[i] for i in bs[bi]], trn=True)
        lr, ls = model(g, t, pad)
        keep_y = y[~pad]
        src = torch.tensor([bool(pool[i].get("gem")) for i in bs[bi]], device=dev_)[:, None].expand_as(pad)[~pad]
        if n_mix:  # Haiku and Gemini pages weighted apart: types and starts from either or both
            ce = F.cross_entropy(lr[~pad], keep_y, weight=w, ignore_index=-100, reduction="none")
            wy = w[keep_y.clamp_min(0)] * (keep_y >= 0) * torch.where(src, a.gem_type_w, 1.0)
            lt = (ce * torch.where(src, a.gem_type_w, 1.0)).sum() / wy.sum().clamp_min(1e-6)
            ws = torch.where(src, 1.0, a.haiku_start_w)
            bce = F.binary_cross_entropy_with_logits(ls[~pad], s[~pad], reduction="none")
            loss = lt + a.start_weight * (bce * ws).sum() / ws.sum().clamp_min(1e-6)
        else:
            lt = F.cross_entropy(lr[~pad], keep_y, weight=w, ignore_index=-100)
            loss = lt + a.start_weight * F.binary_cross_entropy_with_logits(ls[~pad], s[~pad])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl.append(lt.item())
    model.eval()
    rv = evaluate(as_types(model), val) if val else {}; rj = evaluate(as_types(model), test, True)
    # train-side type nll on a fixed sample of training pages, unweighted, to compare with held-out (under- or over-fitting)
    rt = evaluate(as_types(model), [train[i] for i in np.random.default_rng(1).choice(len(train), min(300, len(train)), replace=False)]) if train else {}
    model.train()
    key = rv["types"] if a.select == "types" else rv["types"] + rv["blocks_exact"]
    hist.append({"epoch": ep + 1, "train_type_loss": float(np.mean(tl)), "val": rv, "judged": rj, "train_sample": {k: rt.get(k) for k in ("types", "blocks_exact", "type_nll")}})
    say(f"epoch {ep + 1}/{a.epochs} train type loss {np.mean(tl):.3f} | train sample types {rt.get('types', 0):.1f} nll {rt.get('type_nll', 0):.3f} | "
        f"held-out {fmt(rv)} | judged {rj['types']:.1f} | {time.time() - t0:.0f} s")
    if key > best[0]:
        best = (key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
model.load_state_dict(best[1]); model.eval()
out = a.out or f"out/{a.name}.pt"
save(model.cpu(), vocab, out, labels=TYPES); model.to(dev_)
rv = evaluate(as_types(model), val); rj = evaluate(as_types(model), test, True)
say(f"BEST epoch {best[2]}: held-out {fmt(rv)} | judged {fmt(rj)} -> {out}")
json.dump({"args": vars(a), "best_epoch": best[2], "val": rv, "judged": rj, "history": hist, "n_train": len(train)}, open(f"out/{a.name}.json", "w"), indent=1)
