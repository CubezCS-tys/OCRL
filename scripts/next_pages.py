"""The next round of labelling pages: kinds and a pull-quote marker for candidate documents, a choice by quotas, inputs.

Runs where the candidate documents are (Azure JSON first; PDFs only for the chosen documents). Every page of the first
40 of a document with 3 to 400 lines gets its census kind (scripts/census.py page_row, kind_centres.npz, as
scripts/fresh_pages.py does) and a pull-quote marker from Azure's lines alone: a run of 1-8 lines in type at least
1.3x the page's median height set inside the page (not the top 15% or bottom 8%), or a run of five words that is
repeated three or more lines away (pull quotes repeat a sentence of the article). On labelled pages the marker finds
about half of the pull-quote pages at about 1.5x the base rate: weak, so it only orders pages within a kind.

Pages are taken by quota per kind, rarest first, at most two a document and a few a journal (pull-quote journals a
few more), in a seeded random order. A second run with --have counts the pages already chosen against the quotas
and caps, so a first half can be drawn and brought home while the rest of the candidates arrive.

    python next_pages.py kinds --docs docs --cache kinds.jsonl --workers 16
    python next_pages.py select --cache kinds.jsonl --cands candidates.json --scale 0.5 --out pages_a.json
    python next_pages.py select --cache kinds.jsonl --cands candidates.json --have pages_a.json --out pages_b.json
    python next_pages.py draw --docs docs --pages pages_a.json --dest inputs --workers 16
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEED = 29
WEAK = {"Title and cover pages": 450, "Article first page": 500, "Two dense columns and magazine pages": 800, "Two columns": 800,
        "Mixed: tables, charts, shaded cells": 550, "Tables of numbers": 450, "Small text block": 450}   # 4,000
SPREAD = 4000   # over all 14 kinds by their census share
PQ_KINDS = {"Two dense columns and magazine pages", "Two columns"}   # 4-5% pull-quote pages among Haiku's fresh labels
CAP_DOC, CAP_J, CAP_J_PQ = 2, 4, 6
W = re.compile(r"[\w؀-ۿ]{3,}")


def pq_marker(boxes, texts):
    b = np.array(boxes, float); h = np.maximum(b[:, 3] - b[:, 1], 1e-6); top = b[:, 1].min(); H = b[:, 3].max() - top + 1e-6
    hm, n = np.median(h), len(b); big = h >= 1.3 * hm; i = 0; hit = False
    while i < n and not hit:
        if big[i]:
            j = i
            while j < n and big[j]: j += 1
            yc = ((b[i:j, 1] + b[i:j, 3]) / 2 - top) / H
            hit = 1 <= j - i <= 8 and i >= 3 and j <= n - 3 and yc.min() > 0.15 and yc.max() < 0.92
            i = j
        else: i += 1
    if hit: return "big type run"
    toks = [(w, k) for k, t in enumerate(texts) for w in W.findall(t)]; seen = {}
    for i in range(len(toks) - 4):
        g = tuple(w for w, _ in toks[i:i + 5]); k = toks[i][1]
        if g in seen and abs(k - seen[g]) >= 3: return "repeated sentence"
        seen.setdefault(g, k)
    return ""


def kinds_of(args):
    src, stem = args
    from census import page_row
    z = np.load(HERE / "kind_centres.npz"); lo, hi, mean, std, C = z["lo"], z["hi"], z["mean"], z["std"], z["C"]
    try:
        d = json.loads((Path(src) / stem / f"{stem}.json").read_text(encoding="utf-8")); a = d.get("analyzeResult", d)
        hand = [(s["offset"], s["offset"] + s["length"]) for st in a.get("styles", []) if st.get("isHandwritten") for s in st.get("spans", [])]
        par = Counter(p["boundingRegions"][0]["pageNumber"] for p in a.get("paragraphs", []) if p.get("boundingRegions"))
        out = []
        for pg in a["pages"][:40]:
            ls = pg.get("lines", []); n = len(ls)
            if not (3 <= n <= 400) or abs(pg.get("angle") or 0) > 20:
                continue
            r = page_row(pg, len(a["pages"]), hand, par.get(pg["pageNumber"], 0))
            if not r:
                continue
            v = np.array(r, np.float32); v[0] = np.log1p(v[0]); v[1] = np.log1p(v[1]); v = (np.clip(v, lo, hi) - mean) / std
            q = [l["polygon"] for l in ls]
            boxes = [[min(p[0::2]), min(p[1::2]), max(p[0::2]), max(p[1::2])] for p in q]
            out.append([pg["pageNumber"], n, int(((C - v) ** 2).sum(1).argmin()), pq_marker(boxes, [l["content"] for l in ls])])
        return stem, out, len(a["pages"])
    except Exception as e:
        return stem, None, f"{type(e).__name__}"


def kinds(a):
    cache = Path(a.cache); have = set()
    if cache.exists():
        have = {json.loads(l)["doc"] for l in open(cache)}
    import time
    now = time.time()   # a file still being downloaded is left for the next run
    todo = sorted(p.name for p in Path(a.docs).iterdir() if p.name not in have and (p / f"{p.name}.json").exists()
                  and now - (p / f"{p.name}.json").stat().st_mtime > 60)
    print(f"{len(todo)} new documents ({len(have)} cached)", flush=True)
    with Pool(a.workers, maxtasksperchild=100) as pool, open(cache, "a") as f:
        for stem, rows, info in pool.imap_unordered(kinds_of, [(a.docs, d) for d in todo], chunksize=4):
            if rows is not None:   # a failed read is tried again next run
                f.write(json.dumps({"doc": stem, "pages": rows, "info": info}) + "\n")


def select(a):
    names = {int(k): v for k, v in json.load(open(HERE / "kind_names.json")).items()}
    cands = json.load(open(a.cands))
    pqj = set(json.load(open(a.pqj)))
    share = {k["kind"]: k["share"] for k in json.load(open(HERE / "kinds.json"))["kinds"]}
    target = Counter({k: round(v * a.scale) for k, v in WEAK.items()})
    for k, s in share.items():
        target[names[k]] += round(SPREAD * s * a.scale)
    rows = {}
    for l in open(a.cache):
        r = json.loads(l)
        if r["pages"] and r["doc"] in cands:
            rows[r["doc"]] = r["pages"]
    have = json.load(open(a.have))["pages"] if a.have else []
    per_doc, per_j, got = Counter(p["doc"] for p in have), Counter(p["journal"] for p in have), Counter(p["kind"] for p in have)
    taken = {p["page"] for p in have}
    rng = np.random.default_rng(SEED + len(have))
    pool = defaultdict(list)   # kind -> [(priority, doc, number, lines, marker)]
    for d, ps in rows.items():
        for no, n, k, m in ps:
            if f"{d}_p{no}" in taken: continue
            pri = rng.random() * (3.0 if (m and names[k] in PQ_KINDS) else 1.0) * (1.5 if (d[:4] in pqj and names[k] in PQ_KINDS) else 1.0)
            pool[names[k]].append((pri, d, no, n, m))
    avail = {k: len(v) for k, v in pool.items()}
    for k in pool: pool[k].sort(reverse=True)
    order = sorted(target, key=lambda k: avail.get(k, 0) / max(target[k], 1))   # scarcest first
    pages = []
    for k in order:
        need = target[k] - got[k]
        for pri, d, no, n, m in pool.get(k, []):
            if need <= 0: break
            j = d[:4]
            if per_doc[d] >= CAP_DOC or per_j[j] >= (CAP_J_PQ if j in pqj else CAP_J): continue
            why = ("weak kind" if k in WEAK else "spread") + (f"; pull-quote marker: {m}" if m else "") + ("; journal had pull quotes" if j in pqj else "")
            pages.append({"page": f"{d}_p{no}", "doc": d, "journal": j, "number": int(no), "lines": int(n), "kind": k, "why": why, "pq_marker": m})
            per_doc[d] += 1; per_j[j] += 1; got[k] += 1; need -= 1
    json.dump({"seed": SEED, "scale": a.scale, "candidates": len(rows), "pages": pages, "targets": dict(target), "available": avail},
              open(a.out, "w"), ensure_ascii=False, indent=0)
    print(f"{len(pages)} pages from {len({p['doc'] for p in pages})} documents in {len({p['journal'] for p in pages})} journals ({len(rows)} candidate docs used)")
    for k in order:
        print(f"  {k:40s} target {target[k]:5d} got {got[k]:5d} available {avail.get(k, 0):6d}")
    print("pull-quote marker on", sum(bool(p["pq_marker"]) for p in pages), "pages")


def draw(a):
    from fresh_pages import draw_one
    pages = json.load(open(a.pages))["pages"]
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, maxtasksperchild=50) as pool:
        st = Counter(pool.imap_unordered(draw_one, [(a.docs, a.dest, p) for p in pages], chunksize=4))
    print(dict(st))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["kinds", "select", "draw"])
    ap.add_argument("--docs", default="docs"); ap.add_argument("--cache", default="kinds.jsonl"); ap.add_argument("--cands", default="candidates.json")
    ap.add_argument("--pqj", default="pq_by_journal.json"); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--have", default="")
    ap.add_argument("--out", default="pages.json"); ap.add_argument("--pages", default="pages.json"); ap.add_argument("--dest", default="inputs")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"kinds": kinds, "select": select, "draw": draw}[a.cmd](a)
