"""Choose a subset of the Desk pages as whole documents, spread over journals, decades and layouts; fixed seed.

Layout of a document = one or two columns (whether most of its pages have lines whose right edge ends before the
page middle while others start after it), from the database's boxes.
Strata are journal x decade x layout; documents are drawn round-robin across strata (each stratum shuffled with
the seed) until the page target is reached, so small journals and old decades get in before big ones fill up.
    uv run python scripts/desk_subset.py --pages 4000 --out out/labeller/desk_subset.txt
"""
import argparse, collections, gzip, json, random, sqlite3

DB = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"
ap = argparse.ArgumentParser(); ap.add_argument("--pages", type=int, default=4000); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--src", default="out/labeller/desk_pages.jsonl.gz"); ap.add_argument("--out", default="out/labeller/desk_subset.txt")
a = ap.parse_args()
db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
year = dict(db.execute("select d.stem, p.year_shown from documents d join publications p on p.pub_id=d.pub_id"))
docs = {}
for l in gzip.open(a.src, "rt", encoding="utf-8"):
    d = json.loads(l)
    two = 0
    for p in d["pages"]:
        mid = p["w"] / 2; ls = p["lines"]
        left = sum(x + w < mid * 0.98 for x, y, w, h, t in ls); right = sum(x > mid * 1.02 for x, y, w, h, t in ls)
        two += left >= 5 and right >= 5
    docs[d["stem"]] = ([p["page"] for p in d["pages"]], "2col" if two > len(d["pages"]) / 2 else "1col")
strata = collections.defaultdict(list)
for s, (pages, lay) in docs.items():
    y = year.get(s); dec = f"{(y // 10) * 10}s" if y else "?"
    strata[(s[:4], dec, lay)].append(s)
rng = random.Random(a.seed)
keys = sorted(strata); rng.shuffle(keys)
for k in keys: rng.shuffle(strata[k])
chosen, n = [], 0
while n < a.pages and any(strata[k] for k in keys):
    for k in keys:
        if strata[k] and n < a.pages:
            s = strata[k].pop(); chosen.append(s); n += len(docs[s][0])
chosen.sort()
with open(a.out, "w") as f:
    for s in chosen: f.write(s + "\n")
dec = collections.Counter(); lay = collections.Counter(); jr = collections.Counter()
for s in chosen:
    y = year.get(s); dec[f"{(y // 10) * 10}s" if y else "?"] += len(docs[s][0]); lay[docs[s][1]] += len(docs[s][0]); jr[s[:4]] += 1
print(f"{len(chosen)} documents, {n} pages, {len(jr)} journals (of {len({s[:4] for s in docs})}) | pages by decade {sorted(dec.items())} | by layout {dict(lay)}")
