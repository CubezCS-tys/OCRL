"""How far Haiku's labels on the Desk pages differ from the Gemini roles the Desk database holds, line by line.

Gemini's role of a line = the role of its block, or of the page-furniture region it sits in (as build_structure_data.py
reads it), mapped to the guide's types with ocrl.structure.tag.TO_TYPE. Line id in the labeller input = "PAGE:(line_no - 1)".
    uv run python scripts/desk_compare.py --inputs out/labeller/desk_inputs --labels out/labeller/desk_haiku
"""
import argparse, collections, json, sqlite3
from pathlib import Path

from ocrl.structure.tag import TO_TYPE

DB = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", default="out/labeller/desk_compare.json")
a = ap.parse_args()
db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
vid = dict(db.execute("select stem, version_id from versions where is_current=1"))
SQL = """select m.page, m.azure_line_no, m.block_id, b.role, r.role from line_map m
         left join blocks b on b.version_id=m.version_id and b.block_id=m.block_id
         left join page_regions r on r.version_id=m.version_id and r.region_id=m.region_id
         where m.version_id=? and m.page=? and m.azure_line_no is not null"""
pages = collections.Counter(); pair = collections.Counter(); unmapped = collections.Counter()
for f in sorted(Path(a.labels).glob("*.json")):
    if f.name.startswith("_"): continue
    stem, p = f.stem.rsplit("_p", 1); p = int(p)
    try:
        ans = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        pages["unreadable"] += 1; continue
    haiku = {i: b["type"] for b in ans["blocks"] for i in b["lines"]}
    ids = {l["id"] for l in json.load(open(Path(a.inputs) / f.stem / "lines.json", encoding="utf-8"))}
    pages["pages"] += 1; pages["lines"] += len(ids); pages["left out"] += len(ids - set(haiku))
    pages["pages with lines left out"] += bool(ids - set(haiku))
    for page, no, bid, brole, rrole in db.execute(SQL, (vid[stem], p)):
        role = brole if bid is not None else rrole
        g = TO_TYPE.get(role)
        if g is None:
            unmapped[role] += 1; continue
        h = haiku.get(f"{page}:{no - 1}", "(left out)")
        pair[(g, h)] += 1
tot = sum(pair.values()); agree = sum(n for (g, h), n in pair.items() if g == h)
by_g = collections.Counter(); ok_g = collections.Counter()
for (g, h), n in pair.items():
    by_g[g] += n; ok_g[g] += n * (g == h)
print(f"{pages['pages']} pages, {pages['lines']} lines; {pages['left out']} lines left out by Haiku on {pages['pages with lines left out']} pages; {pages['unreadable']} unreadable files")
print(f"lines with a Gemini role: {tot}; Haiku gives the same type to {agree} ({100 * agree / tot:.1f}%); Gemini roles with no type: {dict(unmapped)}")
print("\nGemini type (ten most common)   lines   Haiku agrees")
for g, n in by_g.most_common(10):
    print(f"  {g:28s} {n:8d}   {100 * ok_g[g] / n:5.1f}%")
print("\nbiggest disagreements (Gemini -> Haiku)")
for (g, h), n in sorted(((k, n) for k, n in pair.items() if k[0] != k[1]), key=lambda x: -x[1])[:10]:
    print(f"  {g} -> {h}: {n} lines ({100 * n / tot:.1f}% of all)")
json.dump({"pages": pages, "pairs": {f"{g}|{h}": n for (g, h), n in pair.most_common()}, "unmapped": unmapped}, open(a.out, "w"), indent=1)
