"""How far Haiku's labels on the Desk pages differ from the Gemini roles the Desk database holds, line by line.

Gemini's role of a line = the role of its block, or of the page-furniture region it sits in (as build_structure_data.py
reads it), mapped to the guide's types with ocrl.structure.tag.TO_TYPE. Line id in the labeller input = "PAGE:(line_no - 1)".
Block starts: Gemini's block starts on this page (a block carried over from the page before starts again at its first
line on this page, as Haiku only sees one page; furniture regions: every line its own block, as build_structure_data.py
does); Haiku's = the first line listed in each of its blocks. Both read over the lines both sides labelled.
--sample N writes N disagreements (random blocks, fixed seed) as picture crops with both labels, for looking at.
    uv run python scripts/desk_compare.py --inputs out/labeller/desk_inputs --labels out/labeller/desk_haiku --sample 30
"""
import argparse, collections, json, random, sqlite3
from pathlib import Path

from ocrl.structure.tag import TO_TYPE

DB = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="out/labeller/desk_compare.json"); ap.add_argument("--sample", type=int, default=0); ap.add_argument("--sheets", default="out/labeller/desk_compare_sheets")
a = ap.parse_args()
db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
vid = dict(db.execute("select stem, version_id from versions where is_current=1"))
SQL = """select m.page, m.azure_line_no, m.block_id, b.role, r.role from line_map m
         left join blocks b on b.version_id=m.version_id and b.block_id=m.block_id
         left join page_regions r on r.version_id=m.version_id and r.region_id=m.region_id
         where m.version_id=? and m.page=? and m.azure_line_no is not null order by m.azure_line_no"""
pages = collections.Counter(); pair = collections.Counter(); unmapped = collections.Counter(); raw = collections.Counter()
st = collections.Counter(); dis = []
for f in sorted(Path(a.labels).glob("*.json")):
    if f.name.startswith("_"): continue
    stem, p = f.stem.rsplit("_p", 1); p = int(p)
    try:
        ans = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        pages["unreadable"] += 1; continue
    haiku = {i: b["type"] for b in ans["blocks"] for i in b["lines"]}
    hstart = {b["lines"][0] for b in ans["blocks"] if b["lines"]}
    lines = json.load(open(Path(a.inputs) / f.stem / "lines.json", encoding="utf-8")); ids = {l["id"] for l in lines}
    pages["pages"] += 1; pages["lines"] += len(ids); pages["left out"] += len(ids - set(haiku))
    pages["pages with lines left out"] += bool(ids - set(haiku))
    seen = set()
    for page, no, bid, brole, rrole in db.execute(SQL, (vid[stem], p)):
        role = brole if bid is not None else rrole
        g = TO_TYPE.get(role); lid = f"{page}:{no - 1}"
        gstart = bid is None or bid not in seen; seen.add(bid)
        if g is None:
            unmapped[role] += 1; continue
        h = haiku.get(lid, "(left out)")
        pair[(g, h)] += 1; raw[(role, h)] += 1
        if h != "(left out)":
            st[(gstart, lid in hstart)] += 1
        if g != h:
            dis.append((f.stem, lid, role, g, h))
tot = sum(pair.values()); agree = sum(n for (g, h), n in pair.items() if g == h)
by_g = collections.Counter(); ok_g = collections.Counter()
for (g, h), n in pair.items():
    by_g[g] += n; ok_g[g] += n * (g == h)
print(f"{pages['pages']} pages, {pages['lines']} lines; {pages['left out']} lines left out by Haiku on {pages['pages with lines left out']} pages; {pages['unreadable']} unreadable files")
print(f"lines with a Gemini role: {tot}; Haiku gives the same type to {agree} ({100 * agree / tot:.1f}%); Gemini roles with no type: {dict(unmapped)}")
print("\nGemini type                 lines   Haiku agrees   Haiku says instead (top 3)")
for g, n in by_g.most_common():
    other = sorted(((h, m) for (gg, h), m in pair.items() if gg == g and h != g), key=lambda x: -x[1])[:3]
    print(f"  {g:24s} {n:8d}   {100 * ok_g[g] / n:5.1f}%      " + ", ".join(f"{h} {m}" for h, m in other))
print("\nbiggest disagreements (Gemini role [type] -> Haiku)")
for (r, h), n in sorted(((k, n) for k, n in raw.items() if TO_TYPE.get(k[0]) != k[1]), key=lambda x: -x[1])[:15]:
    print(f"  {r} [{TO_TYPE[r]}] -> {h}: {n} lines ({100 * n / tot:.1f}% of all)")
both, g_only, h_only, none = st[(True, True)], st[(True, False)], st[(False, True)], st[(False, False)]
print(f"\nblock starts (lines both labelled): agree on {100 * (both + none) / sum(st.values()):.1f}% of lines | Gemini starts {both + g_only}, Haiku starts {both + h_only}, both {both} "
      f"| of Gemini's starts Haiku has {100 * both / (both + g_only):.1f}%, of Haiku's Gemini has {100 * both / (both + h_only):.1f}%")
json.dump({"pages": pages, "pairs": {f"{g}|{h}": n for (g, h), n in pair.most_common()}, "raw": {f"{g}|{h}": n for (g, h), n in raw.most_common()},
           "starts": {f"{k[0]}|{k[1]}": v for k, v in st.items()}, "unmapped": unmapped}, open(a.out, "w"), indent=1)

if a.sample:
    import cv2, numpy as np
    rng = random.Random(7)
    blocks = {}  # one per (page, gemini role, haiku type) run, so the sample is not 30 lines of one table
    for d in dis: blocks.setdefault((d[0], d[2], d[4]), d)
    pick = rng.sample(sorted(blocks.values()), min(a.sample, len(blocks)))
    out = Path(a.sheets); out.mkdir(parents=True, exist_ok=True); rows = []
    for k, (folder, lid, role, g, h) in enumerate(pick):
        img = cv2.imread(str(next((Path(a.inputs) / folder).glob("page_*.jpg"))))
        box = next(l["box"] for l in json.load(open(Path(a.inputs) / folder / "lines.json", encoding="utf-8")) if l["id"] == lid)
        H = img.shape[0]; y0 = max(0, int(box[1] / 1000 * H) - 260); y1 = min(H, int(box[3] / 1000 * H) + 260)
        x0, x1 = int(box[0] / 1000 * 1150), int(box[2] / 1000 * 1150)
        cv2.rectangle(img, (x0 - 4, int(box[1] / 1000 * H) - 4), (x1 + 4, int(box[3] / 1000 * H) + 4), (0, 0, 255), 3)
        crop = img[y0:y1]; crop = cv2.copyMakeBorder(crop, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(crop, f"#{k} {folder} line {lid}: Gemini {role} [{g}]  Haiku {h}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        cv2.imwrite(str(out / f"d{k:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        rows.append({"k": k, "page": folder, "line": lid, "gemini_role": role, "gemini_type": g, "haiku": h})
    json.dump(rows, open(out / "sample.json", "w"), indent=1)
    print(f"\n{len(rows)} disagreement crops in {out}")
