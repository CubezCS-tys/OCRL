"""Pack the Haiku-labelled pages (bulk and desk sets) and the 70 judged pages into one small file for training elsewhere.

Only Azure's lines, the page size (from the page picture's header, without the 70-pixel strip added for line numbers)
and the labels travel; no pictures. A page labelled in both sets keeps both answers ("bulk", "desk").
    uv run python scripts/pack_structure_data.py --out out/structure_v5/pack.json.gz
"""
import argparse, gzip, json
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/structure_v5/pack.json.gz")
ap.add_argument("--exclude-docs", default="0013-002-001-005,0073-000-239-002,0175-000-004-016,0499-026-119-022",
                help="documents kept out of training (the keyed documents)")
a = ap.parse_args()
L = Path("out/labeller"); excl = set(a.exclude_docs.split(","))


def size(folder: Path):
    with Image.open(next(folder.glob("page_*.jpg"))) as im:
        return im.width - 70, im.height


pages = {}
for s in ("bulk", "desk"):
    for f in sorted((L / f"{s}_inputs").iterdir()):
        lab = L / f"{s}_haiku" / f"{f.name}.json"
        doc = f.name.rsplit("_p", 1)[0]
        if not lab.exists() or doc in excl:
            continue
        try:
            ans = json.loads(lab.read_text(encoding="utf-8"))
        except Exception:
            continue
        if f.name not in pages:
            W, H = size(f)
            pages[f.name] = {"page": f.name, "doc": doc, "W": W, "H": H, "lines": json.load(open(f / "lines.json", encoding="utf-8"))}
        pages[f.name][s] = ans
dev = []
for f in sorted((L / "dev_inputs").iterdir()):
    W, H = size(f)
    dev.append({"page": f.name, "doc": f.name.rsplit("_p", 1)[0], "W": W, "H": H, "lines": json.load(open(f / "lines.json", encoding="utf-8")),
                **{w: json.load(open(L / f"dev_{w}" / f"{f.name}.json", encoding="utf-8")) for w in ("opus", "sonnet", "argon")}})
with gzip.open(a.out, "wt", encoding="utf-8") as fh:
    json.dump({"haiku": list(pages.values()), "dev": dev}, fh, ensure_ascii=False)
print(len(pages), "labelled pages;", sum("bulk" in p for p in pages.values()), "bulk,", sum("desk" in p for p in pages.values()), "desk,",
      sum("bulk" in p and "desk" in p for p in pages.values()), "both;", len({p["doc"] for p in pages.values()}), "documents;", len(dev), "judged pages")
