"""Turn the Desk database into a training table for the structure model: one row per Azure line.

Label of a line = the role of the block it belongs to, or of the page-furniture region it sits in,
and whether it is the first line of its block. Documents in the locked sets are left out by id.
"""
import collections, json, sqlite3, time
from pathlib import Path

import numpy as np
import torch

from ocrl.structure.features import FIRST, LAST, ROLE_ID, geometry, text_ids
from ocrl.text import strip_marks

DB = "/home/yassine/data/inkscript_labelled/data/desk/releases/R20261104T151029.db"
locked = set(json.load(open("data/locked_exam_ids.json")))
locked |= set(Path("/home/yassine/inkscript_v2/data/testset/ids.txt").read_text().split())
db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
versions = db.execute("select stem, version_id, run from versions where is_current=1 order by stem").fetchall()
page_size = {(s, p): (w, h) for s, p, w, h in db.execute("select stem, page, width_px, height_px from pages")}
print(len(versions), "documents;", sum(v[0] in locked for v in versions), "locked and skipped", flush=True)

t = time.time()
pages, texts, chars = [], [], collections.Counter()
SQL = """select m.page, m.azure_line_no, m.block_id, b.role, r.role, a.x, a.y, a.w, a.h, a.text
         from line_map m left join blocks b on b.version_id=m.version_id and b.block_id=m.block_id
         left join page_regions r on r.version_id=m.version_id and r.region_id=m.region_id
         join azure_lines a on a.stem=? and a.page=m.page and a.line_no=m.azure_line_no
         where m.version_id=? order by m.page, m.azure_line_no"""
skipped = collections.Counter()
for stem, vid, run in versions:
    if stem in locked:
        continue
    rows = db.execute(SQL, (stem, vid)).fetchall()
    first_of_block = {}
    for page, no, bid, *_ in rows:
        if bid is not None and bid not in first_of_block:
            first_of_block[bid] = (page, no)
    by_page = collections.defaultdict(list)
    for page, no, bid, brole, rrole, x, y, w, h, text in rows:
        role = brole if bid is not None else rrole
        if role not in ROLE_ID or w is None or w <= 0 or h <= 0:
            skipped[role] += 1
            continue
        start = bid is None or first_of_block[bid] == (page, no)
        by_page[page].append((x, y, x + w, y + h, ROLE_ID[role], int(start), text or ""))
    for page, ls in by_page.items():
        if (stem, page) not in page_size or len(ls) > 400:
            skipped["page"] += 1
            continue
        for l in ls:
            chars.update(strip_marks(l[6]))
        pages.append((stem, page, run, page_size[(stem, page)], ls))
print(f"{len(pages):,} pages, {sum(len(p[4]) for p in pages):,} lines in {time.time() - t:.0f} s | skipped:", dict(skipped), flush=True)

vocab = {c: i + 2 for i, (c, n) in enumerate(chars.most_common()) if n >= 40}
G, T, R, S, offsets, meta = [], [], [], [], [0], []
for stem, page, run, (pw, ph), ls in pages:
    boxes = np.array([l[:4] for l in ls], np.float32)
    G.append(geometry(boxes, [l[6] for l in ls], pw, ph))
    T.append(np.stack([text_ids(l[6], vocab) for l in ls]))
    R.append(np.array([l[4] for l in ls], np.int16)); S.append(np.array([l[5] for l in ls], np.int8))
    offsets.append(offsets[-1] + len(ls)); meta.append((stem, page, run))
out = {"geom": torch.from_numpy(np.concatenate(G)), "text": torch.from_numpy(np.concatenate(T)), "role": torch.from_numpy(np.concatenate(R)),
       "start": torch.from_numpy(np.concatenate(S)), "offsets": torch.tensor(offsets), "meta": meta, "vocab": vocab}
torch.save(out, "data/structure/desk_lines.pt")
runs = collections.Counter(m[2] for m in meta)
print("vocabulary", len(vocab), "| pages by run:", runs.most_common(6), "| file", Path("data/structure/desk_lines.pt").stat().st_size // 2**20, "MB")
