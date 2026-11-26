"""Build docs/real_0013_data.js: numbers and pictures for the first real-scan test page."""
import base64, difflib, json, re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from ocrl.blobs.layout import analyze
from ocrl.blobs.viz import overlay

rows = json.load(open("out/real/0013_v2_own_norepair/blocks.json"))
old = json.load(open("out/real/0013_v1_own/blocks.json"))
ref = json.load(open("data/real/parsed/model/v2_long6_from3/0013-002-001-005.master.json", encoding="utf-8"))
blocks = {b["id"]: b for b in ref["blocks"]}
png = lambda img: "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()

by = defaultdict(Counter)
for r in rows:
    by[r["type"]].update(blocks=1, edits=r["edits"], chars=r["chars"])
was = defaultdict(Counter)
for r in old:
    was[r["type"]].update(edits=r["edits"], chars=r["chars"])
types = [{"type": k, "blocks": c["blocks"], "chars": c["chars"], "cer": 100 * c["edits"] / max(c["chars"], 1),
          "was": 100 * was[k]["edits"] / max(was[k]["chars"], 1)}
         for k, c in sorted(by.items(), key=lambda kv: -kv[1]["chars"]) if c["chars"] >= 200]

P = [r for r in rows if r["type"] == "paragraph"]
DIG, chars = set("0123456789٠١٢٣٤٥٦٧٨٩"), sum(r["chars"] for r in P)
kind = lambda ch: "Spaces" if ch == " " else "Digits" if ch in DIG else "Latin letters" if ch.isascii() and ch.isalpha() else "Arabic letters" if "ء" <= ch <= "ي" else "Punctuation"
kinds, sub = Counter(), Counter()
for r in P:
    a, b = r["ref"], r["ours"]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        ra, rb = a[i1:i2], b[j1:j2]
        kinds[kind(ra[0]) if ra else kind(rb[0])] += max(len(ra), len(rb))
        if op == "replace" and len(ra) == len(rb) == 1:
            sub[(ra, rb)] += 1


def diff(a, b):
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        out.append([op[0], a[i1:i2], b[j1:j2]])
    return out


def example(bid, title):
    r = next(x for x in rows if x["id"] == bid)
    part = blocks[bid]["parts"][0]
    img = cv2.imread(f"data/real/0013/pages/p-{part['page']:02d}.png", cv2.IMREAD_GRAYSCALE)
    pg = ref["pages"][part["page"] - 1]
    sx, sy = img.shape[1] / pg["width"], img.shape[0] / pg["height"]
    x0, y0, x1, y1 = part["regions"][0]
    crop = img[max(int(y0 * sy) - 8, 0) : int(y1 * sy) + 8, max(int(x0 * sx) - 8, 0) : int(x1 * sx) + 8]
    crop = crop[:420]  # the first few lines are enough to look at
    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    lay = analyze(crop)
    return {"title": title, "id": bid, "page": part["page"], "cer": 100 * r["edits"] / r["chars"], "chars": r["chars"],
            "scan": png(crop), "overlay": png(overlay(crop, lay, scale=1)), "diff": diff(r["ref"][:420], r["ours"][:420])}


single = [r for r in P if len(blocks[r["id"]]["parts"]) == 1 and len(blocks[r["id"]]["parts"][0]["regions"]) == 1 and r["chars"] > 150]
typical = sorted(single, key=lambda r: abs(r["edits"] / r["chars"] - 0.046))[0]
best = sorted(single, key=lambda r: r["edits"] / r["chars"])[0]
latin = sorted([r for r in single if re.search("[A-Za-z]{4}", r["ref"])], key=lambda r: -r["edits"] / r["chars"])[0]
examples = [example(typical["id"], "A typical paragraph"), example(best["id"], "The best paragraph"),
            example("b143", "Stretched letters, which the first model read as extra letters"), example(latin["id"], "A paragraph with English words")]
data = {
    "types": types, "para": {"chars": chars, "cer": 100 * sum(r["edits"] for r in P) / chars, "blocks": len(P)},
    "all": {"chars": sum(r["chars"] for r in rows), "cer": 100 * sum(r["edits"] for r in rows) / sum(r["chars"] for r in rows), "blocks": len(rows),
            "was": 100 * sum(r["edits"] for r in old) / sum(r["chars"] for r in old)},
    "para_was": 100 * sum(r["edits"] for r in old if r["type"] == "paragraph") / sum(r["chars"] for r in old if r["type"] == "paragraph"),
    "kinds": [[k, 100 * v / chars] for k, v in kinds.most_common()],
    "subs": [[a, b, n] for (a, b), n in sub.most_common(12)],
    "pages": [[p, 100 * sum(r["edits"] for r in P if r["page"] == p) / sum(r["chars"] for r in P if r["page"] == p)] for p in sorted({r["page"] for r in P})],
    "examples": examples, "made_by": ref["made_by"],
}
Path("docs/real_0013_data.js").write_text("window.REAL = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print("examples:", [(e["id"], round(e["cer"], 1)) for e in examples], "| size", Path("docs/real_0013_data.js").stat().st_size // 1024, "KB")
