"""Run Milestone 0 over every font: render pages, run the layout stage, count."""
import glob, json, sys
from collections import Counter
from pathlib import Path

import cv2

from ocrl.blobs.layout import analyze
from ocrl.blobs.viz import overlay
from ocrl.eval.m0 import score_page
from ocrl.render.page import compose_page
from ocrl.render.shaper import Font
from ocrl.text import CHARSET, load_corpus

only = sys.argv[1:]
corpus = load_corpus("data/corpus/sentences.txt")
text = " ".join(corpus[1000:1060])
out = Path("out/m0"); out.mkdir(parents=True, exist_ok=True)
rows = {}
for path in sorted(glob.glob("data/fonts/*.ttf")):
    name = Path(path).stem
    if only and name not in only:
        continue
    total, ex = Counter(), {}
    for px in (32, 44, 60):
        font = Font(path, px)
        page = compose_page(font, text, width=int(px * 36), max_lines=30)
        lay = analyze(page.image)
        total += score_page(page, lay, ex)
        if px == 44:
            cv2.imwrite(str(out / f"{name}.png"), overlay(page.image, lay)[:1400])
    rows[name] = {"counts": dict(total), "examples": {k: v.most_common(12) for k, v in ex.items()},
                  "missing_glyphs": not font.has_glyphs(CHARSET)}
    u = total["units"]
    pc = lambda k: f"{100 * total[k] / u:6.2f}%"
    print(f"{name:26s} units {u:5d} exact {pc('exact')} | ink: as-rules {pc('ink_as_rules_predict')} touch {pc('ink_touching')} "
          f"fewer {pc('ink_fewer_blobs_than_rules')} more {pc('ink_more_blobs_than_rules')} | "
          f"lines {total['lines_found']}/{total['lines']} ordered {total['lines_in_order']} count-ok {total['lines_body_count_matches_rules']}", flush=True)
    if only:
        for k, v in ex.items():
            print("   ", k, " ".join(f"{t}×{n}" for t, n in v.most_common(12)))
json.dump(rows, open(out / "m0.json", "w"), indent=1, ensure_ascii=False)
