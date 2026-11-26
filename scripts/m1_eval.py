"""Milestone 1: page image -> text, end to end, on fonts and sentences never trained on."""
import argparse, json
from collections import Counter

import numpy as np
import torch

from ocrl.blobs.layout import analyze
from ocrl.pipeline import OCR
from ocrl.recog.crops import cluster_labels
from ocrl.recog.data import TextSource, font_paths, latin_paths, make_page
from ocrl.text import load_corpus, strip_tatweel

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/recognizer.pt")
ap.add_argument("--pages", type=int, default=30)
ap.add_argument("--json", default="out/m1.json")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--dressed", action="store_true", help="add vowel marks, stretches, made-up words and Latin, as in training")
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


corpus = load_corpus("data/corpus/sentences.txt")
test_text = corpus[int(len(corpus) * 0.95):]
report = {}
for split, strength in (("held_out", 0.0), ("held_out", 1.0), ("train", 1.0)):
    for repair in (False, True):
        ocr = OCR(args.ckpt, repair=repair)
        rng = np.random.default_rng(555)
        src = TextSource(test_text, rng, plain=not args.dressed)
        fonts, cache, c = font_paths(split), {}, Counter()
        for _ in range(args.pages):
            page = make_page(fonts, src, rng, cache, lines=12, strength=strength, latin_fonts=latin_paths())
            lay = analyze(page.image)
            res = ocr.read(page.image, lay)
            c["pages"] += 1
            c["pages_dots_fusing"] += "dots-fusing" in res.notes
            truth, got = strip_tatweel("\n".join(page.lines)), res.text
            c["chars"] += len(truth)
            c["edits"] += edit(got, truth)
            c["chars_nospace"] += len(truth.replace(" ", ""))
            c["edits_nospace"] += edit(got.replace(" ", ""), truth.replace(" ", ""))
            tw, gw = truth.split(), got.split()
            c["words"] += len(tw)
            c["word_edits"] += edit(gw, tw)
            # Do the rule checks catch the recogniser's mistakes?
            labels = cluster_labels(page, lay)
            for ln in res.lines:
                for cl in ln.clusters:
                    wrong = cl.text != labels.get(cl.body, "")
                    flagged = bool(cl.flags) and cl.flags != ["repaired"]
                    c["clusters"] += 1
                    c["wrong"] += wrong
                    c["flagged"] += flagged
                    c["wrong_flagged"] += wrong and flagged
                    c["touch_notes"] += cl.touching
                    c["touch_notes_read_right"] += cl.touching and not wrong
                    c["repaired"] += cl.flags == ["repaired"]
                    c["repaired_right"] += cl.flags == ["repaired"] and not wrong
        name = f"{'unseen' if split == 'held_out' else 'seen'} fonts, {'clean' if strength == 0 else 'degraded'}, rules {'on' if repair else 'off'}"
        r = {
            "CER %": 100 * c["edits"] / c["chars"],
            "CER ignoring spaces %": 100 * c["edits_nospace"] / c["chars_nospace"],
            "WER %": 100 * c["word_edits"] / c["words"],
            "cluster errors %": 100 * c["wrong"] / max(c["clusters"], 1),
            "errors flagged by rules %": 100 * c["wrong_flagged"] / max(c["wrong"], 1),
            "correct clusters flagged %": 100 * (c["flagged"] - c["wrong_flagged"]) / max(c["clusters"] - c["wrong"], 1),
            "repairs": c["repaired"], "repairs correct": c["repaired_right"], "chars": c["chars"],
            "touching found": c["touch_notes"], "touching read right": c["touch_notes_read_right"],
            "pages": c["pages"], "pages with dots fusing": c["pages_dots_fusing"],
        }
        report[name] = r
        print(name, "|", "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
json.dump(report, open(args.json, "w"), indent=1)
