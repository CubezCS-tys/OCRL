"""Readers against lines that people read blind (inkscript experiment 103), not against Azure.

Two readers typed each line without seeing any candidate; a third settled the lines where they differed.
That settled reading is the truth here. Scored: Azure, the owner's line reader M, and OCRL's reader on the same crops.

    OMP_NUM_THREADS=2 uv run python scripts/blind_scoreboard.py [--ours out/recognizer_v3.pt]
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ocrl.recog.realink import fold
from ocrl.text import strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/yassine/data/inkscript_labelled/experiments")
ap.add_argument("--ours", default="")
ap.add_argument("--lines", default="")
ap.add_argument("--dump", default="", help="write every line's truth and each reader's text here, for the comparison page")
ap.add_argument("--line-reader", nargs="*", default=[], help="line-reader checkpoints, each given as NAME=PATH")
ap.add_argument("--texts", nargs="*", default=[], help="readings made elsewhere, each NAME=FILE.jsonl of {\"id\", \"text\"} (crop ids as in blind/crops)")
a = ap.parse_args()
X = Path(a.root) / "103_writer_as_judge/out"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
norm = lambda t: " ".join(strip_marks(" ".join(fold(w) for w in (t or "").split())).split())
disputes = {r["id"]: r for r in rd(X / "blind/disputes.jsonl")}
wit = {r["id"]: r for r in rd(X / "b/private/witnesses.jsonl")}
A = {r["id"]: r for r in rd(X / "b/settle/A/readings.jsonl")}
B = {r["id"]: r for r in rd(X / "b/settle/B/readings.jsonl")}
C = {r["id"]: r for r in rd(X / "b/settle/C/readings.jsonl")}
year = {}
for l in open(Path(a.root) / "104_page_model/out/targets/targets_b.jsonl", encoding="utf-8"):
    r = json.loads(l); year.setdefault(r["stem"], r.get("year"))
for l in open(Path(a.root) / "76_finetune_pilot/out/inventory76.jsonl", encoding="utf-8"):
    r = json.loads(l); year.setdefault(r["stem"], r.get("year"))

truth, dropped = {}, Counter()
for i, d in disputes.items():
    c = C.get(i)
    t = c["printed"] if c else A[i]["printed"] if norm(A[i]["printed"]) == norm(B[i]["printed"]) else None
    bad = (c or A[i]).get("unreadable") or t is None
    if bad or not norm(t):
        dropped["unreadable or unsettled"] += 1
        continue
    truth[i] = norm(t)
print(f"lines {len(disputes)} | with a settled reading {len(truth)} | dropped {dict(dropped)} | settled by the third reader {sum(i in C for i in truth)}")

readers = defaultdict(dict)  # name -> id -> text
for i in truth:
    cands, names = disputes[i]["candidates"], wit[i]["witnesses"]
    for k, cand in enumerate(cands):
        who = names.get(str(k)) if isinstance(names, dict) else names[k]
        text = cand["text"] if isinstance(cand, dict) else cand
        for w in (who if isinstance(who, list) else [who]):
            readers[str(w)][i] = norm(text)
print("candidate sources:", {k: len(v) for k, v in readers.items()})

if a.line_reader:
    import cv2, torch
    from ocrl.recog.linereader import LineReader
    torch.set_num_threads(2); cv2.setNumThreads(2)
    todo = [i for i in truth if (X / "blind/crops" / f"{i}.png").exists()]
    imgs = [cv2.imread(str(X / "blind/crops" / f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in todo]
    for spec in a.line_reader:
        name, path = spec.split("=", 1)
        for i, t in zip(todo, LineReader(path, device="cpu").read(imgs)):
            readers[name][i] = norm(t)

for spec in a.texts:
    name, path = spec.split("=", 1)
    for r in rd(path):
        if r["id"] in truth:
            readers[name][r["id"]] = norm(r["text"])

if a.ours:
    import cv2, torch
    from ocrl.pipeline import OCR
    torch.set_num_threads(2); cv2.setNumThreads(2)
    ocr = OCR(a.ours, linefinder=a.lines or None)
    for n, i in enumerate(truth):
        f = X / "blind/crops" / f"{i}.png"
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.copyMakeBorder(g, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
        # the crops are 250 dpi; our reader's layout rules were tuned at 300
        g = cv2.resize(g, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        res = ocr.read(g)
        # the crop is one line with slivers of its neighbours: keep the line with the most ink
        best = max(res.lines, key=lambda ln: len(ln.clusters), default=None)
        readers["OCRL " + Path(a.ours).stem][i] = norm(best.text if best else "")


def era(i):
    y = year.get(disputes[i]["stem"])
    m = re.search(r"(1[89]\d\d|20\d\d)", str(y or ""))
    return "unknown" if not m else "before 2000" if int(m.group(1)) < 2000 else "2000 on"


sets = {"all lines": lambda i: True, "control lines (readers agreed with each other)": lambda i: disputes[i]["control"],
        "disputed lines": lambda i: not disputes[i]["control"], "before 2000": lambda i: era(i) == "before 2000", "2000 on": lambda i: era(i) == "2000 on"}
out = {}
for sname, f in sets.items():
    ids = [i for i in truth if f(i)]
    if not ids:
        continue
    print(f"\n{sname}: {len(ids)} lines, {sum(len(truth[i]) for i in ids):,} characters")
    for name, got in sorted(readers.items()):
        have = [i for i in ids if i in got]
        if len(have) < 0.5 * len(ids):
            continue
        e = sum(Levenshtein.distance(got[i], truth[i]) for i in have); n = sum(len(truth[i]) for i in have)
        exact = sum(got[i] == truth[i] for i in have)
        print(f"   {name:22s} lines {len(have):4d}  characters wrong {100 * e / n:5.2f}%  lines exactly right {100 * exact / len(have):5.1f}%")
        out[f"{sname}|{name}"] = {"lines": len(have), "cer": 100 * e / n, "exact": 100 * exact / len(have)}
json.dump(out, open("out/blind_scoreboard.json", "w"), ensure_ascii=False, indent=1)
if a.dump:
    rows = [{"id": i, "era": era(i), "control": bool(disputes[i]["control"]), "truth": truth[i], "readers": {n: g[i] for n, g in readers.items() if i in g}} for i in truth]
    json.dump(rows, open(a.dump, "w"), ensure_ascii=False)
