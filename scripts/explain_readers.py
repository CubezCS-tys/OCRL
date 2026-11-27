"""A page that shows the blind-read lines themselves: the scan, what people read, and what each reader wrote.

    uv run python scripts/blind_scoreboard.py --line-reader V7 --ours out/recognizer_v3.pt --dump out/explain/blind_lines.json
    uv run python scripts/explain_readers.py
Writes out/explain/readers.html and readers_data.js (private scans: out/ is not tracked).
"""
import base64, json
from pathlib import Path

import cv2
from rapidfuzz.distance import Levenshtein

X = Path("/home/yassine/data/inkscript_labelled/experiments/103_writer_as_judge/out/blind/crops")
rows = json.load(open("out/explain/blind_lines.json", encoding="utf-8"))
NAMES = {"azure": "Azure", "agreed": "Azure", "experiment 89 reader (v7)": "Your 8.4M reader", "OCRL recognizer_v3": "Our cluster reader"}
ORDER = ["Azure", "Your 8.4M reader", "Our cluster reader"]


def segments(got: str, truth: str):
    """The reader's text cut into runs: right, wrong (put in or changed), and marks where something is missing."""
    bad, gaps = set(), set()
    for op, i, j in Levenshtein.editops(got, truth):
        (gaps if op == "insert" else bad).add(i)
    out, cur, state = [], "", None
    for i, ch in enumerate(got + "\0"):
        if i in gaps:
            if cur: out.append([cur, state]); cur = ""
            out.append(["", 2]); state = None
        if ch == "\0": break
        s = 1 if i in bad else 0
        if s != state and cur: out.append([cur, state]); cur = ""
        cur += ch; state = s
    if cur: out.append([cur, state])
    return out


data = []
for r in rows:
    g = cv2.imread(str(X / f"{r['id']}.png"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    h = 72; g = cv2.resize(g, (max(int(g.shape[1] * h / g.shape[0]), 8), h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 62])
    readers = {}
    for name, text in r["readers"].items():
        if name in NAMES:
            readers[NAMES[name]] = {"seg": segments(text, r["truth"]), "err": Levenshtein.distance(text, r["truth"])}
    if "Azure" not in readers or "Your 8.4M reader" not in readers:
        continue
    data.append({"id": r["id"], "era": r["era"], "easy": r["control"], "n": len(r["truth"]), "truth": r["truth"], "img": base64.b64encode(buf).decode(), "r": readers})

tot = {n: [sum(d["r"][n]["err"] for d in data if n in d["r"]), sum(d["n"] for d in data if n in d["r"]), sum(d["r"][n]["err"] == 0 for d in data if n in d["r"]), sum(n in d["r"] for d in data)] for n in ORDER}
Path("out/explain/readers_data.js").write_text("const ORDER = " + json.dumps(ORDER) + ";\nconst TOTALS = " + json.dumps(tot) + ";\nconst LINES = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print(len(data), "lines;", {n: f"{100 * t[0] / max(t[1], 1):.2f}% wrong, {t[2]}/{t[3]} exact" for n, t in tot.items()})
