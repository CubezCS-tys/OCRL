"""Pages where two labellers disagree, as a folder of links to their inputs, for a third labeller to settle.

A page goes to the third labeller when more than --lines of its lines get different types (or a line is missing
from one answer), or when fewer than --blocks of the blocks are cut identically (same set of lines; share of the union of
both answers' blocks), or when either answer is missing or unreadable. On the 347 check and unseen pages that have both
Haiku and Flash: --lines 0.02 --blocks 0.8 sends 55.6%; --lines 0.05 --blocks 0.7 sends 40.3%.

    uv run python scripts/next_disagree.py --inputs out/labeller/next_inputs --a out/labeller/next_haiku --b out/labeller/next_flash \
        --out out/labeller/next_argon_inputs --lines 0.02 --blocks 0.8
"""
import argparse, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--lines", type=float, default=0.02); ap.add_argument("--blocks", type=float, default=0.8); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()


def load(p):
    try:
        b = json.load(open(p)).get("blocks", [])
        return {x: k.get("type") for k in b for x in k.get("lines", [])}, {frozenset(k.get("lines", [])) for k in b if k.get("lines")}
    except Exception:
        return None


sent = total = 0
out = Path(a.out)
for d in sorted(Path(a.inputs).iterdir()):
    if not (d / "lines.json").exists(): continue
    total += 1
    A, B = load(Path(a.a) / f"{d.name}.json"), load(Path(a.b) / f"{d.name}.json")
    if A and B:
        ids = set(A[0]) | set(B[0])
        lines = sum(A[0].get(x) != B[0].get(x) for x in ids) / max(len(ids), 1)
        blocks = len(A[1] & B[1]) / max(len(A[1] | B[1]), 1)
        go = lines > a.lines or blocks < a.blocks
    else:
        go = True
    if go:
        sent += 1
        if not a.dry:
            out.mkdir(parents=True, exist_ok=True)
            if not (out / d.name).exists(): os.symlink(d.resolve(), out / d.name)
print(f"{sent} of {total} pages ({sent / max(total, 1):.1%}) {'would go' if a.dry else 'linked into ' + str(out)}")
