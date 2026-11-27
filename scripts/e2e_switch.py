"""Run scripts/e2e_judged.py with some of the rule switches changed, to see which rules still help a structure model.

    uv run python scripts/e2e_switch.py MARKS=0 ENDS=0 -- --lines-from r2_cell --typer out/structure_v5.pt --name v5_nomarks
Switches: MARKS, MARGINS, FOOTNOTE_RULE (ocrl.doc.parse), ENDS, SEQUENCE (ocrl.structure.tag).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
cut = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
flags, rest = sys.argv[1:cut], sys.argv[cut + 1:]
import ocrl.doc.parse as P
import ocrl.structure.tag as T
for f in flags:
    k, v = f.split("=")
    mod = P if hasattr(P, k) else T
    assert hasattr(mod, k), k
    setattr(mod, k, bool(int(v)))
    print(f"{mod.__name__}.{k} = {getattr(mod, k)}")
import e2e_judged as E
sys.argv = [sys.argv[0]] + rest
E.main()
