"""How far do independent labellers agree on the development pages, and how does our structure model do against them?

A line is "settled" when the judges (by default Opus, Sonnet and Gemini 4 Argon) all give it the same type; with a
two-to-one split the majority's type is used and the line is counted as "majority". Nobody here is a person: this
measures agreement between independent strong models, which is the only check this project has.

    uv run python scripts/labeller_agreement.py --judges opus sonnet argon --others haiku flash
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

L = Path("out/labeller")
ap = argparse.ArgumentParser()
ap.add_argument("--judges", nargs="+", default=["opus", "sonnet", "argon"]); ap.add_argument("--others", nargs="*", default=["haiku", "flash"])
a = ap.parse_args()
census = json.loads(Path("out/census/census_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
KIND = {k["kind"]: k["name"] for k in census["kinds"]}
dev = json.loads(Path("out/census/dev_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
draft = {p["id"]: p for p in dev["pages"]}


def load(who, page):
    f = L / f"dev_{who}" / f"{page}.json"
    if not f.exists():
        return None
    try:
        ans = json.load(open(f, encoding="utf-8"))
        types, blocks = {}, set()
        for b in ans["blocks"]:
            for r in b["lines"]: types[r] = b["type"]
            blocks.add(frozenset(b["lines"]))
        return types, blocks
    except Exception:
        return None


pages = sorted(p.name for p in (L / "dev_inputs").iterdir())
everyone = a.judges + a.others
pair, have = defaultdict(Counter), Counter()
status, per_kind, ours, splits, rows = Counter(), defaultdict(Counter), defaultdict(Counter), Counter(), []
for page in pages:
    ids = [l["id"] for l in json.load(open(L / "dev_inputs" / page / "lines.json", encoding="utf-8"))]
    texts = {l["id"]: l["text"] for l in json.load(open(L / "dev_inputs" / page / "lines.json", encoding="utf-8"))}
    got = {w: load(w, page) for w in everyone}
    for w in everyone: have[w] += got[w] is not None
    for i, x in enumerate(everyone):
        for y in everyone[i + 1 :]:
            if got[x] and got[y]:
                pair[x, y]["lines"] += len(ids); pair[x, y]["same"] += sum(got[x][0].get(r) == got[y][0].get(r) for r in ids)
                pair[x, y]["blocks"] += len(got[x][1] | got[y][1]); pair[x, y]["same_blocks"] += len(got[x][1] & got[y][1])
    js = [w for w in a.judges if got[w]]
    if len(js) < len(a.judges):
        continue
    kind = draft[page]["kind"] if page in draft else "?"
    d = draft.get(page)
    for n, r in enumerate(ids):
        votes = Counter(got[w][0].get(r) for w in js)
        top, cnt = votes.most_common(1)[0]
        st = "settled" if cnt == len(js) else "majority" if cnt > len(js) / 2 else "open"
        status[st] += 1; per_kind[kind][st] += 1
        if st != "settled":
            splits[tuple(sorted(str(v) for v in votes))] += 1
            rows.append({"page": page, "kind": kind, "line": r, "text": texts[r][:90], "votes": {w: got[w][0].get(r) for w in everyone if got[w]}, "status": st})
        if st != "open" and d and n < len(d["lines"]):
            ours[kind]["lines"] += 1; ours[kind]["right"] += d["lines"][n]["type"] == top
            ours["ALL"]["lines"] += 1; ours["ALL"]["right"] += d["lines"][n]["type"] == top
            if d["lines"][n]["type"] != top: ours["confusions"][top, d["lines"][n]["type"]] += 1
        for w in a.others:
            if got[w] and st != "open":
                ours[w]["lines"] += 1; ours[w]["right"] += got[w][0].get(r) == top
print("answers in:", {w: have[w] for w in everyone}, "of", len(pages), "pages")
print("\npairwise agreement (line types | blocks that are identical):")
for (x, y), c in pair.items():
    print(f"  {x:7s} {y:7s} {100 * c['same'] / max(c['lines'], 1):5.1f}% | {100 * c['same_blocks'] / max(c['blocks'], 1):5.1f}%   ({c['lines']} lines)")
t = sum(status.values())
if t:
    print(f"\njudges {a.judges}: {t} lines | all agree {100 * status['settled'] / t:.1f}% | two to one {100 * status['majority'] / t:.1f}% | all differ {100 * status['open'] / t:.1f}%")
    print("by kind of page (all agree | our structure model against the judges' label):")
    for k, c in sorted(per_kind.items(), key=lambda kv: kv[1]["settled"] / max(sum(kv[1].values()), 1)):
        n = sum(c.values()); o = ours[k]
        print(f"  {k:38s} {100 * c['settled'] / n:5.1f}% of {n:4d} lines | ours {100 * o['right'] / max(o['lines'], 1):5.1f}%")
    o = ours["ALL"]; print(f"our structure model overall: {100 * o['right'] / max(o['lines'], 1):.1f}% of {o['lines']} lines")
    for w in a.others:
        if ours[w]["lines"]: print(f"{w} against the judges' label: {100 * ours[w]['right'] / ours[w]['lines']:.1f}% of {ours[w]['lines']} lines")
    print("where the judges split most:", ", ".join(f"{' vs '.join(k)} ×{n}" for k, n in splits.most_common(8)))
    print("where ours is wrong most (judges say / we say):", ", ".join(f"{k}/{m} ×{n}" for (k, m), n in ours["confusions"].most_common(8)))
json.dump({"status": dict(status), "rows": rows}, open(L / "dev_disagreements.json", "w"), ensure_ascii=False)
