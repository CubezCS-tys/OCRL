"""How far do independent labellers agree on the development pages, and how does our structure model do against them?

A line is "settled" when the judges (by default Opus, Sonnet and Gemini 4 Argon) all give it the same type; with a
two-to-one split the majority's type is used and the line is counted as "majority". Nobody here is a person: this
measures agreement between independent strong models, which is the only check this project has.

    uv run python scripts/labeller_agreement.py --judges opus sonnet argon --others haiku flash
    uv run python scripts/labeller_agreement.py --set check     # the untouched check pages: Haiku, Flash and Argon, no "ours"
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

L = Path("out/labeller")
ap = argparse.ArgumentParser()
ap.add_argument("--judges", nargs="+", default=None); ap.add_argument("--others", nargs="*", default=None)
ap.add_argument("--set", default="dev", choices=["dev", "check"], help="dev: the 70 development pages; check: the untouched check pages (scripts/check_pages.py)")
a = ap.parse_args()
if a.judges is None: a.judges = ["opus", "sonnet", "argon"] if a.set == "dev" else ["haiku", "flash", "argon"]
if a.others is None: a.others = ["haiku", "flash"] if a.set == "dev" else []
census = json.loads(Path("out/census/census_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
KIND = {k["kind"]: k["name"] for k in census["kinds"]}
if a.set == "dev":
    dev = json.loads(Path("out/census/dev_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
    draft = {p["id"]: p for p in dev["pages"]}
else:  # no draft labels of ours on the check pages: only the page's kind
    draft = {p["page"]: {"kind": p["kind"], "lines": []} for p in json.load(open(L / "check_pages.json", encoding="utf-8"))["pages"]}


def load(who, page):
    f = L / f"{a.set}_{who}" / f"{page}.json"
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


pages = sorted(p.name for p in (L / f"{a.set}_inputs").iterdir())
everyone = a.judges + a.others
pair, have = defaultdict(Counter), Counter()
pair_kind = defaultdict(Counter)  # (kind, x, y) -> lines, same, blocks, same_blocks
status, per_kind, ours, splits, rows = Counter(), defaultdict(Counter), defaultdict(Counter), Counter(), []
for page in pages:
    ids = [l["id"] for l in json.load(open(L / f"{a.set}_inputs" / page / "lines.json", encoding="utf-8"))]
    texts = {l["id"]: l["text"] for l in json.load(open(L / f"{a.set}_inputs" / page / "lines.json", encoding="utf-8"))}
    got = {w: load(w, page) for w in everyone}
    for w in everyone: have[w] += got[w] is not None
    for i, x in enumerate(everyone):
        for y in everyone[i + 1 :]:
            if got[x] and got[y]:
                pair[x, y]["lines"] += len(ids); pair[x, y]["same"] += sum(got[x][0].get(r) == got[y][0].get(r) for r in ids)
                pair[x, y]["blocks"] += len(got[x][1] | got[y][1]); pair[x, y]["same_blocks"] += len(got[x][1] & got[y][1])
                pk = pair_kind[draft[page]["kind"] if page in draft else "?", x, y]
                pk["lines"] += len(ids); pk["same"] += sum(got[x][0].get(r) == got[y][0].get(r) for r in ids)
                pk["blocks"] += len(got[x][1] | got[y][1]); pk["same_blocks"] += len(got[x][1] & got[y][1])
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
    if a.set == "dev": print("by kind of page (all agree | our structure model against the judges' label):")
    for k, c in (sorted(per_kind.items(), key=lambda kv: kv[1]["settled"] / max(sum(kv[1].values()), 1)) if a.set == "dev" else []):
        n = sum(c.values()); o = ours[k]
        print(f"  {k:38s} {100 * c['settled'] / n:5.1f}% of {n:4d} lines | ours {100 * o['right'] / max(o['lines'], 1):5.1f}%")
    if a.set != "dev":  # no draft of ours here: the labellers' agreement by kind instead, so the noise of the reference is known
        prs = [(x, y) for i, x in enumerate(a.judges) for y in a.judges[i + 1 :]]
        print("by kind of page: all three agree / two to one / no majority (share of lines); then each pair: same type % | identical blocks %")
        print(f"  {'kind of page':38s} {'lines':>5s} {'all':>5s} {'2:1':>5s} {'none':>5s}  " + "  ".join(f"{x[:5]}-{y[:5]:<5s}" + " " * 4 for x, y in prs))
        for k, c in sorted(per_kind.items(), key=lambda kv: kv[1]["settled"] / max(sum(kv[1].values()), 1)):
            n = sum(c.values())
            cells = "  ".join(f"{100 * pair_kind[k, x, y]['same'] / max(pair_kind[k, x, y]['lines'], 1):5.1f}|{100 * pair_kind[k, x, y]['same_blocks'] / max(pair_kind[k, x, y]['blocks'], 1):5.1f}" for x, y in prs)
            print(f"  {k:38s} {n:5d} {100 * c['settled'] / n:5.1f} {100 * c['majority'] / n:5.1f} {100 * c['open'] / n:5.1f}  {cells}")
    o = ours["ALL"]; a.set == "dev" and print(f"our structure model overall: {100 * o['right'] / max(o['lines'], 1):.1f}% of {o['lines']} lines")
    for w in a.others:
        if ours[w]["lines"]: print(f"{w} against the judges' label: {100 * ours[w]['right'] / ours[w]['lines']:.1f}% of {ours[w]['lines']} lines")
    print("where the judges split most:", ", ".join(f"{' vs '.join(k)} ×{n}" for k, n in splits.most_common(8)))
    if a.set == "dev": print("where ours is wrong most (judges say / we say):", ", ".join(f"{k}/{m} ×{n}" for (k, m), n in ours["confusions"].most_common(8)))
json.dump({"status": dict(status), "rows": rows}, open(L / f"{a.set}_disagreements.json", "w"), ensure_ascii=False)
