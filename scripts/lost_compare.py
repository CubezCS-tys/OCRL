"""Two lost_causes runs side by side, on the pages both have.

    uv run python scripts/lost_compare.py out/lost/base.json out/lost/new_all.json
"""
import json, sys
from collections import Counter

CAUSES = ["faint on the page", "picture mask", "ruled line", "other cleaning", "finder never marked", "dropped by layout", "read as empty"]
runs = [json.load(open(f)) for f in sys.argv[1:3]]
keys = [{(p["doc"], p["page"]): p for p in r["pages"]} for r in runs]
common = sorted(set(keys[0]) & set(keys[1]))
print(f"{len(common)} pages in both")
for name, k in zip(sys.argv[1:3], keys):
    t, c, kinds = Counter(), Counter(), {}
    for key in common:
        p = k[key]
        t["azure"] += p["azure"]; t["lost"] += p["lost"]; t["ours"] += p["ours"]
        t["chars"] += p.get("text_chars", 0); t["edits"] += p.get("text_edits", 0); t["invented"] += p.get("invented", 0)
        for a, b in p["cases"].items():
            c[a] += b
        for a, b in p["causes"].items():
            t[a] += b
        kk = kinds.setdefault(p["kind"], Counter()); kk["azure"] += p["azure"]; kk["lost"] += p["lost"]
    L = max(t["azure"], 1)
    print(f"\n{name}: {t['azure']} Azure lines, ours {t['ours']} | one-to-one {100 * c['one-to-one'] / L:.1f}% | joins a row {100 * c['ours joins a row'] / L:.1f}% "
          f"| joins lines {100 * c['ours joins lines'] / L:.1f}% | splits {100 * c['ours splits'] / L:.1f}% | tangled {100 * c['tangled'] / L:.1f}% | LOST {100 * t['lost'] / L:.2f}%"
          + (f" | text differs {100 * t['edits'] / t['chars']:.2f}% of {t['chars']}" if t["chars"] else "") + f" | lost {t['lost']} | splits {c['ours splits']} | invented {t['invented']}")
    print("   lost by cause: " + ", ".join(f"{x} {t[x]}" for x in CAUSES if t[x]))
    print("   by kind: " + ", ".join(f"{x} lost {100 * v['lost'] / max(v['azure'], 1):.2f}% of {v['azure']}" for x, v in sorted(kinds.items())))
print("\npages that changed most (lost lines):")
for key in sorted(common, key=lambda q: -abs(keys[0][q]["lost"] - keys[1][q]["lost"]) - abs(keys[0][q].get("invented", 0) - keys[1][q].get("invented", 0)))[:20]:
    a, b = keys[0][key], keys[1][key]
    if a["lost"] != b["lost"] or a["ours"] != b["ours"]:
        print(f"   {key[0]} p{key[1]}: lost {a['lost']} -> {b['lost']}, invented {a.get('invented')} -> {b.get('invented')}, ours {a['ours']} -> {b['ours']}, one-to-one {a['cases'].get('one-to-one', 0)} -> {b['cases'].get('one-to-one', 0)}  {b.get('warnings', [])}")
