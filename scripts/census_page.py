"""Write the census page (out/census/census.html) from out/census/kinds.json. Private page images: out/ is not tracked."""
import json
from pathlib import Path

OUT = Path("out/census")
k = json.load(open(OUT / "kinds.json", encoding="utf-8"))
# Names and a first judgement of how the frozen parser copes, from looking at the example pages (2026-11-27).
NAME = {13: ("Plain single column", "ok", "Running text, comfortable margins."),
        7: ("Dense single column", "ok", "Long lines, small margins, often inside a ruled frame."),
        10: ("Long justified paragraphs", "ok", "Few paragraph breaks, very even left edge."),
        4: ("Large or heavy type", "ok", "Taller pages and bigger letters: older and book-like print."),
        0: ("Lists and references", "partly", "More than a third of the lines open with a number, dash or bullet."),
        9: ("Latin-script pages", "weak", "English or French text. We still read columns and tables right to left."),
        6: ("Text with a table", "weak", "Half the lines sit in rows of short cells. Cells are found; a table is not yet read row by row."),
        5: ("Two columns", "ok", "Every line is under half the page width."),
        8: ("Small text block", "ok", "Wide margins all round: small-format journals scanned on a larger sheet."),
        11: ("Article first page", "partly", "Title, authors, abstract, then text. Front-matter types are the structure model's weaker ones."),
        2: ("Two dense columns and magazine pages", "partly", "About 70 lines a page; includes magazine layouts with pictures."),
        3: ("Tables of numbers", "weak", "Nearly all lines are cells; a fifth of the characters are digits."),
        1: ("Mixed: tables, charts, shaded cells", "weak", "Display type next to tables and figures."),
        12: ("Title and cover pages", "partly", "About ten centred lines and little else.")}
LABEL = {"lines": "lines on the page", "words_per_line": "words per line", "line_height": "line height (share of page height)", "height_spread": "tallest lines against typical",
         "ink_share": "share of the page covered by lines", "top_margin": "top margin", "bottom_margin": "bottom margin", "side_margin": "side margin", "columns": "columns",
         "in_rows": "lines sitting in rows of short cells", "short_lines": "lines under half a column wide", "centred": "centred lines", "latin": "Latin letters",
         "digits": "digits", "marked": "lines opening with a list mark", "low_conf": "words Azure doubts", "handwritten": "handwriting (Azure's flag)",
         "lines_per_par": "lines per Azure paragraph", "first_page": "is the document's first page", "tall_page": "page height against width",
         "big_type_lines": "lines in much bigger type", "ragged": "unevenness of the line starts"}
for kd in k["kinds"]:
    kd["name"], kd["copes"], kd["note"] = NAME.get(kd["kind"], (f"Kind {kd['kind']}", "partly", ""))
    kd["traits"] = [[LABEL[n], med, al] for n, _, med, al in kd["traits"][:4]]
    kd.pop("medians", None)
k.pop("map", None); k.pop("names", None)
(OUT / "census_data.js").write_text("const CENSUS = " + json.dumps(k, ensure_ascii=False) + ";\n", encoding="utf-8")
(OUT / "census.html").write_text(r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What kinds of page are in the corpus</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
:root { --bg:#f5f7fa; --surface:#fff; --fg:#151b23; --muted:#586374; --line:#d3d9e2; --ok:#1c7448; --partly:#9a5b00; --weak:#b3261e; --bar:#3b6fb5;
  --display:"Space Grotesk","Segoe UI",system-ui,sans-serif; --body:"Inter","Segoe UI",system-ui,sans-serif; }
@media (prefers-color-scheme: dark) { :root { --bg:#0e1217; --surface:#161b22; --fg:#e5e9ef; --muted:#98a3b3; --line:#2a323e; --ok:#62cf9b; --partly:#e0a84a; --weak:#f28b82; --bar:#6aa9f0; color-scheme:dark; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 var(--body); }
main { max-width:1120px; margin:0 auto; padding:36px 20px 80px; display:flex; flex-direction:column; gap:30px; }
h1,h2,h3 { font-family:var(--display); font-weight:600; margin:0; text-wrap:balance; }
h1 { font-size:29px; line-height:1.15; } h2 { font-size:20px; } h3 { font-size:17px; }
p { margin:0; max-width:74ch; } .lede { color:var(--muted); font-size:16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }
.tile { background:var(--surface); border:1.5px solid var(--line); border-radius:5px; padding:14px 16px; }
.tile .big { font:600 30px/1.2 var(--display); font-variant-numeric:tabular-nums; } .tile .sub { color:var(--muted); font-size:13.5px; }
.tile.ok { border-color:var(--ok); } .tile.ok .big { color:var(--ok); } .tile.partly .big { color:var(--partly); } .tile.weak .big { color:var(--weak); }
.bars { display:grid; grid-template-columns:max-content minmax(0,1fr) max-content; gap:6px 12px; align-items:center; font-size:14px; }
.bars a { color:inherit; text-decoration:none; } .bars a:hover { text-decoration:underline; }
.bars .t { height:16px; } .bars .b { height:100%; border-radius:0 4px 4px 0; min-width:2px; }
.bars .v { font-variant-numeric:tabular-nums; text-align:right; font-weight:600; }
.b.ok { background:var(--ok); } .b.partly { background:var(--partly); } .b.weak { background:var(--weak); }
.key { display:flex; flex-wrap:wrap; gap:18px; font-size:13.5px; color:var(--muted); }
.key i { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:6px; }
section.kind { background:var(--surface); border:1px solid var(--line); border-radius:5px; padding:16px; display:flex; flex-direction:column; gap:12px; scroll-margin-top:12px; }
.head { display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 16px; }
.head .pct { font:600 22px var(--display); font-variant-numeric:tabular-nums; }
.badge { font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; border:1.5px solid currentColor; border-radius:999px; padding:1px 10px; }
.badge.ok { color:var(--ok); } .badge.partly { color:var(--partly); } .badge.weak { color:var(--weak); }
.meta { color:var(--muted); font-size:13.5px; }
table { border-collapse:collapse; font-size:13.5px; } th,td { text-align:left; padding:3px 16px 3px 0; border-top:1px solid var(--line); }
th { color:var(--muted); font-weight:500; font-size:12px; } td.n { font-variant-numeric:tabular-nums; text-align:right; }
.thumbs { display:flex; gap:8px; overflow-x:auto; padding-bottom:4px; }
.thumbs figure { margin:0; flex:none; } .thumbs img { height:230px; width:auto; display:block; background:#fff; border:1px solid var(--line); }
.thumbs figcaption { font-size:11px; color:var(--muted); margin-top:2px; }
</style>
</head>
<body>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
  <h1>What kinds of page are in the corpus</h1>
  <p class="lede">A first census, made from Azure's line and word boxes alone: no images were read and nothing was labelled. Each page is described by 22 numbers (how many lines, how wide, how many columns, how much sits in rows, which script) and pages that look alike are grouped. The pictures were only drawn afterwards, to show what each group is.</p>
</header>
<div class="tiles" id="tiles"></div>
<section style="display:flex;flex-direction:column;gap:12px">
  <h2>The kinds, by share of pages</h2>
  <p class="key"><span><i style="background:var(--ok)"></i>the frozen parser copes</span><span><i style="background:var(--partly)"></i>copes in part</span><span><i style="background:var(--weak)"></i>weak or not handled</span><span>Colour is my judgement from tonight's work, not a measurement.</span></p>
  <div class="bars" id="bars"></div>
</section>
<div id="kinds" style="display:flex;flex-direction:column;gap:16px"></div>
<section style="display:flex;flex-direction:column;gap:10px">
  <h2>What this is not yet</h2>
  <p>4,400 documents is under half a percent of the corpus, and the sample was drawn at random, so rare kinds (handwriting, forms, receipts, verse, sideways tables) are too few to form groups of their own. The grouping is also coarse: 14 groups were asked for, and a different number would split or join some of them. And there is no time axis, because the files carry no publication year.</p>
</section>
</main>
<script src="census_data.js"></script>
<script>
const C = CENSUS, pct = x => (100 * x).toFixed(1) + "%";
const share = c => C.kinds.filter(k => k.copes === c).reduce((s, k) => s + k.share, 0);
document.getElementById("tiles").innerHTML =
  `<div class="tile"><div class="big">${C.pages.toLocaleString()}</div><div class="sub">pages, from ${C.documents.toLocaleString()} documents in ${C.journals.toLocaleString()} journals</div></div>` +
  [["ok", "of pages are kinds the parser copes with"], ["partly", "are kinds it copes with in part"], ["weak", "are kinds it is weak on"]].map(([c, t]) => `<div class="tile ${c}"><div class="big">${pct(share(c))}</div><div class="sub">${t}</div></div>`).join("");
const widest = Math.max(...C.kinds.map(k => k.share));
document.getElementById("bars").innerHTML = C.kinds.map(k => `<a href="#k${k.kind}">${k.name}</a><div class="t"><div class="b ${k.copes}" style="width:${100 * k.share / widest}%"></div></div><div class="v">${pct(k.share)}</div>`).join("");
const fmt = v => Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
document.getElementById("kinds").innerHTML = C.kinds.map(k => `
  <section class="kind" id="k${k.kind}">
    <div class="head"><span class="pct">${pct(k.share)}</span><h3>${k.name}</h3><span class="badge ${k.copes}">${k.copes === "ok" ? "copes" : k.copes === "partly" ? "in part" : "weak"}</span>
      <span class="meta">${k.pages.toLocaleString()} pages in ${k.journals.toLocaleString()} journals</span></div>
    <p>${k.note}</p>
    <table><tr><th>What sets it apart</th><th>This kind</th><th>All pages</th></tr>${k.traits.map(([n, a, b]) => `<tr><td>${n}</td><td class="n">${fmt(a)}</td><td class="n">${fmt(b)}</td></tr>`).join("")}</table>
    <div class="thumbs">${k.examples.map(e => `<figure><img loading="lazy" alt="Example page" src="img/${e.img}"><figcaption>${e.stem} p${e.page}</figcaption></figure>`).join("")}</div>
  </section>`).join("");
</script>
</body>
</html>
''', encoding="utf-8")
by = {}
for kd in k["kinds"]: by[kd["copes"]] = by.get(kd["copes"], 0) + kd["share"]
print({c: f"{100 * v:.1f}%" for c, v in by.items()})
