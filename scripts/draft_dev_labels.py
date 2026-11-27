"""Draft labels for the development pages, and the page for correcting them.

Each page drawn by the census gets a first answer from the structure model, run on Azure's lines: what
every line is, and where blocks begin. The owner corrects those drafts in out/census/review.html, which
needs no server: corrections live in the browser and are exported as one JSON file.

    uv run python scripts/draft_dev_labels.py
"""
import json
from pathlib import Path

import numpy as np

from ocrl.structure.tag import TO_TYPE, Typer

OUT = Path("out/census")
# The owner's fixed list of block types (inkscript_v2), each with the draft rule from docs/label_guide.md.
TYPES = [
    ("paragraph", "Paragraph", "#56606e", "p", "Running text."),
    ("heading", "Heading", "#a3341f", "h", "A heading inside the text, at any level."),
    ("list_item", "List item", "#c2590a", "l", "One item of a list, all its lines: it opens with a dash, a bullet, a number or a letter mark."),
    ("table", "Table", "#0b7285", "t", "Every line inside a table: cells and header rows."),
    ("table_caption", "Table caption", "#4c5fd5", "c", "The title or caption of a table."),
    ("reference", "Reference", "#5a6f1e", "r", "One entry of a reference list or bibliography, all its lines."),
    ("footnote", "Footnote", "#607d8b", "f", "A note at the foot of the page or the end of the text, with its mark."),
    ("running_head", "Running head", "#8a7f66", "u", "Text repeated at the top of pages: journal, issue, article title."),
    ("running_foot", "Running foot", "#9c8f6b", "o", "Text repeated at the foot of pages."),
    ("page_number", "Page number", "#7b8190", "g", "The printed page number."),
    ("noise", "Scan marks", "#9aa0ab", "n", "Not text: scan marks, stamps, show-through, specks that were read as words."),
    ("title", "Title", "#b4232a", "", "The title of the article, thesis or book, where it is set as the title."),
    ("section_label", "Section label", "#d1495b", "", "A short label above the title naming the section or series."),
    ("author", "Author", "#1f7a46", "", "The line naming who wrote it."),
    ("supervisor", "Supervisor", "#3a9d6b", "", "The line naming a thesis supervisor."),
    ("affiliation", "Affiliation", "#2563a8", "", "Where the author or supervisor works: post, department, university."),
    ("degree_statement", "Degree statement", "#5b8bd0", "", "The statement of the degree a thesis was submitted for."),
    ("standfirst", "Standfirst", "#8e5aa8", "", "The short introduction set between a magazine title and its text."),
    ("abstract", "Abstract", "#7048a8", "a", "The summary of the article, in any language."),
    ("keywords", "Keywords", "#9a6fc4", "k", "The keywords line."),
    ("figure", "Figure", "#0f8a5f", "i", "Text inside a picture, chart or diagram."),
    ("figure_caption", "Figure caption", "#3fae84", "", "The caption of a picture, chart or diagram."),
    ("pull_quote", "Pull quote", "#b0578d", "q", "Display text that repeats a point from the running text."),
    ("equation", "Equation", "#00838f", "e", "A formula set on its own line."),
    ("verse", "Verse", "#a66a00", "v", "Lines of poetry."),
    ("handwriting", "Handwriting", "#6d4c41", "w", "Writing added by hand."),
    ("other", "Other", "#6b6b6b", "x", "Anything that fits none of the above."),
]
KINDS = {k["kind"]: k for k in json.load(open(OUT / "kinds.json", encoding="utf-8"))["kinds"]}
NAMES = json.loads((OUT / "census_data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
KIND_NAME = {k["kind"]: k["name"] for k in NAMES["kinds"]}
typer = Typer("out/structure_v2.pt")
pages = []
for d in json.load(open(OUT / "dev_pages.json", encoding="utf-8")):
    W, H = d["width"], d["height"]
    boxes = np.array([l["box"] for l in d["lines"]]) * 300
    texts = [l["text"] for l in d["lines"]]
    typer.open_item = False
    roles, starts, conf = typer.tag(boxes, texts, W * 300, H * 300)
    kind = int(d["img"].split("_")[1][1:])
    lines = [{"b": [round(l["box"][0] / W, 4), round(l["box"][1] / H, 4), round(l["box"][2] / W, 4), round(l["box"][3] / H, 4)], "t": l["text"],
              "type": TO_TYPE.get(r, "other"), "start": bool(s >= 0.5 or i == 0), "sure": bool(c >= 0.7)}
             for i, (l, r, s, c) in enumerate(zip(d["lines"], roles, starts, conf))]
    pages.append({"id": f"{d['stem']}_p{d['page']}", "stem": d["stem"], "page": d["page"], "kind": KIND_NAME.get(kind, str(kind)), "img": "dev/" + d["img"], "lines": lines})
# one page of each kind first, so that stopping early still leaves every kind covered
order = sorted(range(len(pages)), key=lambda i: (int(pages[i]["img"].rsplit("_", 1)[1].split(".")[0]), i))
pages = [pages[i] for i in order]
data = {"types": [{"id": t, "name": n, "color": c, "key": k, "help": h} for t, n, c, k, h in TYPES], "pages": pages}
(OUT / "dev_data.js").write_text("const DEV = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
guide = ["# Label guide (draft, 2026-11-27)", "",
         "Every line Azure found on a page gets one of these types, and lines are grouped into blocks. This draft was written by Claude from the owner's fixed type list;",
         "the owner corrects it, and the corrected guide is what any labelling model is given.", "", "| Type | What it is |", "|---|---|"]
guide += [f"| `{t}` | {h} |" for t, n, c, k, h in TYPES]
guide += ["", "## Blocks", "", "- A block is one paragraph, one list item, one reference, one footnote, one heading, or one whole table.",
          "- A block that runs over a column or page break is still one block.", "- Every line belongs to exactly one block.", "",
          "## Open questions for the owner", "",
          "1. A paragraph that opens with a dash, such as a question in a list of research questions: list item (the hand keys say yes) or paragraph?",
          "2. A numbered heading (`١- مقدمة`): heading, not list item?", "3. The word `ملخص` or `Abstract` on its own line above an abstract: heading, or part of the abstract?",
          "4. Notes printed directly under a table: table, footnote, or paragraph?", "5. A table of contents: list items, a table, or other?"]
Path("docs/label_guide.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
(OUT / "review.html").write_text(Path("docs/label_review.html").read_text(encoding="utf-8"), encoding="utf-8")  # the tool sits beside its data and images
unsure = sum(not l["sure"] for p in pages for l in p["lines"]); total = sum(len(p["lines"]) for p in pages)
print(f"{len(pages)} pages, {total:,} lines drafted, {unsure:,} of them marked unsure ({100 * unsure / total:.0f}%)")
