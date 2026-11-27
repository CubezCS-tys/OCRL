"""Have a Claude model label pages through the Anthropic API: the same numbered scan, lines and guide as every other labeller.

Two ways to run: `now` sends requests directly (for a small check), `batch` submits everything as one Message Batch
at half price and `collect` fetches the answers when it has ended. The key is read from ~/.config/ocrl/anthropic_key
(or the usual environment variables) and is never printed. Spend is logged per answer.

    uv run --with anthropic python scripts/claude_label.py now     --inputs out/labeller/dev_inputs --out out/labeller/dev_haiku_api
    uv run --with anthropic python scripts/claude_label.py batch   --inputs out/labeller/bulk_inputs --out out/labeller/bulk_haiku
    uv run --with anthropic python scripts/claude_label.py collect --inputs out/labeller/bulk_inputs --out out/labeller/bulk_haiku
"""
import argparse, base64, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

PRICE = {"claude-haiku-5-5": (0.10, 0.50)}  # dollars per million tokens in / out, for prompts under 100K tokens
SPEND = Path("out/labeller/claude_spend.jsonl")
TYPES = ["paragraph", "heading", "list_item", "table", "table_caption", "reference", "footnote", "running_head", "running_foot", "page_number", "noise",
         "title", "section_label", "author", "supervisor", "affiliation", "degree_statement", "standfirst", "abstract", "keywords", "figure",
         "figure_caption", "pull_quote", "equation", "verse", "handwriting", "other"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["blocks"], "properties": {"blocks": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["type", "lines"],
    "properties": {"type": {"type": "string", "enum": TYPES}, "lines": {"type": "array", "items": {"type": "string"}}}}}}}
TASK = """You are labelling scanned document pages (mostly Arabic, some English or French). Say what every line is and how the lines group into blocks. Do not retype any text: cite line ids only.

Each image is one page, with every line the OCR found outlined in blue and its index in small red digits just to the right of the line. The lines are also listed below as: id [x0, y0, x1, y1 in thousandths of the page] text. A line's id is PAGE:INDEX.

Answer with {"blocks": [{"type": "<one type from the guide>", "lines": ["PAGE:INDEX", ...]}, ...]}
- Blocks in reading order, and the lines inside a block in reading order. Arabic reads right to left: on an Arabic page with columns the rightmost column comes first; on an English or French page the leftmost comes first.
- Every line id must appear in exactly one block. Never leave one out and never use one twice.
- One block is one paragraph, one list item, one heading, one reference, one footnote, or one whole table (all its cells in one "table" block, in row order). A page is a single page of a longer document: a paragraph cut off at the top or bottom is still a paragraph.

THE GUIDE

__GUIDE__

LINES

__LINES__
"""
ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["now", "batch", "collect"]); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--effort", default="high", help="measured on 70 judged pages: low 94.6%, medium 96.3%, high 98.2% of line types"); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--cap", type=float, default=8.0, help="stop when logged spend for this model reaches this many dollars"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
GUIDE = Path("out/labeller/guide.md").read_text(encoding="utf-8")


def client():
    f = Path.home() / ".config/ocrl/anthropic_key"
    return anthropic.Anthropic(api_key=f.read_text().strip()) if f.exists() else anthropic.Anthropic()


def spent():
    return sum(json.loads(l)["usd"] for l in SPEND.read_text().splitlines() if l.strip()) if SPEND.exists() else 0.0


def params(d: Path) -> dict:
    lines = json.load(open(d / "lines.json", encoding="utf-8"))
    content = []
    for f in sorted(d.glob("page_*.jpg"), key=lambda f: int(f.stem.split("_")[1])):
        content.append({"type": "text", "text": f"Image of page {f.stem.split('_')[1]}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": TASK.replace("__GUIDE__", GUIDE).replace("__LINES__", "\n".join(f"{l['id']} {l['box']} {l['text']}" for l in lines))})
    return {"model": a.model, "max_tokens": 16000, "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": a.effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def keep(name: str, msg, half: bool) -> str:
    """Write one answer and log what it cost. Returns a short note."""
    pin, pout = PRICE[a.model]; u = msg.usage
    usd = (u.input_tokens * pin + u.output_tokens * pout) / 1e6 * (0.5 if half else 1.0)
    with open(SPEND, "a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "doc": name, "tin": u.input_tokens, "tout": u.output_tokens, "usd": round(usd, 6), "batch": half}) + "\n")
    if msg.stop_reason != "end_turn":
        return f"stopped: {msg.stop_reason}"
    try:
        ans = json.loads(next(b.text for b in msg.content if b.type == "text"))
        ids = {l["id"] for l in json.load(open(Path(a.inputs) / name / "lines.json", encoding="utf-8"))}
        used = [x for b in ans["blocks"] for x in b["lines"]]
        (Path(a.out) / f"{name}.json").write_text(json.dumps(ans, ensure_ascii=False), encoding="utf-8")
        return "ok" + (f", {len(ids - set(used))} lines left out" if ids - set(used) else "") + (f", {len(used) - len(set(used))} used twice" if len(used) != len(set(used)) else "")
    except Exception as e:
        return f"unreadable answer: {type(e).__name__}"


Path(a.out).mkdir(parents=True, exist_ok=True); SPEND.parent.mkdir(parents=True, exist_ok=True)
todo = sorted(p for p in Path(a.inputs).iterdir() if (p / "lines.json").exists() and not (Path(a.out) / f"{p.name}.json").exists())
if a.limit: todo = todo[: a.limit]
c = client(); notes = {}; started = time.strftime("%Y-%m-%d %H:%M:%S")
if a.what == "now":
    def one(d):
        if spent() >= a.cap: return d.name, "refused: cap reached"
        try:
            return d.name, keep(d.name, c.messages.create(**params(d)), False)
        except anthropic.APIStatusError as e:
            return d.name, f"failed: {type(e).__name__} {e.status_code} {str(e.message)[:140]}"
        except anthropic.APIConnectionError as e:
            return d.name, f"failed: {type(e).__name__}"
    with ThreadPoolExecutor(a.workers) as pool:
        for name, note in pool.map(one, todo): notes[name] = note
elif a.what == "batch":
    ids = {}
    # custom ids must be short and plain; keep the mapping beside the answers
    reqs = []
    for k, d in enumerate(todo):
        ids[f"p{k:06d}"] = d.name; reqs.append({"custom_id": f"p{k:06d}", "params": params(d)})
    made = []
    for i in range(0, len(reqs), 400):  # a page is about 0.4 MB as a request; a batch may hold 256 MB
        b = c.messages.batches.create(requests=reqs[i : i + 400]); made.append(b.id)
    json.dump({"batches": made, "ids": ids}, open(Path(a.out) / "_batch.json", "w"))
    print(f"submitted {len(reqs)} pages in {len(made)} batches"); sys.exit()
else:
    meta = json.load(open(Path(a.out) / "_batch.json"))
    for bid in meta["batches"]:
        b = c.messages.batches.retrieve(bid)
        if b.processing_status != "ended":
            print(f"batch {bid[-8:]}: {b.processing_status}, {b.request_counts.processing} still processing, {b.request_counts.succeeded} done"); continue
        for r in c.messages.batches.results(bid):
            name = meta["ids"][r.custom_id]
            if (Path(a.out) / f"{name}.json").exists(): continue
            notes[name] = keep(name, r.result.message, True) if r.result.type == "succeeded" else r.result.type
bad = {n: v for n, v in notes.items() if v != "ok"}
for n, v in list(bad.items())[:12]: print(f"  {n}: {v}")
rows = [json.loads(l) for l in SPEND.read_text().splitlines() if l.strip()] if SPEND.exists() else []
mine = [r for r in rows if r["doc"] in notes and r["when"] >= started]
if mine:
    print(f"{a.model}: {len(notes)} pages, {len(notes) - len(bad)} clean | this run ${sum(r['usd'] for r in mine):.3f}, {1000 * sum(r['usd'] for r in mine) / len(mine):.2f} dollars per 1,000 pages | "
          f"average {sum(r['tin'] for r in mine) / len(mine):.0f} tokens in, {sum(r['tout'] for r in mine) / len(mine):.0f} out | all Claude labelling so far ${spent():.2f}")
else:
    print(f"{len(notes)} pages handled; nothing new was billed")
