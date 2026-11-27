"""Have a Claude model label pages through the Anthropic API: the same numbered scan, lines and guide as every other labeller.

Two ways to run: `now` sends requests directly (for a small check), `batch` submits Message Batches at half price
(built and sent --batch-size pages at a time, never all in memory; progress kept in <out>/_batch.json so a rerun resumes,
--est keeps logged spend plus pages in flight under --cap, --retry resends failed pages once) and `collect` fetches the
answers of every ended batch (each batch is read once; per-page notes go to <out>/_notes.jsonl). The key is read from ~/.config/ocrl/anthropic_key
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
ap.add_argument("--batch-size", type=int, default=400); ap.add_argument("--est", type=float, default=0.75, help="batch: expected dollars per 1,000 pages, to keep submitted work under the cap")
ap.add_argument("--retry", action="store_true", help="batch: resubmit, once, pages whose batch was collected but that have no answer")
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
if a.limit and a.what == "now": todo = todo[: a.limit]
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
else:
    # Batches are built, submitted and dropped one at a time (a page is about 0.4 MB as a request), and _batch.json is
    # rewritten after each one, so an interrupted run resumes without sending a page twice.
    # _batch.json: batches (ids, in order), ids (custom id -> page folder), members (batch id -> custom ids),
    # collected (batches whose results were all read; never read twice, so nothing is billed twice in the log),
    # retried (pages sent a second time after an error, expiry or unusable answer; each page gets one retry).
    mf = Path(a.out) / "_batch.json"
    meta = json.load(open(mf)) if mf.exists() else {}
    for k, v in (("batches", []), ("ids", {}), ("members", {}), ("collected", []), ("retried", [])):
        meta.setdefault(k, v)

    def save():
        json.dump(meta, open(str(mf) + ".tmp", "w")); Path(str(mf) + ".tmp").replace(mf)

if a.what == "batch":
    sent = set(meta["ids"].values()); collected = set(meta["collected"])
    pending = sum(len(meta["members"].get(b, [])) for b in meta["batches"] if b not in collected)
    if a.retry:  # pages whose batch was read but that have no answer file: errored, expired or unusable; once each
        done_ids = {c for b in collected for c in meta["members"].get(b, [])}
        failed = {meta["ids"][c] for c in done_ids} - {p.stem for p in Path(a.out).glob("*.json")} - set(meta["retried"])
        todo = [d for d in todo if d.name in failed]
    else:
        todo = [d for d in todo if d.name not in sent]
    if a.limit: todo = todo[: a.limit]
    made = 0
    for i in range(0, len(todo), a.batch_size):
        part = todo[i : i + a.batch_size]
        projected = spent() + (pending + len(part)) * a.est / 1000
        if projected > a.cap:
            print(f"stopping before {len(todo) - i} pages: logged ${spent():.2f} + {pending + len(part)} pages in flight at ${a.est}/1,000 would pass the ${a.cap} cap"); break
        k0 = len(meta["ids"])
        reqs = [{"custom_id": f"p{k0 + k:06d}", "params": params(d)} for k, d in enumerate(part)]
        b = c.messages.batches.create(requests=reqs)
        meta["batches"].append(b.id); meta["members"][b.id] = [r["custom_id"] for r in reqs]
        for r, d in zip(reqs, part): meta["ids"][r["custom_id"]] = d.name
        if a.retry: meta["retried"] += [d.name for d in part]
        save(); del reqs
        made += 1; pending += len(part)
        print(f"batch {made} ({b.id[-8:]}): {len(part)} pages submitted", flush=True)
    print(f"submitted {sum(len(meta['members'][x]) for x in meta['batches'][-made:]) if made else 0} pages in {made} batches; {len(meta['batches'])} batches in all"); sys.exit()
elif a.what == "collect":
    with open(Path(a.out) / "_notes.jsonl", "a") as nf:
        for bid in meta["batches"]:
            if bid in meta["collected"]: continue
            b = c.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                print(f"batch {bid[-8:]}: {b.processing_status}, {b.request_counts.processing} still processing, {b.request_counts.succeeded} done"); continue
            for r in c.messages.batches.results(bid):
                name = meta["ids"][r.custom_id]
                if (Path(a.out) / f"{name}.json").exists(): continue
                notes[name] = keep(name, r.result.message, True) if r.result.type == "succeeded" else r.result.type
                nf.write(json.dumps({"page": name, "batch": bid, "note": notes[name]}) + "\n")
            meta["collected"].append(bid); save()
    print(f"{len(meta['collected'])} of {len(meta['batches'])} batches collected")
bad = {n: v for n, v in notes.items() if v != "ok"}
for n, v in list(bad.items())[:12]: print(f"  {n}: {v}")
rows = [json.loads(l) for l in SPEND.read_text().splitlines() if l.strip()] if SPEND.exists() else []
mine = [r for r in rows if r["doc"] in notes and r["when"] >= started]
if mine:
    print(f"{a.model}: {len(notes)} pages, {len(notes) - len(bad)} clean | this run ${sum(r['usd'] for r in mine):.3f}, {1000 * sum(r['usd'] for r in mine) / len(mine):.2f} dollars per 1,000 pages | "
          f"average {sum(r['tin'] for r in mine) / len(mine):.0f} tokens in, {sum(r['tout'] for r in mine) / len(mine):.0f} out | all Claude labelling so far ${spent():.2f}")
else:
    print(f"{len(notes)} pages handled; nothing new was billed")
