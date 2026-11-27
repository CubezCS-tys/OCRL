"""Have a Gemini model label pages: the same numbered scan, lines and guide the Claude labellers get.

Run with a Python that has google-genai (the inkscript environment). The key is read from the environment
or ~/inkscript/.env and is never printed. Every call is logged with its cost; a cap stops the run.

    /home/yassine/inkscript/.venv/bin/python scripts/gemini_label.py --model gemini-4-argon --inputs out/labeller/inputs --out out/labeller/argon
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRICES = {"gemini-4-argon": (2.0, 10.0), "gemini-3.8-flash": (1.0, 5.0)}  # dollars per million tokens in / out, from the owner's own code
SPEND = Path("out/labeller/gemini_spend.jsonl")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=sorted(PRICES)); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=float, default=5.0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--thinking", default="low")
a = ap.parse_args()


def api_key():
    key = os.environ.get("GEMINI_API_KEY")
    env = Path.home() / "inkscript" / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("no GEMINI_API_KEY in the environment or in ~/inkscript/.env")
    return key


def spent():
    return sum(json.loads(l)["usd"] for l in SPEND.read_text().splitlines() if l.strip()) if SPEND.exists() else 0.0


GUIDE = Path("out/labeller/guide.md").read_text(encoding="utf-8")
TASK = """You are labelling scanned Arabic document pages. Say what every line is and how the lines group into blocks. Do not retype any text: cite line ids only.

Each image is one page, with every line the OCR found outlined in blue and its index in small red digits just to the right of the line. The lines are also listed below as: id [x0, y0, x1, y1 in thousandths of the page] text. A line's id is PAGE:INDEX.

Return one JSON object: {"blocks": [{"type": "<one type from the guide>", "lines": ["PAGE:INDEX", ...]}, ...]}
- Blocks in reading order, and the lines inside a block in reading order. Arabic reads right to left: on a page with columns the rightmost column comes first.
- Every line id must appear in exactly one block. Never leave one out and never use one twice.
- One block is one paragraph, one list item, one heading, one reference, one footnote, or one whole table (all its cells in one "table" block). A block may run across columns or onto the next page.
- Use only the type names in the guide, spelled exactly.

THE GUIDE

__GUIDE__

LINES

__LINES__
"""


def one(d: Path):
    from google import genai
    from google.genai import types
    out = Path(a.out) / f"{d.name}.json"
    if out.exists():
        return d.name, "kept", 0.0
    lines = json.load(open(d / "lines.json", encoding="utf-8"))
    pin, pout = PRICES[a.model]
    if spent() + 0.25 > a.cap:
        return d.name, "refused: cap reached", 0.0
    parts = []
    for f in sorted(d.glob("page_*.jpg"), key=lambda f: int(f.stem.split("_")[1])):
        parts.append(types.Part(text=f"Image of page {f.stem.split('_')[1]}:"))
        parts.append(types.Part(inline_data=types.Blob(data=f.read_bytes(), mime_type="image/jpeg"),
                                media_resolution=types.PartMediaResolution(level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH)))
    parts.append(types.Part(text=TASK.replace("__GUIDE__", GUIDE).replace("__LINES__", "\n".join(f"{l['id']} {l['box']} {l['text']}" for l in lines))))
    cfg = types.GenerateContentConfig(max_output_tokens=32000, response_mime_type="application/json", thinking_config=types.ThinkingConfig(thinking_level=a.thinking))
    t0 = time.time()
    try:
        client = genai.Client(api_key=api_key())  # held in a name: an unnamed client is closed before its request is sent
        r = client.models.generate_content(model=a.model, contents=[types.Content(role="user", parts=parts)], config=cfg)
    except Exception as e:  # never print the key
        return d.name, f"failed: {type(e).__name__} {str(e).replace(api_key(), '<key>')[:160]}", 0.0
    um = r.usage_metadata
    tin, tout = um.prompt_token_count or 0, (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    usd = tin / 1e6 * pin + tout / 1e6 * pout
    with open(SPEND, "a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "doc": d.name, "tin": tin, "tout": tout, "usd": round(usd, 6), "secs": round(time.time() - t0, 1)}) + "\n")
    try:
        ans = json.loads(r.text)
        ids = [l["id"] for l in lines]; used = [x for b in ans["blocks"] for x in b["lines"]]
        note = f"ok, {len(ans['blocks'])} blocks" + (f", {len(set(ids) - set(used))} lines left out" if set(ids) - set(used) else "") + (f", {len(used) - len(set(used))} used twice" if len(used) != len(set(used)) else "")
        out.write_text(json.dumps(ans, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        note = f"unreadable answer: {type(e).__name__}"
    return d.name, note, usd


Path(a.out).mkdir(parents=True, exist_ok=True); SPEND.parent.mkdir(parents=True, exist_ok=True)
docs = sorted(p for p in Path(a.inputs).iterdir() if (p / "lines.json").exists())
total = 0.0
with ThreadPoolExecutor(a.workers) as pool:
    for name, note, usd in pool.map(one, docs):
        total += usd
        if not note.startswith("ok") or len(docs) <= 6:
            print(f"  {name}: {note} (${usd:.3f})", flush=True)
print(f"{a.model}: {len(docs)} documents, this run ${total:.2f}, all Gemini labelling so far ${spent():.2f} of ${a.cap:.2f}")
