"""Have an open-weights Qwen vision-language model label pages on our own GPU: the same numbered scan, lines, guide and
task wording as scripts/claude_label.py, so its answers score on the same footing as every other labeller.

Runs vLLM in this process (no server), sends every page at once and lets vLLM batch them. An answer is accepted when it
parses as JSON (prose and code fences around it are cut away) and uses every line id exactly once with types from the
list; otherwise the page is asked once more, sampled, and the better of the two answers is kept. A page whose answers
both fail to parse is written as {"blocks": []} so scoring counts all its lines wrong. Run on the rented machine, in the
venv that has vLLM:

    /work/qv/bin/python scripts/qwen_label.py --inputs out/labeller/dev_inputs --out out/labeller/dev_qwen
    ... --thinking            let the model think first (the vendor's default mode)
    ... --max-pixels 4000000  look at the scan in more detail
    ... --schema              constrain decoding to the answer's JSON shape
    ... --lora out/qwen_lora/run1/adapter   the same model with a fine-tuned adapter (scripts/qwen_finetune.py)
"""
import argparse, base64, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# TASK and TYPES are copied word for word from claude_label.py (which cannot be imported: it parses arguments and needs the API key)
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
def messages(d: Path) -> list:
    lines = json.load(open(d / "lines.json", encoding="utf-8"))
    content = []
    for f in sorted(d.glob("page_*.jpg"), key=lambda f: int(f.stem.split("_")[1])):
        content.append({"type": "text", "text": f"Image of page {f.stem.split('_')[1]}:"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.standard_b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": TASK.replace("__GUIDE__", GUIDE).replace("__LINES__", "\n".join(f"{l['id']} {l['box']} {l['text']}" for l in lines))})
    return [{"role": "user", "content": content}]


def extract(text: str):
    """The JSON object in an answer, wherever the model put it. None when there is none."""
    text = re.sub(r"(?s)^.*</think>", "", text)  # any thinking that was not split off
    cands = [m.group(1) for m in re.finditer(r"(?s)```(?:json)?\s*(.*?)```", text)] + [text]
    for c in cands:
        s = c.find("{")
        while s != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(c[s:])
                if isinstance(obj, dict) and isinstance(obj.get("blocks"), list):
                    return obj
            except json.JSONDecodeError:
                pass
            s = c.find("{", s + 1)
    return None


def unprefix(ans):
    """Qwen often copies the template literally and writes "PAGE:5:0" for line 5:0. Strip that one unambiguous slip and say
    whether it was needed, so the report can count it."""
    if ans is None:
        return None, False
    hit = False
    for b in ans["blocks"]:
        if isinstance(b, dict) and isinstance(b.get("lines"), list):
            new = [str(x)[5:] if str(x).startswith("PAGE:") else x for x in b["lines"]]
            hit |= new != b["lines"]; b["lines"] = new
    return ans, hit


def check(ans, ids: list) -> dict:
    """What is wrong with one answer: counts of lines left out, used twice, unknown ids, bad types."""
    if ans is None:
        return {"unparsed": 1}
    blocks = [b for b in ans["blocks"] if isinstance(b, dict) and isinstance(b.get("lines"), list)]
    used = [str(x) for b in blocks for x in b["lines"]]
    return {"missing": len(set(ids) - set(used)), "twice": len(used) - len(set(used)), "unknown": len(set(used) - set(ids)),
            "badtype": sum(1 for b in blocks if b.get("type") not in TYPES), "malformed": len(ans["blocks"]) - len(blocks)}


def clean(ans, ids: list) -> dict:
    """Keep the answer as given, but drop malformed blocks and ids that are not on the page, so the scorer can read it."""
    if ans is None:
        return {"blocks": []}
    known, out = set(ids), []
    for b in ans["blocks"]:
        if isinstance(b, dict) and isinstance(b.get("lines"), list):
            ls = [str(x) for x in b["lines"] if str(x) in known]
            if ls: out.append({"type": str(b.get("type")), "lines": ls})
    return {"blocks": out}


def bad(c: dict) -> int:
    return 10**6 * c.get("unparsed", 0) + sum(v for k, v in c.items() if k not in ("unparsed", "prefixed"))


if __name__ == "__main__":  # vLLM starts its engine with spawn, which imports this file again
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--schema", action="store_true"); ap.add_argument("--max-pixels", type=int, default=0, help="0: the processor's default")
    ap.add_argument("--mem", type=float, default=0.80, help="share of the GPU vLLM may take"); ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora", default="", help="a peft adapter folder to apply (vLLM LoRA)"); ap.add_argument("--lora-rank", type=int, default=16)
    a = ap.parse_args()
    GUIDE = (ROOT / "out/labeller/guide.md").read_text(encoding="utf-8")


    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams as SOP
    except ImportError:
        SOP = None

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = sorted(p for p in Path(a.inputs).iterdir() if (p / "lines.json").exists() and not (out / f"{p.name}.json").exists())
    if a.limit: todo = todo[: a.limit]
    mm = {"max_pixels": a.max_pixels} if a.max_pixels else None
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem, limit_mm_per_prompt={"image": 1, "video": 0},
              mm_processor_kwargs=mm, seed=0, **({"enable_lora": True, "max_lora_rank": a.lora_rank, "max_loras": 1} if a.lora else {}))
    load_s = time.time() - t0
    extra = {}
    if a.schema:
        extra = {"structured_outputs": SOP(json=SCHEMA)} if SOP else {"guided_decoding": __import__("vllm.sampling_params", fromlist=["x"]).GuidedDecodingParams(json=SCHEMA)}
    maxtok = 24000 if a.thinking else 8000
    greedy = SamplingParams(temperature=0.0, max_tokens=maxtok, **extra) if not a.thinking else SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=maxtok, seed=0, **extra)
    sampled = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=maxtok, seed=1, **extra)
    tk = {"enable_thinking": bool(a.thinking)}

    ids = {d.name: [l["id"] for l in json.load(open(d / "lines.json", encoding="utf-8"))] for d in todo}
    convs = [messages(d) for d in todo]


    def run(idx, sp):
        t = time.time()
        lr = None
        if a.lora:
            from vllm.lora.request import LoRARequest
            lr = LoRARequest("ft", 1, str(Path(a.lora).resolve()))
        res = llm.chat([convs[i] for i in idx], sp, chat_template_kwargs=tk, use_tqdm=False, lora_request=lr)
        return res, time.time() - t


    res, gen_s = run(range(len(todo)), greedy)
    best, notes, toks = {}, {}, {"in": 0, "out": 0}
    for i, r in enumerate(res):
        o = r.outputs[0]; ans, pre = unprefix(extract(o.text)); c = {**check(ans, ids[todo[i].name]), "prefixed": int(pre)}
        best[i] = (ans, c, o.text, o.finish_reason); toks["in"] += len(r.prompt_token_ids); toks["out"] += len(o.token_ids)
    again = [i for i, (ans, c, _, _) in best.items() if bad(c)]
    retry_s = 0.0
    if again:
        res2, retry_s = run(again, sampled)
        for i, r in zip(again, res2):
            o = r.outputs[0]; ans, pre = unprefix(extract(o.text)); c = {**check(ans, ids[todo[i].name]), "prefixed": int(pre)}; toks["out"] += len(o.token_ids)
            notes[todo[i].name] = f"retried: first {best[i][1]}, second {c}"
            if bad(c) < bad(best[i][1]):
                best[i] = (ans, c, o.text, o.finish_reason)
    raw = out.parent / (out.name + "_raw"); raw.mkdir(exist_ok=True)
    summary = {"model": a.model, "lora": a.lora, "thinking": a.thinking, "schema": a.schema, "max_pixels": a.max_pixels, "pages": len(todo), "load_s": round(load_s, 1),
               "first_pass_s": round(gen_s, 1), "retry_s": round(retry_s, 1), "retried": len(again), "tokens": toks, "pages_with": {}, "totals": {}, "per_page": {}}
    for i, d in enumerate(todo):
        ans, c, text, fin = best[i]
        (out / f"{d.name}.json").write_text(json.dumps(clean(ans, ids[d.name]), ensure_ascii=False), encoding="utf-8")
        (raw / f"{d.name}.txt").write_text(text, encoding="utf-8")
        summary["per_page"][d.name] = {**c, "finish": fin, "lines": len(ids[d.name]), **({"note": notes[d.name]} if d.name in notes else {})}
        for k, v in c.items():
            summary["totals"][k] = summary["totals"].get(k, 0) + v; summary["pages_with"][k] = summary["pages_with"].get(k, 0) + (v > 0)
    summary["seconds_per_page"] = round((gen_s + retry_s) / max(len(todo), 1), 2)
    json.dump(summary, open(out.parent / (out.name + "_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}, indent=1))
