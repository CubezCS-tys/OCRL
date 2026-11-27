"""Teach Qwen3.5-9B our labelling task with a low-rank adapter (LoRA, bf16 base: the vendor-side guidance for Qwen3.5 is
that 4-bit QLoRA loses too much), using pages Claude Haiku labelled. Each example is exactly the prompt the untrained
model saw in scripts/qwen_label.py (page image + task text + guide + lines, thinking off) with Haiku's JSON as the
answer; the loss counts only the answer tokens.

Documents (folder name before "_p") are split so that about 200 pages of whole documents are held out for validation.
Nothing from dev_inputs is ever read. Run on the rented machine, in the venv that has vLLM, transformers 5, peft and fla:

    /work/qv/bin/python scripts/qwen_finetune.py --train-pages 1200 --out out/qwen_lora/run1
"""
import argparse, json, math, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_label import TASK  # the same task wording, word for word


def split(inputs: Path, answers: Path, val_pages: int, seed: int):
    """(train pages, validation pages): only pages whose Haiku answer uses every line id exactly once; validation is
    whole documents, drawn at random until it holds at least val_pages pages."""
    pages, dropped = [], []
    for d in sorted(p for p in inputs.iterdir() if (p / "lines.json").exists()):
        f = answers / f"{d.name}.json"
        try:
            ans = json.load(open(f, encoding="utf-8"))
            used = [x for b in ans["blocks"] for x in b["lines"]]
        except Exception:
            dropped.append(d.name); continue
        ids = [l["id"] for l in json.load(open(d / "lines.json", encoding="utf-8"))]
        (pages if sorted(used) == sorted(ids) else dropped).append(d.name)
    docs = sorted({p.split("_p")[0] for p in pages}); random.Random(seed).shuffle(docs)
    val_docs, n = set(), 0
    for doc in docs:
        if n >= val_pages: break
        val_docs.add(doc); n += sum(1 for p in pages if p.split("_p")[0] == doc)
    train = [p for p in pages if p.split("_p")[0] not in val_docs]; val = [p for p in pages if p.split("_p")[0] in val_docs]
    random.Random(seed).shuffle(train)
    return train, val, dropped


def example(name, inputs, answers, guide, proc):
    d = inputs / name
    from PIL import Image
    lines = json.load(open(d / "lines.json", encoding="utf-8"))
    content, images = [], []
    for f in sorted(d.glob("page_*.jpg"), key=lambda f: int(f.stem.split("_")[1])):
        content.append({"type": "text", "text": f"Image of page {f.stem.split('_')[1]}:"}); content.append({"type": "image"})
        images.append(Image.open(f).convert("RGB"))
    content.append({"type": "text", "text": TASK.replace("__GUIDE__", guide).replace("__LINES__", "\n".join(f"{l['id']} {l['box']} {l['text']}" for l in lines))})
    prompt = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = json.dumps(json.load(open(answers / f"{name}.json", encoding="utf-8")), ensure_ascii=False) + "<|im_end|>\n"
    enc = proc(text=[prompt + answer], images=images, return_tensors="pt")
    n_prompt = len(proc(text=[prompt], images=images, return_tensors="pt")["input_ids"][0])
    return enc, n_prompt


def loss_of(model, enc, n_prompt, dev):
    import torch
    enc = {k: v.to(dev) for k, v in enc.items()}
    base = model.base_model.model  # peft -> Qwen3_5ForConditionalGeneration
    hs = base.model(**enc).last_hidden_state  # logits only for the answer: the full vocabulary over 5k tokens would not fit
    h = hs[:, n_prompt - 1:-1]; target = enc["input_ids"][:, n_prompt:]
    logits = base.lm_head(h).float()
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1)), target.numel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="out/labeller/bulk_inputs"); ap.add_argument("--answers", default="out/labeller/bulk_haiku")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--out", required=True)
    ap.add_argument("--train-pages", type=int, default=1200); ap.add_argument("--val-pages", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=0, help="validate every this many optimizer steps (0: a quarter of the run)")
    ap.add_argument("--val-sub", type=int, default=60, help="validation pages used in the periodic checks (all of them at the end)")
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=9000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=0, help="stop training after this many minutes and save (0: none)")
    ap.add_argument("--probe", type=int, default=0, help="only time this many training steps, save nothing")
    a = ap.parse_args()
    import torch, transformers
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    inputs, answers, out = ROOT / a.inputs, ROOT / a.answers, ROOT / a.out
    assert "dev_inputs" not in str(inputs)
    guide = (ROOT / "out/labeller/guide.md").read_text(encoding="utf-8")
    train, val, dropped = split(inputs, answers, a.val_pages, a.seed)
    train = train[: a.train_pages]
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train, "val": val, "dropped": dropped}, open(out / "split.json", "w"), indent=0)
    print(f"train {len(train)} pages, validation {len(val)} pages ({len({v.split('_p')[0] for v in val})} documents), dropped {len(dropped)}", flush=True)

    proc = transformers.AutoProcessor.from_pretrained(a.model)
    model = transformers.AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=targets, bias="none"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    dev = torch.device("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_total = max(1, math.ceil(len(train) * a.epochs / a.accum)); warm = max(1, min(20, steps_total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))) * 0.9 + 0.1)
    val_every = a.val_every or max(1, steps_total // 4)
    hist = {"args": vars(a), "train_pages": len(train), "val_pages": len(val), "steps_total": steps_total, "train": [], "val": []}

    def validate(names):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for nm in names:
                enc, n_prompt = example(nm, inputs, answers, guide, proc)
                if enc["input_ids"].shape[1] > a.max_len: continue
                l, k = loss_of(model, enc, n_prompt, dev); tot += l.item() * k; n += k
        model.train(); return tot / max(n, 1)

    order = [train[i % len(train)] for i in range(int(len(train) * a.epochs))]
    if not a.probe:
        v = validate(val[: a.val_sub]); hist["val"].append({"step": 0, "pages": 0, "loss": round(v, 4), "subset": a.val_sub})
        print(f"validation loss before training ({a.val_sub} pages) {v:.4f}", flush=True)
    t0 = time.time(); model.train(); step, acc_loss, acc_tok, skipped, micro = 0, 0.0, 0, 0, 0
    run_loss = []
    for i, nm in enumerate(order):
        enc, n_prompt = example(nm, inputs, answers, guide, proc)
        if enc["input_ids"].shape[1] > a.max_len:
            skipped += 1; continue
        loss, k = loss_of(model, enc, n_prompt, dev)
        (loss / a.accum).backward(); run_loss.append(loss.item()); micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            mean = sum(run_loss) / len(run_loss); run_loss = []
            el = time.time() - t0
            hist["train"].append({"step": step, "pages": i + 1, "loss": round(mean, 4), "lr": sched.get_last_lr()[0], "min": round(el / 60, 1)})
            print(f"step {step}/{steps_total} pages {i+1} loss {mean:.4f} lr {sched.get_last_lr()[0]:.2e} {el/60:.1f} min "
                  f"({el/(i+1):.2f} s/page, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)", flush=True)
            if a.probe and step >= a.probe: print("probe done"); sys.exit(0)
            if step % val_every == 0 and step < steps_total:
                v = validate(val[: a.val_sub]); hist["val"].append({"step": step, "pages": i + 1, "loss": round(v, 4), "subset": a.val_sub})
                print(f"  validation loss ({a.val_sub} pages) {v:.4f}", flush=True)
                json.dump(hist, open(out / "history.json", "w"), indent=1)
            if a.time_limit and el / 60 > a.time_limit:
                print(f"time limit reached after {i+1} pages", flush=True); break
    hist["train_s"] = round(time.time() - t0); hist["skipped_long"] = skipped; hist["pages_seen"] = i + 1
    model.save_pretrained(out / "adapter"); proc.save_pretrained(out / "adapter")
    v = validate(val); hist["val"].append({"step": step, "pages": i + 1, "loss": round(v, 4), "subset": len(val)})
    print(f"final validation loss ({len(val)} pages) {v:.4f}", flush=True)
    json.dump(hist, open(out / "history.json", "w"), indent=1)
