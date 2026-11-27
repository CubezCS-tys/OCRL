"""The structure step asked of a fine-tuned vision-language model (Qwen3.5-9B with a LoRA adapter, scripts/qwen_finetune.py)
served by vLLM on the GPU machine, with the small pair (tag.Typer) as the fallback for whatever it does not answer.

For one page the model is shown exactly what it was trained on (scripts/labeller_dev_inputs.py, scripts/qwen_label.py):
the page scaled to 1150 pixels wide with a white strip on the right, every one of OUR lines outlined in blue with its
index in red to its right, JPEG quality 82; then the task, the labelling guide and the lines as "PAGE:INDEX [x0, y0,
x1, y1 in thousandths] text". It answers {"blocks": [{"type", "lines": [ids]}]} in reading order. Greedy first; an
answer that does not parse, or leaves lines out, uses one twice or names an unknown id or type, is asked once more,
sampled, and the better of the two kept (as qwen_label.py does).

Then: ids not on the page are dropped, a line used twice is kept where it first stands, a block with a type not in
the list gives its lines to the fallback, and every line left out takes the small pair's type and block start for
that line. A page whose answer cannot be parsed at all, or whose request fails or times out, takes the small pair's
answer for the whole page. Every such case is counted (`stats`, `doc`), and `how()` says it in the master file's
"structure" line, which the viewer shows in its header.

Selected by a typer spec (tag.Typer(spec) returns one of these):
    qwen:http://127.0.0.1:8000+out/structure_v2.pt+out/structure_v3.pt
    qwen:http://127.0.0.1:8000?order=ours;rules=lists,ends+out/structure_v2.pt+out/structure_v3.pt
Options after '?', joined by '&' or ';': model= (the served adapter's name, default ocrl), order=qwen|ours (whose
reading order the page's lines are grouped in; default qwen), rules= (comma list of margin, lists, ends: the small
pair's rules applied on top of Qwen's answer; default ends, "none" for none), workers= (pages sent at once, default 8), timeout= (seconds a request, default 240),
quality= (JPEG quality, default 82, as trained), cache= (a folder keeping answers by request, for experiments).
ocrl.doc.parse calls prepare(infos, pages) before typing, which sends every page of the document at once (vLLM
batches them), then tag() per page as for the small pair, and reorder() for Qwen's reading order.

The defaults (Qwen's order, the paragraph-ends rule on top) are what measured best on the 70 development pages with
our own lines (scripts/e2e_judged.py, 2026-11-27): types 95.6%, blocks exact 87.1% (the small pair 95.4%, 84.5%);
Qwen's answer as it is 95.5%, 85.0%; with the list rule too 95.5%, 87.2%. On the untouched check set, once:
90.8% / 83.8% against the small pair's 86.2% / 75.0%.
Those numbers are adapter run4_pass2 (trained on Azure's lines). The page now serves run5_ours (scripts/gpu/platform.sh
QLORA; run4_pass2 stays selectable), also trained on our own lines: 70 pages 95.6% / 87.9%; check set, once,
91.0% / 83.8%; on Azure's lines 96.2% / 84.9% (run4_pass2 96.1% / 85.4%).
"""

from __future__ import annotations

import base64, hashlib, json, re, threading, time, urllib.error, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import numpy as np

from .tag import _MARGIN_OVER, ASIDE, DEFAULT, TO_TYPE, TYPES, _paragraph_ends, list_items

GUIDE = (Path(__file__).with_name("qwen_guide.md")).read_text(encoding="utf-8")  # out/labeller/guide.md, as trained
# word for word as in scripts/qwen_label.py (tests/test_qwen_typer.py checks it)
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
WIDTH, STRIP = 1150, 70


def page_picture(gray: np.ndarray, boxes, quality: int = 82) -> bytes:
    """The page as the labeller inputs draw it: 1150 wide, a 70-pixel white strip on the right, each line's box
    outlined (BGR 200,120,0) and its index in red just right of it. `boxes` [n, 4] in the page's pixels."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    s = WIDTH / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 0, 0, 0, STRIP, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    sx, sy = WIDTH / gray.shape[1], img.shape[0] / gray.shape[0]
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), (200, 120, 0), 1)
        cv2.putText(img, str(i), (int(x1 * sx) + 3, int((y0 + y1) / 2 * sy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes()


def page_lines(number: int, boxes, texts: list[str], w: float, h: float) -> list[dict]:
    """The lines as lines.json lists them: id PAGE:INDEX, box in thousandths of the page, text."""
    return [{"id": f"{number}:{i}", "box": [round(1000 * x0 / w), round(1000 * y0 / h), round(1000 * x1 / w), round(1000 * y1 / h)], "text": t}
            for i, ((x0, y0, x1, y1), t) in enumerate(zip(boxes, texts))]


def messages(number: int, jpeg: bytes, lines: list[dict]) -> list:
    content = [{"type": "text", "text": f"Image of page {number}:"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode()}},
               {"type": "text", "text": TASK.replace("__GUIDE__", GUIDE).replace("__LINES__", "\n".join(f"{l['id']} {l['box']} {l['text']}" for l in lines))}]
    return [{"role": "user", "content": content}]


def extract(text: str):
    """The JSON object in an answer, wherever the model put it (qwen_label.extract). None when there is none."""
    text = re.sub(r"(?s)^.*</think>", "", text or "")
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


def _ids(x) -> str:
    x = str(x).strip()
    return x[5:] if x.startswith("PAGE:") else x  # Qwen sometimes copies the template: "PAGE:5:0" for 5:0


def faults(ans, ids: list[str]) -> int:
    """How wrong an answer is: lines left out + used twice + unknown ids + bad types + malformed blocks; 10**6 unparsed."""
    if ans is None:
        return 10**6
    blocks = [b for b in ans["blocks"] if isinstance(b, dict) and isinstance(b.get("lines"), list)]
    used = [_ids(x) for b in blocks for x in b["lines"]]
    return (len(set(ids) - set(used)) + len(used) - len(set(used)) + len(set(used) - set(ids))
            + sum(1 for b in blocks if b.get("type") not in TYPES) + len(ans["blocks"]) - len(blocks))


def assign(ans, n: int, number: int) -> tuple[list, Counter]:
    """Per line i of the page: (type, block index, place in Qwen's reading order) or None (left to the fallback)."""
    got: list = [None] * n
    c = Counter()
    if ans is None:
        return got, c
    at = {f"{number}:{i}": i for i in range(n)}
    pos = 0
    for k, b in enumerate(ans["blocks"]):
        if not (isinstance(b, dict) and isinstance(b.get("lines"), list)):
            c["malformed"] += 1
            continue
        t = b.get("type")
        for x in b["lines"]:
            i = at.get(_ids(x))
            if i is None:
                c["unknown"] += 1
            elif got[i] is not None:
                c["twice"] += 1
            elif t not in TYPES:
                c["badtype"] += 1
            else:
                got[i] = (t, k, pos); pos += 1
    return got, c


def _key(boxes, texts) -> tuple:
    """A page's lines, whatever their order: how tag() and reorder() find the answer prepare() got."""
    return tuple(sorted((tuple(int(v) for v in b), t) for b, t in zip(np.asarray(boxes).tolist(), texts)))


class QwenTyper:
    def __init__(self, spec: str, lists: bool = True):
        from .tag import Typer

        head, _, rest = spec.partition("+")
        url, _, q = head[len("qwen:"):].partition("?")
        o = {k: v[-1] for k, v in parse_qs(q.replace(";", "&")).items()}
        self.url = url.rstrip("/")
        self.model = o.get("model", "ocrl")
        self.order = o.get("order", "qwen")
        self.rules = {r for r in o.get("rules", "ends").split(",") if r and r != "none"}
        self.workers = int(o.get("workers", 8))
        self.timeout = float(o.get("timeout", 240))
        self.quality = int(o.get("quality", 82))
        self.cache = Path(o["cache"]) if o.get("cache") else None
        self.fallback_spec = rest or DEFAULT
        self.pair = Typer(self.fallback_spec, lists=lists)
        self.lists = lists
        self._open = False
        self.answers: dict[tuple, dict] = {}
        self.stats = Counter()  # since this typer was made
        self.doc = Counter()  # the last document
        self._lock = threading.Lock()

    # the harness and the upload page reset this for each document
    @property
    def open_item(self):
        return self._open

    @open_item.setter
    def open_item(self, v):
        self._open = v
        self.pair.open_item = v

    # ------------------------------------------------------------------ asking
    def _post(self, body: dict) -> str:
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""

    def _ask(self, msgs: list, sampled: bool) -> str:
        body = {"model": self.model, "messages": msgs, "max_tokens": 8000, "chat_template_kwargs": {"enable_thinking": False}}
        body.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 1} if sampled else {"temperature": 0.0, "seed": 0})
        if self.cache is None:
            return self._post(body)
        f = self.cache / (hashlib.sha1(json.dumps([self.model, body], sort_keys=True).encode()).hexdigest() + ".txt")
        if f.exists():
            return f.read_text(encoding="utf-8")
        text = self._post(body)
        self.cache.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
        return text

    def ask_page(self, number: int, gray: np.ndarray, boxes, texts: list[str]) -> dict:
        """One page: {"ans": the better answer or None, "error": why there is none, "retried": bool, "s": seconds}."""
        t = time.time()
        h, w = gray.shape[:2]
        lines = page_lines(number, boxes, texts, w, h)
        ids = [l["id"] for l in lines]
        msgs = messages(number, page_picture(gray, boxes, self.quality), lines)
        out = {"ans": None, "error": "", "retried": False}
        try:
            ans = extract(self._ask(msgs, False))
            bad = faults(ans, ids)
            if bad:
                out["retried"] = True
                ans2 = extract(self._ask(msgs, True))
                if faults(ans2, ids) < bad:
                    ans = ans2
            out["ans"] = ans
            if ans is None:
                out["error"] = "unparsed"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as e:
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        out["s"] = time.time() - t
        return out

    def prepare(self, infos, pages) -> None:
        """Ask for every page of a document at once. `infos` the PageInfo list, `pages` the grey images, same order."""
        self.answers = {}
        self.doc = Counter()
        jobs = []
        for p, g in zip(infos, pages):
            ls = sorted(p.lines, key=lambda l: (l.region, l.order))  # the order _typed_blocks types them in
            if ls:
                jobs.append((p.number, g, [[l.x0, l.y0, l.x1, l.y1] for l in ls], [l.text for l in ls]))
        if not jobs:
            return
        t = time.time()
        with ThreadPoolExecutor(max(1, min(self.workers, len(jobs)))) as ex:
            res = list(ex.map(lambda j: self.ask_page(*j), jobs))
        for (number, g, boxes, texts), r in zip(jobs, res):
            got, c = assign(r["ans"], len(texts), number)
            self.answers[_key(boxes, texts)] = {"number": number, "got": {(tuple(int(v) for v in b), tx): a for b, tx, a in zip(boxes, texts, got)},
                                                "error": r["error"], "faults": c}
            self.doc.update(pages=1, lines=len(texts), retried=int(r["retried"]), pages_failed=int(r["ans"] is None),
                            lines_fallback=sum(a is None for a in got), **c)
            if r["error"]:
                self.doc["errors_" + r["error"].split(":")[0]] += 1
        self.doc["seconds"] = round(time.time() - t, 2)
        with self._lock:
            self.stats.update(self.doc)

    # ------------------------------------------------------------------ answering
    def _answer(self, boxes, texts):
        return self.answers.get(_key(boxes, texts))

    def reorder(self, ls: list) -> list:
        """With order=qwen: the page's lines in Qwen's reading order; a line it left out stays after the line it
        followed in ours. Otherwise unchanged."""
        if self.order != "qwen" or not ls:
            return ls
        a = self._answer([[l.x0, l.y0, l.x1, l.y1] for l in ls], [l.text for l in ls])
        if a is None or all(v is None for v in a["got"].values()):
            return ls
        rank, last = [], -1.0
        for i, l in enumerate(ls):
            g = a["got"].get(((l.x0, l.y0, l.x1, l.y1), l.text))
            last = g[2] if g is not None else last + 1e-3
            rank.append((last, i))
        return [ls[i] for _, i in sorted(rank)]

    def tag(self, boxes, texts: list[str], page_w: float, page_h: float, ink=None, crops=None, margin: dict | None = None):
        """As tag.Typer.tag: (type per line, P(line starts a block), confidence), lines in the order given."""
        n = len(texts)
        roles_fb, starts_fb, conf_fb = self.pair.tag(boxes, texts, page_w, page_h, ink, crops, margin)  # the fallback, always ready
        a = self._answer(boxes, texts) if n else None
        if a is None:  # never asked (no prepare) or no lines: the small pair's answer
            if n:
                with self._lock:
                    self.stats.update(pages_unasked=1, lines_fallback=n)
                    self.doc.update(pages_unasked=1, lines_fallback=n)
            self._open = self.pair.open_item
            return roles_fb, starts_fb, conf_fb
        got = [a["got"].get((tuple(int(v) for v in b), t)) for b, t in zip(np.asarray(boxes).tolist(), texts)]
        roles, starts, conf = [], np.zeros(n), np.ones(n)
        prev = None  # the block the line before belongs to: ("q", k) for Qwen's, ("f", ...) for the fallback's
        fb_block = 0
        for i, g in enumerate(got):
            if g is None:
                roles.append(TO_TYPE.get(roles_fb[i], "other")); conf[i] = conf_fb[i]
                fb_block += int(starts_fb[i] >= 0.5 or prev is None or prev[0] != "f")
                cur = ("f", fb_block)
            else:
                roles.append(g[0])
                cur = ("q", g[1])
            starts[i] = 1.0 if cur != prev else 0.0
            prev = cur
        # The page's first line of running text: whether it carries on a block from the page before is not something
        # Qwen, seeing one page, can say. The small pair's start stands there (same type: else it is a new block).
        first = next((i for i in range(n) if roles[i] not in ASIDE), None)
        if first is not None and got[first] is not None and starts_fb[first] < 0.5 and TO_TYPE.get(roles_fb[first], "other") == roles[first] \
                and all(got[k] is None or roles[k] in ASIDE for k in range(first)):
            starts[first] = starts_fb[first]
        if "ends" in self.rules:
            latin = sum(c.isascii() and c.isalpha() for t in texts for c in t)
            starts = _paragraph_ends(starts, texts, boxes, ltr=latin > 0.5 * sum(c.isalpha() for t in texts for c in t))
        if "margin" in self.rules:
            for i, kind in (margin or {}).items():
                if roles[i] in _MARGIN_OVER:
                    roles[i] = kind; starts[i] = 1.0
                    if i + 1 < n:
                        starts[i + 1] = max(starts[i + 1], 0.6)
        if self.lists and "lists" in self.rules:
            roles, starts, self._open = list_items(roles, starts, texts, boxes, self._open, single=True)
            roles = [TO_TYPE.get(r, "other") for r in roles]
        return roles, starts, conf

    # ------------------------------------------------------------------ saying what happened
    def how(self) -> str:
        """For the master file's "structure" line (the viewer's header): what typed the pages, and how much fell back."""
        d = self.doc
        fb = d["lines_fallback"]
        who = " + ".join(Path(t).stem for t in self.fallback_spec.split("+"))
        parts = [f"Qwen3.5-9B fine-tuned (vision and text), on the GPU machine itself ({fb} of {d['lines']} lines from the fallback, {who}"]
        if d["pages_failed"]:
            parts[0] += f"; {d['pages_failed']} of {d['pages']} pages wholly"
        if d["pages_unasked"]:
            parts[0] += f"; {d['pages_unasked']} pages never asked"
        parts[0] += ")"
        if self.rules:
            parts.append("with our " + ", ".join(sorted(self.rules)) + " rules on top")
        parts.append(f"reading order: {'Qwen' if self.order == 'qwen' else 'layout'}")
        return ", ".join(parts) + ". Local, nothing sent outside. Not checked by a person."
