"""A local page for trying the parser: drop a PDF on it, see the scan with its blocks beside the parsed document.

Everything runs on this machine: the trained line finder, a line reader and the structure model. Nothing is
sent anywhere. Uploaded files and results are kept under out/uploads/ (not tracked).

Documents are parsed one after another (this machine has little memory to spare). The page shows, for each
document, which stage it is at, how many pages are done, and a time left worked out from the pace measured
on that document so far; a waiting document shows how many are ahead of it.

    uv run python scripts/serve.py            then open http://localhost:8765
"""
import argparse, itertools, json, queue, re, subprocess, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch

from ocrl.doc.parse import parse, to_master
from ocrl.doc.viewer import write_viewer
from ocrl.pipeline import OCR
from ocrl.structure.tag import DEFAULT as TYPER, Typer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--reader", default="/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt")
ap.add_argument("--lines", default="out/linefinder.pt"); ap.add_argument("--typer", default=TYPER, help="a structure checkpoint, or two joined by '+' (the combined typer)")
ap.add_argument("--max-pages", type=int, default=30); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); cv2.setNumThreads(2)
UP = Path("out/uploads"); UP.mkdir(parents=True, exist_ok=True)
print("loading the three models ...", flush=True)
ocr = OCR("out/recognizer_v3.pt", linefinder=a.lines, line_reader=a.reader)
typer = Typer(a.typer)

# Seconds a page, measured on this laptop (4 threads, 300 dpi). Replaced by each finished document's own pace.
pace = {"draw": 0.2, "read": 6.3, "finish": 0.1}
jobs: dict[str, dict] = {}
pending: "queue.Queue[str]" = queue.Queue()  # first in, first out: documents are parsed one after another
seq = itertools.count(1)
ACTIVE = ("waiting", "drawing", "reading", "structure", "writing")


def pdf_pages(pdf: Path) -> int:
    m = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=60).stdout)
    if not m:
        raise ValueError("could not count the pages of this PDF")
    return int(m.group(1))


def left(job: dict) -> float:
    """Seconds this job still needs, from the pace measured on it so far (or the last document's pace)."""
    n, st = job["pages"], job["state"]
    draw = job.get("draw_pace") or pace["draw"]
    read = job.get("read_pace") or pace["read"]
    finish = pace["finish"] * n + 1
    if st == "waiting":
        return n * (draw + read) + finish
    if st == "drawing":
        return (n - job.get("drawn", 0)) * draw + n * read + finish
    if st == "reading":
        return (n - job.get("done", 0)) * read + finish
    if st in ("structure", "writing"):
        return finish
    return 0.0


def work(job_id: str) -> None:
    job = jobs[job_id]
    pdf = Path(job["pdf"]); out = pdf.parent; pages_dir = out / "pages"; pages_dir.mkdir(exist_ok=True)
    first, last, n = job["first"], job["last"], job["pages"]
    try:
        began = time.time()
        job.update(state="drawing", drawn=0, began=began, stage_began=began)
        images = []
        for i, p in enumerate(range(first, last + 1), 1):
            # One page at a time, so the page can say how far it got. Plain grey pixels (PGM): the same
            # pixels as the PNG this used to write, without the time spent compressing a file read once.
            stem = pages_dir / f"p-{p:04d}"
            subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-f", str(p), "-l", str(p), "-singlefile", str(pdf), str(stem)],
                           check=True, timeout=300, capture_output=True)
            f = stem.with_suffix(".pgm")
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"page {p} could not be drawn")
            images.append(img); f.unlink()
            job.update(drawn=i, draw_pace=(time.time() - began) / i)

        t = time.time()
        job.update(state="reading", done=0, stage_began=t)

        def progress(done: int, total: int) -> None:
            job.update(done=done, read_pace=(time.time() - t) / done)
            if done == total:
                job.update(state="structure", stage_began=time.time())

        typer.open_item = False
        infos, blocks = parse(ocr, images, first, typer, progress=progress)
        how = f"OCRL structure model ({' + '.join(Path(t).stem for t in a.typer.split('+'))}) with the list rule, local. Not checked by a person."
        master = to_master(pdf.stem, infos, blocks, 300, Path(a.reader).parent.name + " line reader", how)
        secs = time.time() - t
        job.update(state="writing", stage_began=time.time())
        (out / "master.json").write_text(json.dumps(master, ensure_ascii=False, indent=1), encoding="utf-8")
        tw = time.time()
        write_viewer(master, images, out / "view.html", 300, secs)
        try:
            pages_dir.rmdir()  # the drawn pages were removed once read; the PDF itself is kept
        except OSError:
            pass
        end = time.time()
        pace.update(draw=job["draw_pace"], read=job["read_pace"], finish=max((end - tw) / n, 0.02))
        job.update(state="done", seconds=round(end - began), parse_seconds=round(secs), blocks=len(master["blocks"]), ended=end)
        print(f"{job_id} {job['name']}: {n} pages in {end - began:.0f} s (drawing {job['draw_pace']:.2f} s a page, "
              f"reading {job['read_pace']:.2f} s a page)", flush=True)
    except Exception as e:
        traceback.print_exc()
        job.update(state="failed", error=f"{type(e).__name__}: {str(e)[:200]}", ended=time.time())


def worker() -> None:
    while True:
        job_id = pending.get()
        try:
            work(job_id)
        finally:
            pending.task_done()


threading.Thread(target=worker, daemon=True).start()


def public(job_id: str) -> dict:
    """What the page is told about one job: where it is, and for a waiting one, what is ahead of it."""
    j = jobs[job_id]
    out = {k: v for k, v in j.items() if k != "pdf"}
    out.update(id=job_id, now=time.time())
    if j["state"] in ACTIVE:
        out["left"] = round(left(j))
        out["pace_measured"] = bool(j.get("read_pace"))
    if j["state"] == "waiting":
        ahead = sorted((o for o in jobs.values() if o["state"] in ACTIVE and o["seq"] < j["seq"]), key=lambda o: o["seq"])
        out["ahead"] = len(ahead)
        out["starts_in"] = round(sum(left(o) for o in ahead))
        running = next((o for o in ahead if o["state"] != "waiting"), None)
        if running:
            out["current"] = {k: running.get(k) for k in ("name", "state", "pages", "drawn", "done", "first", "last")}
            out["current"]["left"] = round(left(running))
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCRL: parse a document</title>
<style>
:root { --bg:#f5f7fa; --surface:#fff; --fg:#151b23; --muted:#586374; --line:#d3d9e2; --accent:#1464c8; --good:#1c7448; --bad:#b3261e; --note:#8a5a00; }
@media (prefers-color-scheme: dark) { :root { --bg:#0e1217; --surface:#161b22; --fg:#e5e9ef; --muted:#98a3b3; --line:#2a323e; --accent:#6aa9f0; --good:#62cf9b; --bad:#f28b82; --note:#e8b45a; color-scheme:dark; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 "Segoe UI",system-ui,sans-serif; }
main { max-width:860px; margin:0 auto; padding:40px 20px 80px; display:flex; flex-direction:column; gap:24px; }
h1 { font-size:27px; line-height:1.15; margin:0; font-weight:600; text-wrap:balance; } h2 { font-size:17px; margin:0; font-weight:600; }
p { margin:0; max-width:68ch; } .muted { color:var(--muted); }
.drop { background:var(--surface); border:2px dashed var(--line); border-radius:8px; padding:36px 20px; display:flex; flex-direction:column; align-items:center; gap:12px; text-align:center; }
.drop.over { border-color:var(--accent); background:color-mix(in srgb, var(--accent) 8%, var(--surface)); }
button, .pick { font:600 14px "Segoe UI",system-ui,sans-serif; color:var(--bg); background:var(--fg); border:1.5px solid var(--fg); border-radius:6px; padding:8px 18px; cursor:pointer; }
button:focus-visible, .pick:focus-within, input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
button:disabled { opacity:.5; cursor:default; }
.row { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; justify-content:center; }
input[type=number] { width:76px; font:inherit; padding:5px 8px; border:1.5px solid var(--line); border-radius:6px; background:var(--surface); color:var(--fg); }
#jobs { display:flex; flex-direction:column; gap:12px; } #jobs:empty { display:none; }
.status { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:16px; display:flex; flex-direction:column; gap:6px; }
.status .top { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; align-items:baseline; }
.status .big { font-size:17px; font-weight:600; }
.bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:4px 0; } .bar div { height:100%; background:var(--accent); width:0; transition:width .6s; }
.waiting .bar div { background:var(--muted); }
.note { color:var(--note); border-left:3px solid var(--note); padding-left:10px; }
.bad { color:var(--bad); } .good { color:var(--good); }
.num { font-variant-numeric:tabular-nums; }
ul { margin:0; padding-left:20px; display:flex; flex-direction:column; gap:4px; }
a { color:var(--accent); }
.vh { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
</style>
</head>
<body>
<main>
  <header style="display:flex;flex-direction:column;gap:10px">
    <h1>Parse a document, on this machine</h1>
    <p class="muted">Drop a scanned PDF below. Its pages are read here by three models: one finds the lines, one reads them, one says what each line is. Nothing leaves this computer.</p>
  </header>
  <div class="drop" id="drop">
    <p id="chosen">Drop a PDF here</p>
    <div class="row">
      <label class="pick">Choose a file<input class="vh" id="file" type="file" accept="application/pdf,.pdf"></label>
      <label class="muted">from page <input id="first" type="number" min="1" value="1"></label>
      <label class="muted">to page <input id="last" type="number" min="1" placeholder="last"></label>
      <button id="go" disabled>Parse it</button>
    </div>
    <p class="muted" style="font-size:13px">At most __MAX__ pages at a time; for a longer PDF, parse it in parts with the page boxes. A page takes about __PACE__ seconds, so __MAX__ pages take about __FULL__ minutes. Documents are parsed one after another.</p>
  </div>
  <div id="jobs" aria-live="polite"></div>
  <section style="display:flex;flex-direction:column;gap:8px">
    <h2>Parsed before</h2>
    <ul id="history"><li class="muted">Nothing yet.</li></ul>
  </section>
  <section style="display:flex;flex-direction:column;gap:8px">
    <h2>What to expect</h2>
    <ul class="muted">
      <li>Printed Arabic in one or two columns: lines, text and reading order are mostly right.</li>
      <li>Tables are found cell by cell but read column by column. English pages are ordered like Arabic ones.</li>
      <li>Vowelled text, handwriting and filled forms are not handled yet.</li>
    </ul>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let file = null;
const watching = new Map();   // job id -> {timer, sawRunning}

// The ids of jobs started from this tab live in the address (#jobs=a,b), so a reload picks them up.
function mine() { const m = location.hash.match(/jobs=([0-9a-f,]+)/); return m ? m[1].split(",").filter(Boolean) : []; }
function setMine(ids) { history.replaceState(null, "", ids.length ? "#jobs=" + ids.join(",") : location.pathname); }

function dur(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return s + " s";
  const m = Math.floor(s / 60), r = s % 60;
  if (m >= 60) return Math.floor(m / 60) + " h " + (m % 60) + " min";
  return m + " min" + (r ? " " + r + " s" : "");
}
function about(s) { return s < 60 ? "under a minute" : "about " + dur(Math.round(s / 10) * 10); }
function rangeNote(j) {
  if (!j.cut) return "";
  const next = j.last + 1, to = Math.min(j.asked_last || j.of, j.last + j.limit);
  return `<p class="note">This PDF has ${j.of} pages. Only pages ${j.first} to ${j.last} are parsed now (at most ${j.limit} at a time). `
       + `For the rest, choose the file again and set “from page” to ${next}${to > next ? " and “to page” to " + to : ""}.</p>`;
}
const STAGE = { drawing: "Drawing the pages from the PDF", reading: "Finding and reading the lines", structure: "Working out the structure", writing: "Writing the result page" };

function card(id) {
  let el = $("job-" + id);
  if (!el) { el = document.createElement("div"); el.id = "job-" + id; el.className = "status"; $("jobs").append(el); }
  return el;
}

function render(j) {
  const el = card(j.id), name = esc(j.name || "document");
  el.classList.toggle("waiting", j.state === "waiting");
  if (j.state === "done") {
    el.innerHTML = `<b class="good big">Done: ${name}</b>${(() => { const f = [j.pages && j.pages + " pages", j.blocks != null && j.blocks + " blocks", j.seconds && "parsed in " + dur(j.seconds)].filter(Boolean); return f.length ? "<span>" + f.join(", ") + ".</span>" : ""; })()}${rangeNote(j)}<a href="/view/${j.id}">Open the parsed document</a>`;
    return;
  }
  if (j.state === "failed") { el.innerHTML = `<b class="bad big">That did not work: ${name}</b><span>${esc(j.error)}</span>`; return; }
  const range = j.pages ? `pages ${j.first} to ${j.last}` : "";
  if (j.state === "waiting") {
    const waited = j.now - j.queued, ahead = j.ahead || 0;
    let cur = "";
    if (j.current) {
      const c = j.current, pos = c.state === "drawing" ? `drawing page ${c.drawn || 0} of ${c.pages}` : c.state === "reading" ? `${c.done || 0} of ${c.pages} pages read` : (STAGE[c.state] || c.state).toLowerCase();
      cur = `<span>Now parsing <b>${esc(c.name)}</b>: ${pos}, ${about(c.left)} left.</span>`;
    }
    el.innerHTML = `<div class="top"><b class="big">Waiting: ${name}</b><span class="muted num">${range} · waiting ${dur(waited)}</span></div>`
      + `<span>${ahead === 1 ? "One document is" : ahead + " documents are"} ahead of this one. It should start in ${about(j.starts_in)} and then take ${about(j.left)}.</span>`
      + cur + `<div class="bar"><div style="width:0"></div></div>` + rangeNote(j);
    return;
  }
  const el2 = j.now - j.began, n = j.pages;
  let pos, frac;
  if (j.state === "drawing") { pos = `drawing page ${j.drawn || 0} of ${n}`; frac = 0.04 * (j.drawn || 0) / n; }
  else if (j.state === "reading") { pos = `${j.done || 0} of ${n} pages read`; frac = 0.04 + 0.93 * (j.done || 0) / n; }
  else { pos = `all ${n} pages read`; frac = j.state === "structure" ? 0.97 : 0.99; }
  const how = j.pace_measured ? `at ${j.read_pace.toFixed(1)} s a page, measured on this document` : "a first guess, until the first page is read";
  el.innerHTML = `<div class="top"><b class="big">${name}</b><span class="muted num">${range}</span></div>`
    + `<span>${STAGE[j.state] || j.state}: <b class="num">${pos}</b></span>`
    + `<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(100 * frac)}"><div style="width:${(100 * frac).toFixed(1)}%"></div></div>`
    + `<span class="muted num">${dur(el2)} so far · ${about(j.left)} left (${how})</span>` + rangeNote(j);
}

async function poll(id) {
  let j;
  try { const r = await fetch("/status/" + id); j = await r.json(); }
  catch (e) { const el = card(id); el.innerHTML = `<b class="bad">The page lost touch with the parser.</b><span class="muted">It may have been restarted. Trying again ...</span>`; return; }
  j.id = id;
  const w = watching.get(id);
  if (j.state === "done" || j.state === "failed") {
    clearInterval(w.timer); watching.delete(id);
    const left = mine().filter(x => x !== id); setMine(left);
    render(j); loadHistory();
    // Open the result straight away only if this page watched it being parsed and nothing else from here is still going.
    if (j.state === "done" && w.sawRunning && !left.some(x => watching.has(x))) location.href = "/view/" + id;
    return;
  }
  w.sawRunning = true; render(j);
}
function watch(id) {
  if (watching.has(id)) return;
  const w = { sawRunning: false, timer: null }; watching.set(id, w); card(id);
  w.timer = setInterval(() => poll(id), 1500); poll(id);
}

function choose(f) { if (!f) return; file = f; $("chosen").textContent = f.name + " (" + (f.size / 1048576).toFixed(1) + " MB)"; $("go").disabled = false; }
$("file").onchange = e => choose(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => choose(e.dataTransfer.files[0]));
$("go").onclick = async () => {
  if (!file) return; $("go").disabled = true;
  const tmp = card("sending"); tmp.innerHTML = `<b class="big">Sending ${esc(file.name)} ...</b>`;
  const q = new URLSearchParams({ name: file.name, first: $("first").value || 1, last: $("last").value || 0 });
  let r;
  try { r = await fetch("/upload?" + q, { method: "POST", body: file }); } catch (e) { r = null; }
  tmp.remove(); $("go").disabled = false;
  if (!r || !r.ok) { const el = card("refused"); el.innerHTML = `<b class="bad big">Upload refused</b><span>${esc(r ? await r.text() : "The parser could not be reached.")}</span>`; return; }
  $("job-refused")?.remove();
  const j = await r.json();
  setMine([...mine(), j.id]); render(Object.assign({ state: "waiting", now: Date.now() / 1000, queued: Date.now() / 1000, ahead: 0, starts_in: 0, left: 0 }, j)); watch(j.id);
};
async function loadHistory() {
  const h = await (await fetch("/history")).json();
  if (h.length) $("history").innerHTML = h.map(x => `<li><a href="/view/${x.id}">${esc(x.name)}</a> <span class="muted">· ${x.when}</span></li>`).join("");
}
// On load: pick up this tab's jobs (from the address) and anything still running or waiting, from any tab.
(async () => {
  const ids = new Set(mine());
  try { for (const j of await (await fetch("/jobs")).json()) ids.add(j.id); } catch (e) {}
  [...ids].sort().forEach(watch);
})();
loadHistory();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def send(self, body: bytes, kind: str = "application/json", code: int = 200):
        self.send_response(code); self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith("text") or kind.endswith("json") else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            full = a.max_pages * (pace["draw"] + pace["read"] + pace["finish"]) / 60
            page = PAGE.replace("__MAX__", str(a.max_pages)).replace("__PACE__", f"{pace['draw'] + pace['read']:.0f}").replace("__FULL__", f"{full:.0f}")
            return self.send(page.encode(), "text/html")
        if path == "/history":
            rows = []
            for d in sorted((d for d in UP.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True):
                if (d / "view.html").exists():
                    pdf = next(d.glob("*.pdf"), None)
                    rows.append({"id": d.name, "name": pdf.name if pdf else d.name, "when": time.strftime("%d %b %H:%M", time.localtime((d / "view.html").stat().st_mtime))})
            return self.send(json.dumps(rows[:30]).encode())
        if path == "/jobs":  # documents still running or waiting, in the order they will be parsed
            ids = sorted((i for i, j in list(jobs.items()) if j["state"] in ACTIVE), key=lambda i: jobs[i]["seq"])
            return self.send(json.dumps([public(i) for i in ids]).encode())
        m = re.fullmatch(r"/(status|view)/([0-9a-f]{8,20})", path)
        if m and m.group(1) == "status":
            if m.group(2) in jobs:
                return self.send(json.dumps(public(m.group(2))).encode())
            d = UP / m.group(2)
            done = (d / "view.html").exists()  # parsed before this server started
            pdf = next(d.glob("*.pdf"), None) if done else None
            info = {"state": "done", "name": pdf.name if pdf else ""}
            if done and (d / "master.json").exists():
                try:
                    mj = json.loads((d / "master.json").read_text(encoding="utf-8"))
                    info.update(pages=len(mj["pages"]), blocks=len(mj["blocks"]))
                except Exception:
                    pass
            return self.send(json.dumps(info if done else {"state": "failed", "error": "This job is not known here (the parser may have been restarted)."}).encode())
        if m and (UP / m.group(2) / "view.html").exists():
            return self.send((UP / m.group(2) / "view.html").read_bytes(), "text/html")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/upload":
            return self.send(b"not found", "text/plain", 404)
        q = parse_qs(u.query); n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= 200 * 1048576:
            return self.send(b"The file must be a PDF under 200 MB.", "text/plain", 400)
        body = self.rfile.read(n)
        if not body.startswith(b"%PDF"):
            return self.send(b"That file is not a PDF.", "text/plain", 400)
        try:
            want_first = max(int(q.get("first", ["1"])[0] or 1), 1); want_last = int(q.get("last", ["0"])[0] or 0)
        except ValueError:
            return self.send(b"The page numbers must be whole numbers.", "text/plain", 400)
        job_id = f"{int(time.time() * 1000):x}"
        name = re.sub(r"[^\w.\-]+", "_", Path(q.get("name", ["document.pdf"])[0]).name) or "document.pdf"
        d = UP / job_id; d.mkdir(parents=True)
        pdf = d / (name if name.lower().endswith(".pdf") else name + ".pdf"); pdf.write_bytes(body)
        try:
            n_pdf = pdf_pages(pdf)
        except Exception:
            return self.send(b"This PDF could not be opened, so its pages could not be counted.", "text/plain", 400)
        if want_first > n_pdf:
            return self.send(f"This PDF has only {n_pdf} pages, so it cannot start at page {want_first}.".encode(), "text/plain", 400)
        asked_last = min(want_last or n_pdf, n_pdf)
        last = min(asked_last, want_first + a.max_pages - 1)
        if last < want_first:
            return self.send(b"The last page comes before the first.", "text/plain", 400)
        jobs[job_id] = {"state": "waiting", "seq": next(seq), "name": Path(q.get("name", ["document.pdf"])[0]).name, "pdf": str(pdf),
                        "first": want_first, "last": last, "pages": last - want_first + 1, "of": n_pdf, "limit": a.max_pages,
                        "cut": last < asked_last, "asked_last": asked_last, "queued": time.time()}
        pending.put(job_id)
        self.send(json.dumps(public(job_id)).encode())


print(f"ready: open http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
