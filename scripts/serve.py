"""A local page for trying the parser: drop a PDF on it, see the scan with its blocks beside the parsed document.

Everything runs on this machine: the trained line finder, a line reader and the structure model. Nothing is
sent anywhere. Uploaded files and results are kept under out/uploads/ (not tracked).

    uv run python scripts/serve.py            then open http://localhost:8765
"""
import argparse, json, re, subprocess, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch

from ocrl.doc.parse import parse, to_master
from ocrl.doc.viewer import write_viewer
from ocrl.pipeline import OCR
from ocrl.structure.tag import Typer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--reader", default="/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt")
ap.add_argument("--lines", default="out/linefinder.pt"); ap.add_argument("--typer", default="out/structure_v2.pt")
ap.add_argument("--max-pages", type=int, default=30); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); cv2.setNumThreads(2)
UP = Path("out/uploads"); UP.mkdir(parents=True, exist_ok=True)
print("loading the three models ...", flush=True)
ocr = OCR("out/recognizer_v3.pt", linefinder=a.lines, line_reader=a.reader)
typer = Typer(a.typer)
jobs: dict[str, dict] = {}
one_at_a_time = threading.Lock()  # this machine has little memory to spare: documents are parsed one after another


def work(job_id: str, pdf: Path, first: int, last: int) -> None:
    job = jobs[job_id]
    with one_at_a_time:
        try:
            out = pdf.parent; (out / "pages").mkdir(exist_ok=True)
            n_pdf = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
            first = max(first, 1); last = min(last or n_pdf, n_pdf, first + a.max_pages - 1)
            job.update(state="drawing the pages", pages=last - first + 1, of=n_pdf, first=first, started=time.time())
            subprocess.run(["pdftoppm", "-cropbox", "-r", "300", "-gray", "-png", "-f", str(first), "-l", str(last), str(pdf), str(out / "pages" / "p")], check=True, timeout=900)
            images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted((out / "pages").glob("p-*.png"))]
            job.update(state="finding lines, reading, and working out the structure", started=time.time())
            typer.open_item = False
            t = time.time()
            infos, blocks = parse(ocr, images, first, typer)
            how = f"OCRL structure model ({Path(a.typer).stem}) with the list rule, local. Not checked by a person."
            master = to_master(pdf.stem, infos, blocks, 300, Path(a.reader).parent.name + " line reader", how)
            secs = time.time() - t
            (out / "master.json").write_text(json.dumps(master, ensure_ascii=False, indent=1), encoding="utf-8")
            write_viewer(master, images, out / "view.html", 300, secs)
            job.update(state="done", seconds=round(secs), blocks=len(master["blocks"]))
        except Exception as e:
            traceback.print_exc()
            job.update(state="failed", error=f"{type(e).__name__}: {str(e)[:200]}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCRL: parse a document</title>
<style>
:root { --bg:#f5f7fa; --surface:#fff; --fg:#151b23; --muted:#586374; --line:#d3d9e2; --accent:#1464c8; --good:#1c7448; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#0e1217; --surface:#161b22; --fg:#e5e9ef; --muted:#98a3b3; --line:#2a323e; --accent:#6aa9f0; --good:#62cf9b; --bad:#f28b82; color-scheme:dark; } }
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
.status { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:16px; display:flex; flex-direction:column; gap:8px; }
.bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; } .bar div { height:100%; background:var(--accent); width:0; transition:width .6s; }
.bad { color:var(--bad); } .good { color:var(--good); }
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
    <p class="muted" style="font-size:13px">At most __MAX__ pages at a time. A page takes about three to four seconds.</p>
  </div>
  <div class="status" id="status" hidden></div>
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
let file = null, timer = null;
function choose(f) { if (!f) return; file = f; $("chosen").textContent = f.name + " (" + (f.size / 1048576).toFixed(1) + " MB)"; $("go").disabled = false; }
$("file").onchange = e => choose(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => choose(e.dataTransfer.files[0]));
function show(html) { $("status").hidden = false; $("status").innerHTML = html; }
async function poll(id, name) {
  const j = await (await fetch("/status/" + id)).json();
  if (j.state === "done") { clearInterval(timer); $("go").disabled = false; show(`<b class="good">Done: ${name}</b><span>${j.pages} pages, ${j.blocks} blocks, ${j.seconds} seconds.</span><a href="/view/${id}">Open the parsed document</a>`); history(); location.href = "/view/" + id; return; }
  if (j.state === "failed") { clearInterval(timer); $("go").disabled = false; show(`<b class="bad">That did not work</b><span>${j.error || ""}</span>`); return; }
  const el = j.started ? (Date.now() / 1000 - j.started) : 0, est = (j.pages || 1) * 3.6;
  const part = j.state.startsWith("finding") ? Math.min(95, 100 * el / est) : 3;
  show(`<b>${name}</b><span>${j.state}${j.pages ? " · " + j.pages + " pages" + (j.of > j.pages ? " of " + j.of : "") : ""}</span><div class="bar"><div style="width:${part}%"></div></div>`);
}
$("go").onclick = async () => {
  if (!file) return; $("go").disabled = true; show("<b>Sending the file ...</b>");
  const q = new URLSearchParams({ name: file.name, first: $("first").value || 1, last: $("last").value || 0 });
  const r = await fetch("/upload?" + q, { method: "POST", body: file });
  if (!r.ok) { show(`<b class="bad">Upload refused</b><span>${await r.text()}</span>`); $("go").disabled = false; return; }
  const { id } = await r.json(); const name = file.name;
  timer = setInterval(() => poll(id, name), 1500); poll(id, name);
};
async function history() {
  const h = await (await fetch("/history")).json();
  if (h.length) $("history").innerHTML = h.map(x => `<li><a href="/view/${x.id}">${x.name}</a> <span class="muted">· ${x.when}</span></li>`).join("");
}
history();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def send(self, body: bytes, kind: str = "application/json", code: int = 200):
        self.send_response(code); self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith("text") or kind.endswith("json") else ""))
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self.send(PAGE.replace("__MAX__", str(a.max_pages)).encode(), "text/html")
        if path == "/history":
            rows = []
            for d in sorted(UP.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
                if (d / "view.html").exists():
                    pdf = next(d.glob("*.pdf"), None)
                    rows.append({"id": d.name, "name": pdf.name if pdf else d.name, "when": time.strftime("%d %b %H:%M", time.localtime((d / "view.html").stat().st_mtime))})
            return self.send(json.dumps(rows[:30]).encode())
        m = re.fullmatch(r"/(status|view)/([0-9a-f]{8,20})", path)
        if m and m.group(1) == "status":
            return self.send(json.dumps(jobs.get(m.group(2), {"state": "done"} if (UP / m.group(2) / "view.html").exists() else {"state": "failed", "error": "unknown job"})).encode())
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
        job_id = f"{int(time.time() * 1000):x}"
        name = re.sub(r"[^\w.\-]+", "_", Path(q.get("name", ["document.pdf"])[0]).name) or "document.pdf"
        d = UP / job_id; d.mkdir(parents=True)
        pdf = d / (name if name.lower().endswith(".pdf") else name + ".pdf"); pdf.write_bytes(body)
        jobs[job_id] = {"state": "waiting for the document before it"}
        threading.Thread(target=work, args=(job_id, pdf, int(q.get("first", ["1"])[0] or 1), int(q.get("last", ["0"])[0] or 0)), daemon=True).start()
        self.send(json.dumps({"id": job_id}).encode())


print(f"ready: open http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
