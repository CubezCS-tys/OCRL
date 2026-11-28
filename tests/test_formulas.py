"""Formulas read as LaTeX by the served vision model (structure/qwen.py read_formulas, doc/parse.py _read_formulas):
answers cleaned and checked, failures counted and left as pictures, the master and the viewer carry the LaTeX."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ocrl.structure import qwen as Q


def test_clean_latex():
    assert Q.clean_latex("```latex\nx^2 + 1\n```") == "x^2 + 1"
    assert Q.clean_latex("$$\\frac{a}{b}$$") == "\\frac{a}{b}"
    assert Q.clean_latex("\\begin{equation*} y=1 \\end{equation*}") == "y=1"
    assert Q.clean_latex("$y(t)=\\varphi(t)$ \\tag{2.2}") == "y(t)=\\varphi(t)  \\tag{2.2}"  # $ round a formula with no words: dropped
    mixed = "حيث $j = 1, \\dots, N$ عدد النقاط"
    assert Q.clean_latex(mixed) == mixed  # Arabic words with formulas between them: kept as text with formulas


def test_latex_ok():
    assert Q.latex_ok("\\frac{1}{2}\\left(x\\right)")
    assert Q.latex_ok("N = \\frac{\\sum t_i}{Cd} \\ldots\\ldots" + "." * 40 + "(2)")  # a leader of dots
    assert not Q.latex_ok("")
    assert not Q.latex_ok("\\frac{1}{2")
    assert not Q.latex_ok("\\left( x")
    assert not Q.latex_ok("\\begin{aligned} x \\end{array}")
    assert not Q.latex_ok("G = 1 \\; " + "\\; " * 40)  # a model looping
    assert not Q.latex_ok("a $b")


class _Fake(BaseHTTPRequestHandler):
    answers = []  # one per request, in turn
    seen = []
    by_width = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        # Formula requests are sent side by side, so they may arrive in either order: when `by_width` is set the
        # answer is chosen by the picture's width (narrow first answer, wide second), not by arrival.
        if type(self).by_width:
            import base64, cv2
            url = body["messages"][0]["content"][0]["image_url"]["url"]
            img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
            ans = type(self).answers[int(img.shape[1] > 1000)]
        else:
            ans = type(self).answers[len(type(self).seen) - 1]
        out = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def _server(answers, by_width=False):
    _Fake.answers, _Fake.seen, _Fake.by_width = answers, [], by_width
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_read_formulas_counts_and_falls_back(tmp_path):
    srv = _server(["$$x^2$$", "\\frac{1}{2"], by_width=True)
    try:
        t = Q.QwenTyper(f"qwen:http://127.0.0.1:{srv.server_port}?workers=1;cache={tmp_path}")
        crops = {"a": np.full((40, 120), 255, np.uint8), "b": np.zeros((40, 3000), np.uint8)}
        got = t.read_formulas(crops)
        assert got == {"a": "x^2"}  # the unbalanced answer is not used: that formula stays a picture
        assert t.doc["formulas"] == 2 and t.doc["formulas_read"] == 1 and t.doc["formulas_unusable"] == 1
        assert _Fake.seen[0]["model"] == "ocrl" and _Fake.seen[0]["messages"][0]["content"][1]["text"] == Q.FORMULA_PROMPT
        assert "Formulas: 1 of 2 read as LaTeX" in t.how()
        assert t.read_formulas(crops) == {"a": "x^2"} and len(_Fake.seen) == 2  # answered from the cache
    finally:
        srv.shutdown()
    off = Q.QwenTyper("qwen:http://127.0.0.1:9?formulas=off")
    assert off.read_formulas({"a": np.zeros((5, 5), np.uint8)}) == {} and not off.doc["formulas"]
    down = Q.QwenTyper("qwen:http://127.0.0.1:9?timeout=2")  # nothing listens there
    assert down.read_formulas({"a": np.zeros((5, 5), np.uint8)}) == {} and down.doc["formulas_failed"] == 1


def _doc():
    from ocrl.doc.parse import Block, PageInfo, TextLine
    eq = [TextLine(1, 0, 900, 600 + 50 * i, 1500, 640 + 50 * i, 635 + 50 * i, 30.0, "x = 1", 5, i) for i in range(2)]
    para = TextLine(1, 0, 300, 300, 2100, 340, 335, 30.0, "نص عادي في الصفحة", 20, 2)
    info = PageInfo(1, 2400, 3300, eq + [para], [(100, 100, 2300, 3200)], [], [])
    blocks = [Block([para], "paragraph", order=1, id="b0"), Block(eq, "equation", order=2, id="b1")]
    return [info], blocks


class _Reader:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def read_formulas(self, crops):
        self.asked.append({k: c.shape for k, c in crops.items()})
        return {k: self.answer for k in crops} if self.answer else {}


def test_master_and_viewer_carry_latex(tmp_path):
    from ocrl.doc import parse as P
    from ocrl.doc.viewer import write_viewer

    page = np.full((3300, 2400), 255, np.uint8)
    infos, blocks = _doc()
    r = _Reader("x = \\frac{1}{2}")
    P._read_formulas(infos, [page], 1, blocks, r)
    assert len(r.asked) == 1 and len(r.asked[0]) == 1  # one equation, one picture; the paragraph is not asked about
    (h, w), = r.asked[0].values()
    assert w == 600 + 1 + 2 * 12 and h == 90 + 1 + 2 * 12  # the viewer's crop: the lines and EQ_MARGIN round them
    m = P.to_master("d", infos, blocks, 300, "r", "s")
    eq = next(b for b in m["blocks"] if b["type"] == "equation")
    assert eq["latex"] == "x = \\frac{1}{2}" and eq["picture"][0]["latex"] == eq["latex"] and "not checked" in eq["latex_by"]
    write_viewer(m, [page], tmp_path / "v.html")
    v = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert 'data-tex="x = \\frac{1}{2}"' in v and "katex" in v and "not checked by a person" in v

    infos, blocks = _doc()
    P._read_formulas(infos, [page], 1, blocks, _Reader(""))  # nothing usable: today's picture, no KaTeX
    m = P.to_master("d", infos, blocks, 300, "r", "s")
    eq = next(b for b in m["blocks"] if b["type"] == "equation")
    assert "latex" not in eq and eq["shown_as"] == "picture"
    write_viewer(m, [page], tmp_path / "w.html")
    v = (tmp_path / "w.html").read_text(encoding="utf-8")
    assert "katex" not in v and "formulas are not transcribed" in v


def test_no_equation_asks_nothing():
    from ocrl.doc import parse as P
    infos, blocks = _doc()
    blocks[1].type = "paragraph"
    r = _Reader("x")
    P._read_formulas(infos, [np.full((3300, 2400), 255, np.uint8)], 1, blocks, r)
    assert r.asked == []
