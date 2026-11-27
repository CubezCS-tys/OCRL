"""The Qwen structure client (ocrl.structure.qwen): same prompt as trained, answers cleaned, failures fall back, counted."""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocrl.structure import qwen as Q  # noqa: E402

PAIR = ROOT / "out/structure_v2.pt"
needs_models = pytest.mark.skipif(not (PAIR.exists() and (ROOT / "out/structure_v3.pt").exists()), reason="structure models not on this machine")


def test_prompt_is_the_trained_one():
    import qwen_label
    assert Q.TASK == qwen_label.TASK
    g = ROOT / "out/labeller/guide.md"
    if g.exists():
        assert Q.GUIDE == g.read_text(encoding="utf-8")


def test_assign_cleans():
    ans = {"blocks": [{"type": "heading", "lines": ["3:0"]}, {"type": "paragraph", "lines": ["PAGE:3:1", "3:2", "3:1", "9:9"]},
                      {"type": "nonsense", "lines": ["3:3"]}, "junk"]}
    got, c = Q.assign(ans, 5, 3)
    assert [g and g[0] for g in got] == ["heading", "paragraph", "paragraph", None, None]
    assert c == {"twice": 1, "unknown": 1, "badtype": 1, "malformed": 1}
    assert Q.extract('noise ```json\n{"blocks": []}\n``` more') == {"blocks": []}
    assert Q.extract("no json here") is None


class _Fake(BaseHTTPRequestHandler):
    answer = ""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        out = json.dumps({"choices": [{"message": {"content": type(self).answer}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def _page():
    from ocrl.doc.parse import PageInfo, TextLine
    ls = [TextLine(1, 0, 100, 100 + 60 * i, 2000, 140 + 60 * i, 135 + 60 * i, 30.0, f"سطر رقم {i} من النص", 10, i) for i in range(4)]
    return PageInfo(1, 2400, 3300, ls, [(100, 100, 2000, 400)], [], []), np.full((3300, 2400), 255, np.uint8)


@needs_models
@pytest.mark.parametrize("answer,fallback", [('{"blocks": [{"type": "heading", "lines": ["1:0"]}, {"type": "paragraph", "lines": ["1:1", "1:2"]}]}', 1),
                                             ("I cannot do that", 4)])
def test_typer_end_to_end_with_fallback(answer, fallback):
    from ocrl.doc.parse import _typed_blocks
    from ocrl.structure.tag import Typer

    _Fake.answer = answer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = Typer(f"qwen:http://127.0.0.1:{srv.server_port}+out/structure_v2.pt+out/structure_v3.pt")
        assert isinstance(t, Q.QwenTyper)
        info, gray = _page()
        t.prepare([info], [gray])
        blocks = _typed_blocks([info], t)
        assert sum(len(b.lines) for b in blocks) == 4
        assert t.doc["lines_fallback"] == fallback and t.doc["retried"] == 1
        if fallback == 1:
            assert blocks[0].type == "heading" and len(blocks[1].lines) == 2
        else:
            assert t.doc["pages_failed"] == 1
        assert f"{fallback} of 4 lines from the fallback" in t.how()
    finally:
        srv.shutdown()


@needs_models
def test_unreachable_server_falls_back():
    from ocrl.doc.parse import _typed_blocks
    from ocrl.structure.tag import Typer

    t = Typer("qwen:http://127.0.0.1:9?timeout=2")
    info, gray = _page()
    t.prepare([info], [gray])
    blocks = _typed_blocks([info], t)
    assert sum(len(b.lines) for b in blocks) == 4 and t.doc["pages_failed"] == 1 and t.doc["lines_fallback"] == 4
