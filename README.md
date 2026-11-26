# OCRL

Arabic-first OCR. The connected blobs of ink on the page are the map; the rules of
the script are constraints; a small neural network only reads letters.

    page -> blobs -> lines, bodies, satellites      geometry, no learning   ocrl/blobs
         -> letters per cluster                     CNN + BiGRU + CTC       ocrl/recog
         -> checked, spaced, ordered text           script rules            ocrl/rules, ocrl/pipeline.py

A *unit* is a maximal run of joined characters (a sub-word; or a single digit or
punctuation mark). On clean ink a unit is one body blob plus its dots and marks.
Every character in the output carries the ids of the blobs it was read from.

## Layout

| Path | What it holds |
|---|---|
| `src/ocrl/rules/` | Joining classes from Unicode's `ArabicShaping.txt`, unit splitting, forced word breaks, ink signatures |
| `src/ocrl/render/` | HarfBuzz + FreeType page renderer with per-pixel ownership, and scan-style degradation |
| `src/ocrl/blobs/` | Connected components, line finding, body/satellite split, attachment, overlays; `regions.py` finds columns by whitespace |
| `src/ocrl/recog/` | Cluster crops, the recogniser, CTC decoding, the endless training stream |
| `src/ocrl/pipeline.py` | Page image to text, with rule checks and word spacing |
| `scripts/` | `m0_report.py`, `train.py`, `m1_eval.py`, `real_eval.py` and `real_layout.py` (a real scan against a reference reading), `gpu/` (Vast.ai rental) |
| `docs/` | `vision.html`, `milestone1.html`, `real_0013.html`: open in a browser |

## Running

    uv sync
    uv run pytest
    uv run python scripts/m0_report.py            # blob map vs rules, per font -> out/m0/
    uv run python scripts/train.py --steps 6000   # recogniser -> out/recognizer.pt
    uv run python scripts/m1_eval.py              # end-to-end error rates -> out/m1.json

Fonts go in `data/fonts/*.ttf` and the corpus (one `id<TAB>sentence` per line) in
`data/corpus/sentences.txt`; both are kept out of git. `NotoNaskhArabic`,
`IBMPlexSansArabic-Regular` and `Mirza-Regular` are never trained on, and the last
5% of the corpus is test-only.

`train.py` uses a GPU if PyTorch can see one. The project pins the CPU wheel of
PyTorch. `scripts/gpu/run.sh OFFER_ID` rents a Vast.ai machine, uploads the project
and starts training; `scripts/gpu/wait.sh` fetches the model and destroys the
machine. `ops/vast_guard.sh` must be running first.

Latin fonts for English words go in `data/fonts_latin/*.ttf`. Real scans and
reference readings live under `data/real/` and are never uploaded.
