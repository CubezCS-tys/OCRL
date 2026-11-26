#!/bin/bash
# Runs ON the rented GPU, in /work, after the project was unpacked there. Writes train.done or train.failed.
cd /work
pip install -q "numpy>=2" opencv-python-headless uharfbuzz freetype-py > pip.log 2>&1 || { echo pip > train.failed; exit 1; }
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" > gpu.txt 2>&1 || { echo nogpu > train.failed; exit 1; }
W=$(( $(nproc) - 2 )); [ $W -gt 20 ] && W=20
PYTHONPATH=src python -u scripts/train.py --steps ${STEPS:-40000} --batch ${BATCH:-256} --lr ${LR:-0.003} \
  --workers $W --eval-every 4000 --out out/${OUTNAME:-recognizer_gpu.pt} > train.log 2>&1
[ $? -eq 0 ] && touch train.done || echo train > train.failed
