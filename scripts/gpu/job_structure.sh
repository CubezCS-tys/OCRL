#!/bin/bash
# Runs ON the rented GPU, in /work: trains the structure model in two sizes. Writes train.done or train.failed.
cd /work
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" > gpu.txt 2>&1 || { echo nogpu > train.failed; exit 1; }
pip install -q rapidfuzz > pip.log 2>&1
E=${EPOCHS:-12}
PYTHONPATH=src python -u scripts/train_structure.py --epochs $E --batch 64 --out out/structure_small.pt > train.log 2>&1 || { echo small > train.failed; exit 1; }
PYTHONPATH=src python -u scripts/train_structure.py --epochs $E --batch 64 --d 256 --layers 6 --out out/structure_big.pt >> train.log 2>&1 || { echo big > train.failed; exit 1; }
touch train.done
