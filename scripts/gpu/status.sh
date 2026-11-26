#!/bin/bash
# Progress of the job on the rented GPU: last log lines and GPU use.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); read ID HOST PORT < $ROOT/out/gpu/gpu_instance.txt
timeout 40 ssh -i ~/.ssh/id_ed25519_vast -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" '
cd /work; cat gpu.txt 2>/dev/null; ls train.done train.failed 2>/dev/null
grep -a "eval" train.log | tail -4; grep -a "^step" train.log | tail -1; tail -2 job.out
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>&1 | grep -v -i "welcome to vast\|have fun"
