#!/bin/bash
# Wait for train.done / train.failed on the rented GPU (or a hard stop MAXMIN minutes after this starts), fetch the
# checkpoint and logs into out/gpu/, then destroy the instance and log its cost.   [MAXMIN=90] scripts/gpu/wait.sh
ROOT=$(cd "$(dirname "$0")/../.." && pwd); OUT=$ROOT/out/gpu
read ID HOST PORT < $OUT/gpu_instance.txt
[ -f $OUT/gpu_job.env ] && . $OUT/gpu_job.env
SSH="ssh -i $HOME/.ssh/id_ed25519_vast -p $PORT -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@$HOST"
T0=$(date +%s); MAXMIN=${MAXMIN:-90}
until $SSH "ls /work/train.done /work/train.failed 2>/dev/null | grep -q ." 2>/dev/null || [ $(( ($(date +%s) - T0) / 60 )) -ge $MAXMIN ]; do sleep 60; done
$SSH "cd /work && ls train.done train.failed 2>/dev/null; tar czf result.tgz out/*.pt train.log job.out gpu.txt pip.log 2>/dev/null; ls -la result.tgz"
scp -q -i ~/.ssh/id_ed25519_vast -P $PORT root@$HOST:/work/result.tgz $OUT/ && tar xzf $OUT/result.tgz -C $OUT && echo fetched
DPH=${DPH:-0} bash "$ROOT/scripts/gpu/end.sh" "${JOB:-recogniser training}"
