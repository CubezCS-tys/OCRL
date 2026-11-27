#!/bin/bash
# One rented machine used as the workstation for a working session. Heavy jobs run THERE, not on the laptop.
#
#   box.sh up OFFER_ID DPH        rent, wait until it runs (gives up after 7 min), test the link, set it up
#   box.sh push PATH [PATH...]    upload files or folders (paths relative to the project) into /work
#   box.sh put SRC DEST           upload one local folder to an absolute place on the machine
#   box.sh run NAME 'COMMAND'     start COMMAND in /work in the background; log in /work/logs/NAME.log;
#                                 leaves /work/logs/NAME.done or NAME.failed
#   box.sh wait NAME [MINUTES]    block until NAME is done or failed, then print the end of its log
#   box.sh log NAME [LINES]       the end of a job's log
#   box.sh sh 'COMMAND'           run a short command there and print its output
#   box.sh save                   copy every new model, result file and log back to out/box/ (wait does this itself)
#   box.sh pull PATH [PATH...]    copy results back into out/box/ on the laptop
#   box.sh down 'WHAT RAN'        copy /work/out and /work/logs back, destroy the machine, log the cost
#
# Everything on the machine is lost at "down" except what was pulled. Spend is logged in out/gpu/gpu_spend.jsonl.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); V=~/.local/venvs/vast/bin/vastai; OUT=$ROOT/out/gpu; mkdir -p $OUT $ROOT/out/box; LOG=$OUT/gpu_spend.jsonl
BOX=$OUT/box.env; KEY=$HOME/.ssh/id_ed25519_vast
[ -f $BOX ] && . $BOX
SSHO="-i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
rsh() { ssh $SSHO -p "$PORT" "root@$HOST" "$@" 2>&1 | grep -v -i "welcome to vast\|have fun"; return ${PIPESTATUS[0]}; }  # ssh's own exit status, not grep's
need() { [ -n "$ID" ] || { echo "no machine is up (run: box.sh up OFFER DPH)"; exit 1; }; }
upload() {   # one file, in 8 parallel pieces, joined on the far side: $1 local file, $2 remote directory
  local f=$1 b=$(basename $1) d=/tmp/pup.$$; rm -rf $d; mkdir -p $d; split -n 8 -d "$f" $d/$b.part
  for p in $d/$b.part*; do (for t in 1 2 3; do scp -q $SSHO -P $PORT "$p" root@$HOST:$2/ 2>/dev/null && break; sleep 5; done) & done; wait
  rsh "cd $2 && cat $b.part* > $b && rm -f $b.part*"; rm -rf $d
}
case "$1" in
up)
  OID=$2; DPH=${3:-0}
  pgrep -f "[v]ast_guard.sh" > /dev/null || { echo "ops/vast_guard.sh is not running; start it first"; exit 1; }
  ID=$($V create instance $OID --image pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime --disk ${DISK:-100} --ssh --direct --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  [ -z "$ID" ] && { echo "create failed"; exit 1; }
  T0=$(date +%s); echo "{\"event\": \"start\", \"time\": \"$(date -Is)\", \"instance\": $ID, \"offer\": $OID, \"dph\": $DPH, \"job\": \"workstation\"}" >> $LOG
  printf "ID=%s\nDPH=%s\nT0=%s\n" "$ID" "$DPH" "$T0" > $BOX
  ok=0; for i in $(seq 42); do $V show instance $ID --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d and d.get('actual_status')=='running' else 1)" 2>/dev/null && { ok=1; break; }; sleep 10; done
  if [ $ok = 0 ]; then
    $V destroy instance $ID -y > /dev/null
    H=$(python3 -c "print(round(($(date +%s)-$T0)/3600, 3))")
    echo "{\"event\": \"end\", \"time\": \"$(date -Is)\", \"instance\": $ID, \"hours\": $H, \"dph\": $DPH, \"cost\": $(python3 -c "print(round($H*$DPH,3))"), \"jobs\": \"none: never left loading\"}" >> $LOG
    rm -f $BOX; echo "DESTROYED $ID (never started)"; exit 2
  fi
  URL=$($V ssh-url $ID); HP=${URL#ssh://root@}; HOST=${HP%:*}; PORT=${HP#*:}
  printf "ID=%s\nDPH=%s\nT0=%s\nHOST=%s\nPORT=%s\n" "$ID" "$DPH" "$T0" "$HOST" "$PORT" > $BOX
  for i in $(seq 12); do rsh "mkdir -p /work/logs /work/out" > /dev/null 2>&1 && break; sleep 10; done
  head -c 20000000 /dev/urandom > /tmp/linktest.bin; S=$(date +%s.%N); upload /tmp/linktest.bin /work; E=$(date +%s.%N)
  RATE=$(python3 -c "print(round(20/($E-$S),2))"); rsh "rm -f /work/linktest.bin"; echo "link test: $RATE MB/s"
  if ! python3 -c "import sys; sys.exit(0 if $RATE >= ${MINRATE:-0.5} else 1)"; then $0 down "none: link test $RATE MB/s"; exit 2; fi
  # Same absolute paths as on the laptop, so scripts run unchanged.
  rsh "mkdir -p /home/yassine/data /home/yassine/inkscript_v2/data && ln -sfn /work /home/yassine/OCRL && (apt-get update -qq && apt-get install -y -qq poppler-utils > /dev/null 2>&1; pip install -q 'numpy>=2' opencv-python-headless uharfbuzz freetype-py rapidfuzz 2>&1 | tail -1); nproc; free -g | awk 'NR==2{print \$2\" GB RAM\"}'; nvidia-smi --query-gpu=name --format=csv,noheader; which pdftoppm"
  echo "UP $ID $HOST $PORT" ;;
push)
  need; shift; tar czf /tmp/box_push.tgz -C "$ROOT" "$@" && upload /tmp/box_push.tgz /work && rsh "cd /work && tar xzf box_push.tgz && rm box_push.tgz && echo pushed: $*"; rm -f /tmp/box_push.tgz ;;
put)
  need; tar cf /tmp/box_put.tar -C "$2" . && rsh "mkdir -p $3" && upload /tmp/box_put.tar "$3" && rsh "cd $3 && tar xf box_put.tar && rm box_put.tar && echo put: \$(ls | wc -l) entries in $3"; rm -f /tmp/box_put.tar ;;
run)
  need; rsh "cd /work && rm -f logs/$2.done logs/$2.failed && (setsid nohup bash -c 'export PYTHONPATH=/work/src PYTHONUNBUFFERED=1; ($3) > logs/$2.log 2>&1 && touch logs/$2.done || touch logs/$2.failed' > /dev/null 2>&1 < /dev/null &) ; echo started $2" ;;
wait)
  need; M=${3:-60}; T=$(date +%s)
  until rsh "ls /work/logs/$2.done /work/logs/$2.failed 2>/dev/null" | grep -q .; do [ $(( ($(date +%s) - T) / 60 )) -ge $M ] && { echo "still running after $M min"; break; }; sleep 30; done
  rsh "ls /work/logs/$2.done /work/logs/$2.failed 2>/dev/null; tail -${4:-12} /work/logs/$2.log"
  $0 save ;;
save)   # results must never live only on the rented machine: it can vanish without warning
  need; rsh "cd /work && find out logs -maxdepth 2 -type f -size -400M \\( -name '*.pt' -o -name '*.json' -o -name '*.log' \\) -newer /tmp/box_saved 2>/dev/null > /tmp/box_new || find out logs -maxdepth 2 -type f -size -400M \\( -name '*.pt' -o -name '*.json' -o -name '*.log' \\) > /tmp/box_new; touch /tmp/box_saving; tar czf /tmp/box_save.tgz -T /tmp/box_new 2>/dev/null; wc -l < /tmp/box_new" > /tmp/box_save.n
  if [ "$(tail -1 /tmp/box_save.n)" != "0" ]; then scp -q $SSHO -P $PORT root@$HOST:/tmp/box_save.tgz /tmp/box_save.tgz && tar xzf /tmp/box_save.tgz -C $ROOT/out/box && rsh "mv /tmp/box_saving /tmp/box_saved" && echo "saved to out/box: $(tail -1 /tmp/box_save.n) new files"; fi
  rm -f /tmp/box_save.tgz /tmp/box_save.n ;;
log) need; rsh "tail -${3:-12} /work/logs/$2.log; ls /work/logs/$2.done /work/logs/$2.failed 2>/dev/null" ;;
sh) need; rsh "cd /work && export PYTHONPATH=/work/src && $2" ;;
pull)
  need; shift; rsh "cd /work && tar czf /tmp/box_pull.tgz $* 2>/dev/null; ls -la /tmp/box_pull.tgz | awk '{print \$5}'" && scp -q $SSHO -P $PORT root@$HOST:/tmp/box_pull.tgz /tmp/box_pull.tgz && tar xzf /tmp/box_pull.tgz -C $ROOT/out/box && echo "pulled into out/box: $*"; rm -f /tmp/box_pull.tgz ;;
down)
  need; [ -n "$HOST" ] && $0 pull out logs
  $V destroy instance $ID -y > /dev/null
  H=$(python3 -c "print(round(($(date +%s)-$T0)/3600, 3))")
  echo "{\"event\": \"end\", \"time\": \"$(date -Is)\", \"instance\": $ID, \"hours\": $H, \"dph\": $DPH, \"cost\": $(python3 -c "print(round($H*$DPH,3))"), \"jobs\": \"$2\"}" >> $LOG
  rm -f $BOX; tail -1 $LOG; echo "instances now: $($V show instances --raw | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')" ;;
*) sed -n 2,16p "$0" ;;
esac
