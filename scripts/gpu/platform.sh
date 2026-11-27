#!/bin/bash
# The upload page (scripts/serve.py) on the rented workstation, reached from the Windows browser by an SSH tunnel.
#
#   platform.sh start            use the machine in out/gpu/box.env; if none is up, list offers and stop (never rents)
#   platform.sh start --rent     if none is up, rent the cheapest qualifying RTX 5090 (box.sh up; up to three offers)
#   platform.sh stop             close the tunnel, stop the page's serve.py, delete uploads and results on the machine
#   platform.sh stop --down      ...then destroy the machine (box.sh down), unless another python job of ours runs there
#   platform.sh status           machine, page, tunnel, since when, cost so far
#
# Settings (environment):  LPORT=8090 (laptop end of the tunnel; 8090 reaches Windows from WSL2, 8766 did not)
#   MAXPAGES=120  THREADS=8  CPU=1 (run serve.py on the machine's CPU, for when the GPU is full)
#   DRY=1 (print the rent / destroy commands instead of running them; start first lists the models and the
#          serve.py command it would use, and stops there when no machine is up)
#   "Other jobs" for --down: any python process, or a box.sh run wrapper (bash ... PYTHONUNBUFFERED) between steps.
#
# The page's code goes to /work/platform/{src,scripts}, not /work/src: other jobs on the machine import /work/src
# and must not change under them. It runs in /work, so the models are read from /work/out and the uploads land in
# /work/out/uploads. The remote port is the first free one from 8765; the tunnel binds 127.0.0.1 on both ends.
# What was started is in out/gpu/platform.env (remote serve PID and port, local tunnel PID, start time).
ROOT=$(cd "$(dirname "$0")/../.." && pwd); G=$ROOT/scripts/gpu; OUT=$ROOT/out/gpu; mkdir -p "$OUT"
BOXENV=${BOXENV:-$OUT/box.env}; PENV=${PENV:-$OUT/platform.env};  # overridable only for DRY tests
KEY=$HOME/.ssh/id_ed25519_vast
LPORT=${LPORT:-8090}; MAXPAGES=${MAXPAGES:-120}; THREADS=${THREADS:-8}
READER=/home/yassine/data/inkscript_labelled/experiments/89_final_reader/out/weights/v7/last.pt   # copied, never changed
WINCURL=/mnt/c/Windows/System32/curl.exe
# The models the page needs (laptop file:name in /work/out) and how serve.py is started with them: the structure
# step is the combined typer, two models joined by '+', as serve.py and parse_doc.py use by default.
MODELS=("$ROOT/out/recognizer_v3.pt:recognizer_v3.pt" "$ROOT/out/linefinder.pt:linefinder.pt"
        "$ROOT/out/structure_v2.pt:structure_v2.pt" "$ROOT/out/structure_v3.pt:structure_v3.pt" "$READER:reader_v7.pt")
SERVE_ARGS="--reader out/reader_v7.pt --lines out/linefinder.pt --typer out/structure_v2.pt+out/structure_v3.pt"

say() { echo "platform: $*"; }
die() { echo "platform: $*" >&2; exit 1; }
loadbox() { ID= DPH= T0= HOST= PORT=; [ -f "$BOXENV" ] && . "$BOXENV"; }
loadp() { SPID= RPORT= TPID= TPORT= PT0= PBOX=; [ -f "$PENV" ] && . "$PENV"; }
savep() { printf "SPID=%s\nRPORT=%s\nTPID=%s\nTPORT=%s\nPT0=%s\nPBOX=%s\n" "$SPID" "$RPORT" "$TPID" "$TPORT" "$PT0" "$PBOX" > "$PENV"; }
rsh() { timeout "${2:-60}" "$G/box.sh" sh "$1"; }                       # short command on the machine, in /work
alive() { loadbox; [ -n "$ID" ] && [ -n "$HOST" ] && [ "$(rsh 'echo ok' 40 | tail -1)" = ok ]; }
hours() { python3 -c "print(round(($(date +%s)-${1:-0})/3600, 2))"; }
cost() { python3 -c "print(round(($(date +%s)-${1:-0})/3600*${2:-0}, 2))"; }
# our serve.py on the machine: PID alive and its command line really is the page we started
serve_ok() { [ -n "$SPID" ] && [ "$PBOX" = "$ID" ] && rsh "tr '\\0' ' ' < /proc/$SPID/cmdline 2>/dev/null" | grep -q "platform/scripts/serve.py"; }
# our tunnel on the laptop: PID alive and it is an ssh forwarding our port
tunnel_ok() { [ -n "$TPID" ] && tr '\0' ' ' 2>/dev/null < /proc/$TPID/cmdline | grep -q "^ssh .*-L 127.0.0.1:$TPORT:127.0.0.1:"; }
page_code() {   # HTTP code of the page as the Windows browser sees it (inside WSL if there is no Windows side)
  if [ -x "$WINCURL" ]; then "$WINCURL" -s -o NUL -m 6 -w "%{http_code}" "http://localhost:$LPORT/" 2>/dev/null
  else curl -s -o /dev/null -m 6 -w "%{http_code}" "http://127.0.0.1:$LPORT/" 2>/dev/null; fi
}
run_or_dry() { if [ "${DRY:-0}" = 1 ]; then echo "DRY: would run: $*"; return 0; fi; "$@"; }

close_tunnel() {
  loadp
  if tunnel_ok; then kill "$TPID"; sleep 1; tunnel_ok && kill -9 "$TPID"; say "tunnel closed (PID $TPID)"
  elif [ -n "$TPID" ]; then say "tunnel PID $TPID was not running"; fi
  TPID=; [ -f "$PENV" ] && savep
}

open_tunnel() {
  loadbox; loadp
  if tunnel_ok && [ "$TPORT" = "$LPORT" ]; then say "tunnel already open (PID $TPID)"; return 0; fi
  tunnel_ok && close_tunnel
  ss -ltn "sport = :$LPORT" 2>/dev/null | grep -q LISTEN && die "laptop port $LPORT is taken by something else; free it or use LPORT=8764 (or 9321)"
  # setsid + nohup: the tunnel outlives the shell that ran start. In a script the child is not a group leader,
  # so setsid execs in place and $! is ssh itself.
  setsid nohup ssh -N -i "$KEY" -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=20 \
    -L "127.0.0.1:$LPORT:127.0.0.1:$RPORT" -p "$PORT" "root@$HOST" > "$OUT/platform_tunnel.log" 2>&1 < /dev/null &
  TPID=$!; TPORT=$LPORT; savep
  for i in $(seq 15); do tunnel_ok && ss -ltn "sport = :$LPORT" | grep -q LISTEN && break; sleep 1; done
  tunnel_ok || { cat "$OUT/platform_tunnel.log"; die "the tunnel did not stay up"; }
  say "tunnel open: laptop 127.0.0.1:$LPORT -> machine 127.0.0.1:$RPORT (PID $TPID)"
}

rent() {
  pgrep -f "[v]ast_guard.sh" > /dev/null || die "ops/vast_guard.sh is not running; start it first (box.sh up refuses without it)"
  say "renting: cheapest qualifying RTX 5090 offers"
  "$G/offers.sh" RTX_5090 | tail -n +2 | head -3 > "$OUT/platform_offers.txt"
  [ -s "$OUT/platform_offers.txt" ] || die "no qualifying offer right now"
  while read -r OID P REST; do
    say "trying offer $OID at \$$P/h"
    if [ "${DRY:-0}" = 1 ]; then echo "DRY: would run: $G/box.sh up $OID $P"; continue; fi
    "$G/box.sh" up "$OID" "$P" < /dev/null; rc=$?
    [ $rc = 0 ] && [ -f "$BOXENV" ] && return 0
    [ $rc = 2 ] && { say "offer $OID never started or its link was too slow (box.sh destroyed it); next"; continue; }
    say "offer $OID could not be created (taken meanwhile?) (box.sh up exit $rc); next"
  done < "$OUT/platform_offers.txt"
  [ "${DRY:-0}" = 1 ] && { say "DRY: stopping here (no machine was rented)"; exit 0; }
  die "three offers tried, none came up"
}

start() {
  if [ "${DRY:-0}" = 1 ]; then   # what would be copied and run; contacts no machine and no offer list
    for pair in "${MODELS[@]}"; do src=${pair%%:*}; [ -f "$src" ] || die "missing model $src"; say "DRY: model $src -> /work/out/${pair##*:}"; done
    say "DRY: code src/ scripts/ -> /work/platform"
    say "DRY: would start: python platform/scripts/serve.py $SERVE_ARGS --max-pages $MAXPAGES --threads $THREADS --port <first free from 8765>"
    alive || { say "DRY: no machine is up; stopping here (nothing rented)"; exit 0; }
  fi
  if ! alive; then
    [ -f "$BOXENV" ] && say "out/gpu/box.env names machine $ID but it does not answer over SSH"
    if [ "$1" != "--rent" ]; then
      say "no workstation is up. Cheapest qualifying offers:"; "$G/offers.sh" RTX_5090
      die "not renting on my own. To rent the cheapest and start the page:  scripts/gpu/platform.sh start --rent"
    fi
    [ -f "$BOXENV" ] && die "remove or fix out/gpu/box.env first (it names machine $ID, which does not answer); renting a second one is not done automatically"
    rent; alive || die "the new machine does not answer"
  fi
  loadbox; loadp; T=$(date +%s)
  say "machine $ID ($HOST:$PORT), \$$DPH/h, up $(hours $T0) h"
  if serve_ok; then
    say "the page is already running there (PID $SPID, port $RPORT)"
  else
    [ -n "$SPID" ] && [ "$PBOX" = "$ID" ] && say "the saved page PID $SPID is gone; starting a new one"
    # code: into /work/platform, so jobs that import /work/src are not changed under them
    S=$(mktemp -d); mkdir -p "$S/src" "$S/scripts"
    tar cf - -C "$ROOT" --exclude=__pycache__ src scripts | tar xf - -C "$S"
    rsh "rm -rf /work/platform/src /work/platform/scripts && mkdir -p /work/platform /work/out /work/logs" > /dev/null
    timeout 300 "$G/box.sh" put "$S" /work/platform || { rm -rf "$S"; die "could not upload the code"; }
    rm -rf "$S"
    # models: upload only those missing or different on the machine
    S=$(mktemp -d); need=0
    have=$(rsh "cd /work/out && md5sum ${MODELS[*]##*:} 2>/dev/null")
    for pair in "${MODELS[@]}"; do
      src=${pair%%:*}; dst=${pair##*:}; [ -f "$src" ] || { rm -rf "$S"; die "missing model $src"; }
      if echo "$have" | grep -q "^$(md5sum < "$src" | cut -d' ' -f1)  $dst\$"; then continue; fi
      cp "$src" "$S/$dst"; need=1; say "will upload $dst"
    done
    if [ $need = 1 ]; then timeout 600 "$G/box.sh" put "$S" /work/out || { rm -rf "$S"; die "could not upload the models"; }; else say "models already in place"; fi
    rm -rf "$S"
    # GPU memory: the page needs about 4.5 GB
    FREE=$(rsh "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits" | tail -1 | awk -F', *' '{print $1-$2}')
    DEV="the GPU"; CUDA=""
    if [ "${CPU:-0}" = 1 ]; then DEV="the machine's CPU (CPU=1)"; CUDA="export CUDA_VISIBLE_DEVICES=;"
    elif [ -n "$FREE" ] && [ "$FREE" -lt 5000 ]; then
      die "only $FREE MiB of GPU memory is free (other jobs hold the rest). Wait, or run on the machine's CPU: CPU=1 scripts/gpu/platform.sh start"
    fi
    RPORT=$(rsh "python3 -c \"
import socket
for p in range(8765, 8800):
    s = socket.socket()
    try: s.bind(('127.0.0.1', p)); print(p); break
    except OSError: pass
    finally: s.close()
\"" | tail -1)
    [ -n "$RPORT" ] || die "no free port on the machine"
    say "starting serve.py on $DEV, port $RPORT, up to $MAXPAGES pages, $THREADS threads"
    "$G/box.sh" run platform_serve "echo \$BASHPID > logs/platform_serve.pid; $CUDA export PYTHONPATH=/work/platform/src; exec python platform/scripts/serve.py $SERVE_ARGS --max-pages $MAXPAGES --threads $THREADS --port $RPORT" > /dev/null
    sleep 2; SPID=$(rsh "cat logs/platform_serve.pid" | tail -1); PBOX=$ID; PT0=$(date +%s); TPID=; TPORT=; savep
    for i in $(seq 60); do   # up to 5 minutes for the models to load
      L=$(rsh "grep -a -m1 '^ready' logs/platform_serve.log; ls logs/platform_serve.failed logs/platform_serve.done 2>/dev/null")
      echo "$L" | grep -q '^ready' && break
      echo "$L" | grep -q 'platform_serve\.\(failed\|done\)' && { rsh "tail -20 logs/platform_serve.log"; die "serve.py stopped while loading"; }
      sleep 5
    done
    echo "$L" | grep -q '^ready' || { rsh "tail -20 logs/platform_serve.log"; die "serve.py did not say ready in 5 minutes (it is still running: PID $SPID; platform.sh stop removes it)"; }
    say "serve.py ready (PID $SPID) after $(( $(date +%s) - T )) s"
  fi
  open_tunnel
  for i in $(seq 10); do C=$(page_code); [ "$C" = 200 ] && break; sleep 2; done
  [ "$C" = 200 ] || die "the page does not answer from Windows (HTTP '${C:-none}'); see out/gpu/platform_tunnel.log"
  say "page answers from Windows (HTTP 200); start took $(( $(date +%s) - T )) s"
  echo
  echo "  open:  http://localhost:$LPORT/"
  echo "  price: \$$DPH an hour (machine $ID, up $(hours $T0) h, about \$$(cost $T0 $DPH) so far)"
  echo "  stop:  scripts/gpu/platform.sh stop      (add --down to destroy the machine too)"
}

stop() {
  loadbox; loadp
  close_tunnel; loadp
  if alive; then
    if serve_ok; then
      rsh "kill $SPID; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -0 $SPID 2>/dev/null && kill -9 $SPID; true" > /dev/null
      serve_ok && die "serve.py PID $SPID would not stop"
      say "serve.py stopped on the machine (PID $SPID)"
    elif [ -n "$SPID" ]; then say "serve.py PID $SPID was not running on machine $ID"
    else say "no page of ours was running on machine $ID"; fi
    # only when we started a page there: another serve.py (run by hand) may be using the same folder
    if [ -n "$SPID" ] && [ "$PBOX" = "$ID" ]; then rsh "rm -rf /work/out/uploads" > /dev/null && say "deleted /work/out/uploads on the machine"
    else say "left /work/out/uploads alone (this script started no page there)"; fi
    rm -f "$PENV"
  else
    [ -n "$ID" ] && say "machine $ID does not answer; nothing to stop there" || say "no machine is up"
    rm -f "$PENV"
  fi
  if [ "$1" = "--down" ]; then
    loadbox; [ -n "$ID" ] || die "no machine to destroy"
    if alive; then
      OTHER=$(rsh "ps -eo pid,etime,args | awk '\$3 ~ /(^|\\/)python[0-9.]*\$/ || /PYTHONUNBUFFERE[D]/' | cut -c1-200")
      if [ -n "$OTHER" ]; then
        echo "$OTHER"; die "other python jobs are running on machine $ID (above); not destroying it. Stop them first, or destroy it yourself with box.sh down"
      fi
    fi
    say "destroying machine $ID (\$$DPH/h, up $(hours $T0) h, about \$$(cost $T0 $DPH))"
    run_or_dry "$G/box.sh" down "upload page"
  fi
}

status() {
  loadbox; loadp
  [ -n "$ID" ] || { say "no machine is up (out/gpu/box.env is absent)"; exit 0; }
  if alive; then say "machine $ID ($HOST:$PORT) answers; up since $(date -d @$T0 '+%d %b %H:%M'), $(hours $T0) h, about \$$(cost $T0 $DPH) at \$$DPH/h"
  else say "machine $ID is in out/gpu/box.env but does not answer over SSH"; fi
  if serve_ok; then say "page: serve.py PID $SPID on machine port $RPORT, since $(date -d @$PT0 '+%d %b %H:%M')"
  else say "page: not running"; fi
  if tunnel_ok; then say "tunnel: PID $TPID, laptop port $TPORT; from Windows: HTTP $(page_code) at http://localhost:$TPORT/"
  else say "tunnel: not open"; fi
}

case "$1" in
  start) start "$2" ;;
  stop) stop "$2" ;;
  status) status ;;
  *) sed -n 2,17p "$0"; exit 1 ;;
esac
