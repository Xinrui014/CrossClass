#!/usr/bin/env bash
# Overnight run of tools/pcb21_vlm_class.py (owner 2026-10-19: "run it tonight in the background").
# Waits until L40 GPU 2 or 3 is free (< 2 GB used; GPU 0 is never used), starts the Qwen3.8-27B-FP8 vLLM server on it
# (tools/serve_qwen38_fp8_1gpu.sh, port 9393), runs 21PCB then WACV, then stops ONLY this server (its own process group).
# Gives up at 09:00. Log: logs/vlm_night.log.  Usage: setsid nohup bash tools/vlm_night.sh > /dev/null 2>&1 < /dev/null &
set -u
cd ~/projects/crossclass; export CROSSCLASS=$PWD; LOG=logs/vlm_night.log; mkdir -p logs
say() { echo "$(date '+%F %T') $*" >> $LOG; }
end=$(date -d 'tomorrow 09:00' +%s); G=""
while [ -z "$G" ]; do
  for g in 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 2000 ]; then G=$g; break; fi
  done
  [ -n "$G" ] && break
  [ "$(date +%s)" -gt "$end" ] && { say "no free GPU before 09:00, giving up"; exit 1; }
  sleep 300
done
say "GPU $G free, starting vLLM"
GPUS=$G setsid bash tools/serve_qwen38_fp8_1gpu.sh > logs/vlm_serve.log 2>&1 < /dev/null &
SPID=$!; PGID=$(ps -o pgid= -p $SPID | tr -d ' ')
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:9393/v1/models > /dev/null && break; sleep 10; done
if ! curl -s --max-time 3 http://127.0.0.1:9393/v1/models > /dev/null; then say "server did not come up (see logs/vlm_serve.log)"; kill -- -$PGID; exit 1; fi
say "server up (pgid $PGID)"
for ds in 21pcb wacv; do python3 tools/pcb21_vlm_class.py --dataset $ds --tag v1 >> $LOG 2>&1; say "$ds finished"; done
kill -- -$PGID; sleep 20; say "server stopped; VLM_NIGHT_DONE"
