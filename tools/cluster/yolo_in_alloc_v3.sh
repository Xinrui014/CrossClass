#!/bin/bash
# Detector training (tools/det/train_yolo_kicad_v3.py, fp32-safe trainer tools/det/fp32_trainer.py) as a job step inside an existing
# cluster02 allocation:  srun --jobid=$JID --overlap --gres=gpu:pro6000:2 bash tools/cluster/yolo_in_alloc_v3.sh --name NAME --epochs E --device 0,1 ...
# v3 (2026-10-28, cros-det; main 16:12): = yolo_in_alloc_v2_rs60b.sh calling train_yolo_kicad_v3.py; the log header also records the md5 of
# fp32_trainer.py. Never edit this file while a run uses it (bash reads it while the step runs).
# v2 (2026-10-26, cros-det) = yolo_in_alloc.sh calling train_yolo_kicad_v2.py (+ --val); the guard sees both script versions.
# rs60b copy (2026-10-26 23:5x, option (b) after the 23:18 OOM of rs60 / e60): + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less
# GPU fragmentation -> fewer TaskAlignedAssigner GPU OOMs and CPU fallbacks); the recipe is unchanged.
# (from ~/xrssd/crossclass). Env /projects/_ssd/xrssd/envs/yolo84 (ultralytics 8.4.37 = v22's version, torch 2.11 cu128); caches,
# scratch and the ultralytics settings dir on the SSD project dir; log logs/det_kicad/<name>.log, run folder runs/det_kicad/<name>.
set -euo pipefail
: "${SLURM_JOB_ID:?run as a job step: srun --jobid=<job> --overlap --gres=gpu:<model>:1 bash tools/cluster/yolo_in_alloc_v3.sh ...}"
NAME=$(echo "$@" | sed -n 's/.*--name \([^ ]*\).*/\1/p'); [ -n "$NAME" ] || { echo "need --name"; exit 1; }
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/yolo84/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass; K=$CROSSCLASS/.cache; STEP=${SLURM_JOB_ID}_${SLURM_STEP_ID:-0}
export TMPDIR=$K/tmp/$STEP PIP_CACHE_DIR=$K/pip HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCH_HOME=$K/torch XDG_CACHE_HOME=$K/xdg YOLO_CONFIG_DIR=$K/ultralytics
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" $K/pip $K/hf $K/triton $K/torch $K/xdg $K/ultralytics; trap 'rm -rf "$TMPDIR"' EXIT
cd "$CROSSCLASS"; mkdir -p logs/det_kicad runs/det_kicad
if pgrep -u "$USER" -f "train_yolo_kicad(_v2|_v3)?\.py.*--name $NAME( |$)" > /dev/null; then echo "$NAME already trains on $(hostname) -> exit"; exit 1; fi
LOG=logs/det_kicad/$NAME.log
{ echo "job $SLURM_JOB_ID step ${SLURM_STEP_ID:-?} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date -u '+%F %T') UTC"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  echo "code $(git rev-parse --short HEAD) train_yolo_kicad_v3.py $(md5sum < tools/det/train_yolo_kicad_v3.py | cut -c1-8) fp32_trainer.py $(md5sum < tools/det/fp32_trainer.py | cut -c1-8), PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF, args: $*"; } | tee -a "$LOG"
$PY tools/det/train_yolo_kicad_v3.py "$@" >> "$LOG" 2>&1 || echo "train exit code $?" | tee -a "$LOG"
