#!/bin/bash
# Config-based LinkFormer training run INSIDE an existing cluster02 allocation (owner 2026-10-25: a hold job with GPUs that the owner
# submits himself). One run per GPU, started as a job step:
#   srun --jobid=$JID --overlap --gres=gpu:pro6000:1 bash tools/cluster/run_in_alloc.sh configs/<run>.yaml
#   (PIN_GPU=1 srun ... --gres=gpu:pro6000:2 ...: the step sees all GPUs and uses GPU 1 only)
# (run it from ~/xrssd/crossclass on the login node; add e.g. EXTRA="--wandb crossclass --wandb_group v3_diag" WORKERS=4 in front).
# Same environment, caches, wandb dir and logs as tools/cluster/linkformer_cfg.sbatch: logs/linkformer/<tag>.log, the resolved config
# next to the ckpt, caches under the SSD project dir. No sibling guard (no queue copies here); instead the step refuses to start when a
# training of the same tag is already running in this allocation (ps on the node + squeue steps of the job).
set -euo pipefail
CFG=${1:?usage: run_in_alloc.sh configs/<run>.yaml}
: "${SLURM_JOB_ID:?run as a job step: srun --jobid=<hold job> --overlap --gres=gpu:<model>:1 bash tools/cluster/run_in_alloc.sh <cfg>}"
[ -n "${PIN_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$PIN_GPU          # several runs per GPU (2026-10-25): request all GPUs of the job, pin one
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs; K=$CROSSCLASS/.cache
STEP=${SLURM_JOB_ID}_${SLURM_STEP_ID:-0}
export TMPDIR=$K/tmp/$STEP PIP_CACHE_DIR=$K/pip CONDA_PKGS_DIRS=$K/conda HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCH_HOME=$K/torch XDG_CACHE_HOME=$K/xdg
mkdir -p "$TMPDIR" $K/pip $K/conda $K/hf $K/triton $K/torch $K/xdg; trap 'rm -rf "$TMPDIR"' EXIT
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
EXTRA="--workers ${WORKERS:-3} ${EXTRA:-}"
# shellcheck disable=SC2086
TAG=$($PY tools/linkformer.py --config "$CFG" $EXTRA --show_config | awk '$1 == "tag:" {print $2}'); [ -n "$TAG" ] || { echo "no tag from $CFG"; exit 1; }
if pgrep -u "$USER" -f "tools/linkformer.py --config $CFG --device cuda" > /dev/null; then echo "a training of $CFG already runs on $(hostname) -> exit"; exit 1; fi
LOG=logs/linkformer/$TAG.log
echo "job $SLURM_JOB_ID step ${SLURM_STEP_ID:-?} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date -u '+%F %T') UTC" | tee -a "logs/alloc_${TAG}.out"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "logs/alloc_${TAG}.out"
echo "code $(git rev-parse --short HEAD) $(md5sum < tools/linkformer.py | cut -c1-8), config $CFG $(md5sum < "$CFG" | cut -c1-8), extra: $EXTRA" | tee -a "logs/alloc_${TAG}.out"
df -BG --output=avail "$CROSSCLASS" | tail -1 | xargs echo "xrssd free:" | tee -a "logs/alloc_${TAG}.out"
if [ -s "$LOG" ]; then P=$LOG.prev.$(date -u +%Y%m%dT%H%M%SZ); mv "$LOG" "$P"; echo "old log of $TAG kept as $P" | tee -a "logs/alloc_${TAG}.out"; fi   # never empty a log (review #2)
# shellcheck disable=SC2086
$PY tools/linkformer.py --config "$CFG" --device cuda $EXTRA > "$LOG" 2>&1 || echo "linkformer exit code $?" | tee -a "logs/alloc_${TAG}.out"
grep -h "final\|DISK ALERT\|stop_at" "$LOG" | tee -a "logs/alloc_${TAG}.out" || true
