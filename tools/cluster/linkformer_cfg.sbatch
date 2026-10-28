#!/bin/bash
# Config-based LinkFormer training run on cluster02 (task 29, 2026-10-25): ONE model, ONE GPU per job (DDP gave no speed-up for these
# runs), regular rose QOS only (no killable QOS, no resume runs). The run is fully defined by a configs/*.yaml file (tools/linkformer.py
# --config); EXTRA adds or overrides flags (explicit flags win over the config). Logs: logs/linkformer/<tag>.log (+ logs/%x_%j.out);
# ckpts: the config's save path under ~/xrssd/crossclass (ckpt/). The resolved config is written next to the ckpt by linkformer.py.
# Cluster rules (github.com/NTUEEECluster/docs, 2026-10-25): GPU jobs set ONLY a typed --gres=gpu:<model>:1 and --time (max 3 days);
# NEVER --mem / --cpus-per-task / --exclusive: 4 CPUs + a fixed RAM come with each GPU, so the DataLoader uses WORKERS=3 (default) and
# the prefetch thread. Caches and scratch go to the SSD project dir. Submit e.g.
#   sbatch --job-name=<tag of the config> --gres=gpu:l40:1 --time=1-00:00:00 \
#          --export=ALL,CFG=configs/v3_E2TF_img_rope4_nogeo_nosize_detachtext125k_eager.yaml,EXTRA="--wandb crossclass --wandb_group v3_img" \
#          tools/cluster/linkformer_cfg.sbatch
#   --job-name must be the run's tag (review #3, 2026-10-28: one name = one run). The same run may be queued on several GPU types with
#   that name: the sibling guard below keeps only one copy.
#SBATCH --job-name=linkformer_cfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=1-00:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs; K=$CROSSCLASS/.cache
export TMPDIR=$K/tmp/$SLURM_JOB_ID PIP_CACHE_DIR=$K/pip CONDA_PKGS_DIRS=$K/conda HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCH_HOME=$K/torch XDG_CACHE_HOME=$K/xdg
mkdir -p "$TMPDIR" $K/pip $K/conda $K/hf $K/triton $K/torch $K/xdg; trap 'rm -rf "$TMPDIR"' EXIT
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
: "${CFG:?set CFG=configs/<run>.yaml with --export}"; EXTRA="--workers ${WORKERS:-3} ${EXTRA:-}"   # 4 CPUs per GPU: 3 workers + main
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "code $(git rev-parse --short HEAD) $(md5sum < tools/linkformer.py | cut -c1-8), config $CFG $(md5sum < "$CFG" | cut -c1-8), extra: $EXTRA"
df -BG --output=avail "$CROSSCLASS" | tail -1 | xargs echo "xrssd free:"
# shellcheck disable=SC2086
TAG=$($PY tools/linkformer.py --config "$CFG" $EXTRA --show_config | awk '$1 == "tag:" {print $2}'); [ -n "$TAG" ] || { echo "no tag from $CFG"; exit 1; }
[ "$SLURM_JOB_NAME" = "$TAG" ] || { echo "job name $SLURM_JOB_NAME is not the tag of $CFG: submit with --job-name=$TAG"; exit 1; }
# Same run queued on several GPU types (same --job-name = tag): the first copy to start cancels its queued siblings; a copy that finds an
# older sibling already running exits. Never two trainings of one run at once. (OLD_JOB: an extra job id treated the same way.)
for line in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o "%i:%T") ${OLD_JOB:+$OLD_JOB:$(squeue -h -j "$OLD_JOB" -o %T 2>/dev/null)}; do
  id=${line%%:*}; st=${line#*:}; [ "$id" = "$SLURM_JOB_ID" ] && continue
  case "$st" in
    PENDING) scancel "$id" && echo "cancelled queued sibling $id" ;;
    RUNNING|CONFIGURING) if [ "$id" -lt "$SLURM_JOB_ID" ]; then echo "older sibling $id is running -> exit"; exit 0; else scancel "$id" && echo "cancelled newer sibling $id"; fi ;;
  esac
done
LOG=logs/linkformer/$TAG.log
if [ -s "$LOG" ]; then P=$LOG.prev.$(date -u +%Y%m%dT%H%M%SZ); mv "$LOG" "$P"; echo "old log of $TAG kept as $P"; fi   # never empty a log (review #2)
# shellcheck disable=SC2086
$PY tools/linkformer.py --config "$CFG" --device cuda $EXTRA > "$LOG" 2>&1
grep -h "final\|DISK ALERT\|stop_at" "$LOG" || true
