#!/bin/bash
# Two config-based LinkFormer runs in ONE 2-GPU cluster02 job (owner 2026-10-25 23:30: the one-change stability runs): CFG0 on GPU 0,
# CFG1 on GPU 1, each with WORKERS DataLoader workers (default 2: the job has 8 CPUs and more runs may join as job steps). More runs can
# be added later as steps: srun --jobid=$SLURM_JOB_ID --overlap --gres=gpu:pro6000:2 bash tools/cluster/run_in_alloc.sh <cfg> with
# CUDA_VISIBLE_DEVICES set by run_in_alloc's caller. The batch script ends only when its own runs AND every other step of the job are done,
# so added steps are never killed by the end of the batch script. Cluster rules: only typed --gres and --time; no --mem / --cpus-per-task.
#   sbatch --job-name=E2TF_c02_j2 --export=ALL,CFG0=configs/A.yaml,CFG1=configs/B.yaml,EXTRA="--wandb crossclass --wandb_group v3_diag" \
#          tools/cluster/linkformer_2gpu.sbatch
#SBATCH --job-name=linkformer_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --time=12:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -uo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs; K=$CROSSCLASS/.cache
export TMPDIR=$K/tmp/$SLURM_JOB_ID PIP_CACHE_DIR=$K/pip CONDA_PKGS_DIRS=$K/conda HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCH_HOME=$K/torch XDG_CACHE_HOME=$K/xdg
mkdir -p "$TMPDIR" $K/pip $K/conda $K/hf $K/triton $K/torch $K/xdg; trap 'rm -rf "$TMPDIR"' EXIT
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
: "${CFG0:?set CFG0}"; : "${CFG1:?set CFG1}"; EXTRA="--workers ${WORKERS:-2} ${EXTRA:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "code $(git rev-parse --short HEAD) linkformer.py $(md5sum < tools/linkformer.py | cut -c1-8); GPU0 $CFG0 $(md5sum < "$CFG0" | cut -c1-8); GPU1 $CFG1 $(md5sum < "$CFG1" | cut -c1-8); extra: $EXTRA"
run() {   # GPU index, config
  # shellcheck disable=SC2086
  local tag log p rc; tag=$($PY tools/linkformer.py --config "$2" $EXTRA --show_config | awk '$1 == "tag:" {print $2}'); [ -n "$tag" ] || { echo "no tag from $2"; return 1; }
  echo "$(date -u '+%F %T') UTC: GPU $1 -> $tag"; log=logs/linkformer/$tag.log
  # never empty a log (review #2; this script has no set -e, so a failed mv returns before the redirect)
  if [ -s "$log" ]; then p=$log.prev.$(date -u +%Y%m%dT%H%M%SZ); mv "$log" "$p" || { echo "GPU $1 $tag: cannot move the old log $log -> not started"; return 1; }; echo "old log of $tag kept as $p"; fi
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$1 $PY tools/linkformer.py --config "$2" --device cuda $EXTRA > "$log" 2>&1; rc=$?   # rc before the next $(date): $? would be date's
  echo "$(date -u '+%F %T') UTC: GPU $1 $tag exit code $rc"; grep -h "final\|DISK ALERT\|stop_at" "$log" || true
}
run 0 "$CFG0" & run 1 "$CFG1" & wait
while squeue -h -s -j "$SLURM_JOB_ID" -o "%i" | grep -vE "\.(batch|extern)$" | grep -q .; do sleep 60; done   # steps added with srun
echo "$(date -u '+%F %T') UTC: all runs and steps done"
