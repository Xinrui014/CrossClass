#!/bin/bash
# Generic CPU-only job on cluster02 (task 34, 2026-10-25; cluster rules github.com/NTUEEECluster/docs): NO --gres (runs on a CPU node,
# free), --cpus-per-task=N gives 3 GiB RAM per CPU, --time <= 3 days. Runs the given command (the script arguments) in the CrossClass
# env with all caches and scratch on the SSD project dir (one scratch dir per job). Use it for any heavy work: evals, md5 / du sweeps,
# data checks (the login node allows only 3 cores / 16 GB and no broad du / find / md5). Example:
#   sbatch --job-name=hc --cpus-per-task=16 --time=6:00:00 tools/cluster/cpu_job.sbatch bash logs/diag/head_context/run_head_context.sh
#SBATCH --job-name=cpu_job
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: sbatch ... cpu_job.sbatch <command> [args]"; exit 1; }
export PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python CROSSCLASS=/home/xinrui004/xrssd/crossclass; K=$CROSSCLASS/.cache
export TMPDIR=$K/tmp/$SLURM_JOB_ID PIP_CACHE_DIR=$K/pip CONDA_PKGS_DIRS=$K/conda HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCH_HOME=$K/torch XDG_CACHE_HOME=$K/xdg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$TMPDIR" $K/pip $K/conda $K/hf $K/triton $K/torch $K/xdg; trap 'rm -rf "$TMPDIR"' EXIT
cd "$CROSSCLASS"; echo "$(date -u '+%F %T') UTC: job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, code $(git rev-parse --short HEAD): $*"
"$@"
echo "$(date -u '+%F %T') UTC: job $SLURM_JOB_ID done"
