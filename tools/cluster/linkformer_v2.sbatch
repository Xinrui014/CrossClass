#!/bin/bash
# Linker v2 (branch linker-v2, VERSIONS.md): LightGlue-style matcher experiments on CrossClass_data_v1, same recipe as the v1 best
# (XXL, lr 1e-4, 150k steps x 4 boards, bf16, augmentation, cosine) plus EXTRA flags, e.g. A: --head lg; B: --head lg --geobias;
# C: --head lg --geobias --strnoise with TOKENS=E2. Zero-shot evaluation on KiCad val, the old WACV sets and WACV-clean v1 (clean_test41).
#SBATCH --job-name=v2_A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
# Same run queued on several GPU types (same --job-name): the first copy to start cancels its queued siblings; a copy that finds an
# older sibling already running exits. Never two trainings of one run at once. (OLD_JOB: an extra job id treated the same way.)
for line in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o "%i:%T") ${OLD_JOB:+$OLD_JOB:$(squeue -h -j "$OLD_JOB" -o %T 2>/dev/null)}; do
  id=${line%%:*}; st=${line#*:}; [ "$id" = "$SLURM_JOB_ID" ] && continue
  case "$st" in
    PENDING) scancel "$id" && echo "cancelled queued sibling $id" ;;
    RUNNING|CONFIGURING) if [ "$id" -lt "$SLURM_JOB_ID" ]; then echo "older sibling $id is running -> exit"; exit 0; else scancel "$id" && echo "cancelled newer sibling $id"; fi ;;
  esac
done
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv
RUN=${RUN:-v2_A}; LR=${LR:-1e-4}; NG=${NG:-2}; TOKENS=${TOKENS:-E1}; EXTRA=${EXTRA:-}   # GPUs; submit with --gres=gpu:pro6000:2 --cpus-per-task=8 --mem=64G --export=ALL,NG=2 for two
COMMON="--data CrossClass_data_v1 --device cuda --threads 1 --amp --size XXL --tokens $TOKENS $EXTRA --dropout 0.1 --wd 0.01 --aug --cosine --accum 4 --workers 3"
DDP="$PY -m torch.distributed.run --standalone --nproc_per_node=$NG tools/linkformer.py"
if timeout 900 $DDP $COMMON --train ktrain --steps 30 --lr 3e-4 --warmup 10 --eval dev3 --eval_every 30 --tag ${RUN}_ddp_smoke > logs/linkformer/${RUN}_ddp_smoke.log 2>&1; then
  TRAIN="$DDP"; echo "DDP smoke test passed: $(grep -o 'step 30 .*' logs/linkformer/${RUN}_ddp_smoke.log)"
else
  TRAIN="$PY tools/linkformer.py"; echo "DDP smoke test FAILED -> single GPU"; tail -5 logs/linkformer/${RUN}_ddp_smoke.log
fi
rm -f results/linkformer/${RUN}_ddp_smoke.json
$TRAIN $COMMON --train ktrain --steps 150000 --lr $LR --warmup 2000 --eval kvalsub,dev3 --eval_every 5000 --ckpt_every 25000 \
    --final_eval kval,dev3,all45,clean_test41,clean_dev3 --save ckpt/$RUN.pt --wandb crossclass --wandb_group v2 --tag $RUN > logs/linkformer/$RUN.log 2>&1
grep -h "final" logs/linkformer/$RUN.log
$PY tools/linkformer_report.py --run $RUN --out ${RUN}_report.md
