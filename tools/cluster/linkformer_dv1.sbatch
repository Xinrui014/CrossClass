#!/bin/bash
# dv1_E1_XL_150k: XL transformer (39.3 M), geometry tokens, pre-trained on CrossClass_data_v1 (KiCad train 23,403 sides) with data-parallel
# training on 2 GPUs (torchrun; 2 ranks x 2 boards = 4 boards per step, same schedule as planned), then six short WACV fine-tunes from its
# final checkpoint: _ft_f<k> (WACV fold k only) and _mix_f<k> (WACV fold k + KiCad 1:1), three per GPU. A 30-step DDP smoke test runs
# first; if it fails, pre-training falls back to one GPU. Settings fixed before the run; the 42 WACV test images are scored once at the end.
# OLD_JOB (the earlier single-A6000 submission of the same run) is cancelled as soon as this job starts.
#SBATCH --job-name=dv1_E1_XL_150k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
[ -n "${OLD_JOB:-}" ] && scancel "$OLD_JOB" 2>/dev/null && echo "cancelled earlier submission $OLD_JOB" || true
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv
RUN=dv1_E1_XL_150k; NG=2
COMMON="--data CrossClass_data_v1 --device cuda --threads 1 --amp --size XL --tokens E1 --dropout 0.1 --wd 0.01 --aug --cosine --accum 4 --workers 3"
DDP="$PY -m torch.distributed.run --standalone --nproc_per_node=$NG tools/linkformer.py"
# 1) smoke test of the multi-GPU path (no wandb, nothing saved)
if timeout 900 $DDP $COMMON --train ktrain --steps 30 --lr 3e-4 --warmup 10 --eval dev3 --eval_every 30 --tag ${RUN}_ddp_smoke > logs/linkformer/${RUN}_ddp_smoke.log 2>&1; then
  TRAIN="$DDP"; echo "DDP smoke test passed: $(grep -o 'step 30 .*' logs/linkformer/${RUN}_ddp_smoke.log)"
else
  TRAIN="$PY tools/linkformer.py"; echo "DDP smoke test FAILED -> single GPU"; tail -5 logs/linkformer/${RUN}_ddp_smoke.log
fi
rm -f results/linkformer/${RUN}_ddp_smoke.json
# 2) pre-training
$TRAIN $COMMON --train ktrain --steps 150000 --lr 3e-4 --warmup 2000 --eval kvalsub,dev3 --eval_every 5000 --ckpt_every 25000 \
    --final_eval kval,dev3,all45 --save ckpt/$RUN.pt --wandb crossclass --wandb_group dv1 --tag $RUN > logs/linkformer/$RUN.log 2>&1
# 3) fine-tunes from the final checkpoint, three per GPU
FT="$PY tools/linkformer.py $COMMON --workers 1 --init ckpt/$RUN.pt --steps 1000 --lr 1e-4 --warmup 100 --eval_every 250 --wandb crossclass --wandb_group dv1"
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 $FT --train fold${k}train --eval fold${k}train --final_eval fold${k}test --tag ${RUN}_ft_f$k > logs/linkformer/${RUN}_ft_f$k.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $FT --train fold${k}train --mix ktrain --mix_p 0.5 --eval fold${k}train --final_eval fold${k}test --tag ${RUN}_mix_f$k > logs/linkformer/${RUN}_mix_f$k.log 2>&1 &
done
wait
grep -h "final" logs/linkformer/${RUN}*.log
$PY tools/linkformer_report.py --run $RUN --out ${RUN}_report.md
