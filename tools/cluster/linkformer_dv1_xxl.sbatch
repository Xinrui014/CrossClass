#!/bin/bash
# dv1_E1_XXL_150k: model-size test. XXL transformer (768 wide x 16 layers, ~114 M), geometry tokens, CrossClass_data_v1 train, identical
# recipe to dv1_E1_XL_150k (150k steps x 4 boards, bf16, augmentation, cosine lr 3e-4), data-parallel on 4 GPUs (1 board per GPU per step).
# Zero-shot evaluation only (KiCad val, 3 dev images, all 45 WACV images; report on the 42 test images). DDP smoke test first, single-GPU fallback.
#SBATCH --job-name=dv1_E1_XXL_150k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv
RUN=dv1_E1_XXL_150k; NG=4
COMMON="--data CrossClass_data_v1 --device cuda --threads 1 --amp --size XXL --tokens E1 --dropout 0.1 --wd 0.01 --aug --cosine --accum 4 --workers 3"
DDP="$PY -m torch.distributed.run --standalone --nproc_per_node=$NG tools/linkformer.py"
if timeout 900 $DDP $COMMON --train ktrain --steps 30 --lr 3e-4 --warmup 10 --eval dev3 --eval_every 30 --tag ${RUN}_ddp_smoke > logs/linkformer/${RUN}_ddp_smoke.log 2>&1; then
  TRAIN="$DDP"; echo "DDP smoke test passed: $(grep -o 'step 30 .*' logs/linkformer/${RUN}_ddp_smoke.log)"
else
  TRAIN="$PY tools/linkformer.py"; echo "DDP smoke test FAILED -> single GPU"; tail -5 logs/linkformer/${RUN}_ddp_smoke.log
fi
rm -f results/linkformer/${RUN}_ddp_smoke.json
$TRAIN $COMMON --train ktrain --steps 150000 --lr 3e-4 --warmup 2000 --eval kvalsub,dev3 --eval_every 5000 --ckpt_every 25000 \
    --final_eval kval,dev3,all45 --save ckpt/$RUN.pt --wandb crossclass --wandb_group dv1 --tag $RUN > logs/linkformer/$RUN.log 2>&1
grep -h "final" logs/linkformer/$RUN.log
$PY tools/linkformer_report.py --run $RUN --out ${RUN}_report.md
