#!/bin/bash
# dv1_E1_XL_150k: XL transformer (38.8 M), geometry tokens, pre-trained on CrossClass_data_v1 (KiCad train 23,403 sides), then six short
# WACV fine-tunes from its final checkpoint: _ft_f<k> (WACV fold k only) and _mix_f<k> (WACV fold k + KiCad 1:1). Settings fixed before the
# run; the 42 WACV test images are scored once at the end of each run. Needs linkdata/CrossClass_data_v1/ next to the clone.
#SBATCH --job-name=dv1_E1_XL_150k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv
RUN=dv1_E1_XL_150k
LF="$PY tools/linkformer.py --data CrossClass_data_v1 --device cuda --threads 1 --amp --size XL --tokens E1 --dropout 0.1 --wd 0.01 --aug --cosine --accum 4 --wandb crossclass --wandb_group dv1"
$LF --train ktrain --steps 150000 --lr 3e-4 --warmup 2000 --eval kvalsub,dev3 --eval_every 5000 --ckpt_every 25000 \
    --final_eval kval,dev3,all45 --save ckpt/$RUN.pt --tag $RUN > logs/linkformer/$RUN.log 2>&1
for k in 0 1 2; do
  $LF --init ckpt/$RUN.pt --train fold${k}train --steps 1000 --lr 1e-4 --warmup 100 --eval fold${k}train --eval_every 250 \
      --final_eval fold${k}test --tag ${RUN}_ft_f$k > logs/linkformer/${RUN}_ft_f$k.log 2>&1 &
  $LF --init ckpt/$RUN.pt --train fold${k}train --mix ktrain --mix_p 0.5 --steps 1000 --lr 1e-4 --warmup 100 --eval fold${k}train --eval_every 250 \
      --final_eval fold${k}test --tag ${RUN}_mix_f$k > logs/linkformer/${RUN}_mix_f$k.log 2>&1 &
done
wait
grep -h "final" logs/linkformer/${RUN}*.log
$PY tools/linkformer_report.py --run $RUN --out ${RUN}_report.md
