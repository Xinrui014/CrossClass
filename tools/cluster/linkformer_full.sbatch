#!/bin/bash
# Full transformer-linker runs (LINKING.md §10) on one GPU. Phase A in parallel: KiCad pre-training (all / 1,000 / 300 projects, E2;
# all, E1) and WACV-only training per fold (K0). Phase B: fine-tune the full pre-trained models on each WACV fold. Phase C: report on
# the 42 test images. Settings fixed before the run; test folds are only evaluated once, at the end of each run (--final_eval).
# Needs linkdata/{wacv,kicad_all,folds}.json (git-ignored) next to the clone at ~/xrssd/crossclass.
#SBATCH --job-name=linkformer_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp WANDB_DIR=/home/xinrui004/xrssd/crossclass/logs
cd "$CROSSCLASS"; mkdir -p logs/linkformer logs/wandb ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv
LF="$PY tools/linkformer.py --device cuda --threads 1 --size L --dropout 0.1 --wd 0.01 --aug --cosine --accum 4 --wandb crossclass --wandb_group linkformer_full_$SLURM_JOB_ID"
# Phase A
for spec in "full kicadtrain E2" "full kicadtrain E1" "1000 kicad1000 E2" "300 kicad300 E2"; do
  set -- $spec
  $LF --train $2 --tokens $3 --steps 20000 --lr 3e-4 --warmup 500 --eval kicadval --eval_every 2000 \
      --final_eval kicadval,all45 --save ckpt/P_$1_$3.pt --tag P_$1_$3 > logs/linkformer/P_$1_$3.log 2>&1 &
done
for k in 0 1 2; do
  $LF --train fold${k}train --tokens E2 --steps 4000 --lr 3e-4 --warmup 200 --eval fold${k}train --eval_every 500 \
      --final_eval fold${k}test --tag K0_E2_fold$k > logs/linkformer/K0_E2_fold$k.log 2>&1 &
done
wait
# Phase B
for tok in E2 E1; do for k in 0 1 2; do
  $LF --init ckpt/P_full_$tok.pt --train fold${k}train --tokens $tok --steps 1000 --lr 1e-4 --warmup 100 --eval fold${k}train --eval_every 250 \
      --final_eval fold${k}test --tag FT_full_${tok}_fold$k > logs/linkformer/FT_full_${tok}_fold$k.log 2>&1 &
done; done
wait
grep -h "final" logs/linkformer/*.log
$PY tools/linkformer_report.py
