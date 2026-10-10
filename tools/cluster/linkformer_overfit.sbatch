#!/bin/bash
# Overfit test of the whole-board transformer linker (LINKING.md §10) on one GPU: {3 dev boards, 30 fold-0 boards} x {L, XL} x {E1, E2}.
# Four runs share the GPU at a time (each < 4 GB). Repo: git clone git@github.com:Xinrui014/CrossClass.git ~/xrssd/crossclass;
# linkdata/{wacv,kicad199,folds}.json (git-ignored, WACV-derived) are copied in separately.
#SBATCH --job-name=linkformer_overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=rose
#SBATCH --output=/home/xinrui004/xrssd/crossclass/logs/%x_%j.out
set -euo pipefail
source /etc/profile.d/z00-lmod.sh
module load CUDA/12.8.0 || true
PY=/projects/_ssd/xrssd/envs/qwen_edit_flash/bin/python
export CROSSCLASS=/home/xinrui004/xrssd/crossclass TMPDIR=/home/xinrui004/xrssd/.tmp
cd "$CROSSCLASS"; mkdir -p logs/linkformer
nvidia-smi --query-gpu=name,memory.total --format=csv
for sz in L XL; do
  for tk in E1 E2; do
    $PY tools/linkformer.py --device cuda --threads 1 --train dev3 --eval dev3 --size $sz --tokens $tk --steps 1500 --eval_every 250 \
        --tag O1_dev3_${sz}_${tk} > logs/linkformer/O1_dev3_${sz}_${tk}.log 2>&1 &
    $PY tools/linkformer.py --device cuda --threads 1 --train fold0train --eval fold0train,fold0test --size $sz --tokens $tk --steps 4000 \
        --eval_every 500 --tag O2_fold0_${sz}_${tk} > logs/linkformer/O2_fold0_${sz}_${tk}.log 2>&1 &
  done
  wait
done
grep -h "step" logs/linkformer/O*.log | tail -40
