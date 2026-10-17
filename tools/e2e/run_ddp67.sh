#!/bin/bash
# One CrossClass linker training on intern02 GPUs 6+7 with DDP (owner 2026-10-16: "use 6, 7"). Marks both GPUs busy so gpu_guard.sh lets its
# holders exit and starts none; frees them at the end. Total 16 boards per step (8 per GPU), same recipe as the queue runs.
# Usage: run_ddp67.sh TAG ARGS...
cd /dev_vepfs/xiyu/crossclass; TAG=$1; shift; export CROSSCLASS=$PWD WANDB_MODE=offline
C="--data CrossClass_data_v2 --device cuda --threads 1 --amp --size XXL --dropout 0.1 --wd 0.01 --aug --cosine --workers 4 --train ktrain --bps 16 --size_ratio 1e9 --tok_budget 16384 --steps 37500 --lr 2e-4 --warmup 500 --eval kvalsub,clean_dev3 --eval_every 1250 --ckpt_every 0 --final_eval kval,dev3,all45,clean_test41,clean_dev3"
touch .busy_gpu6 .busy_gpu7 .release_gpu6 .release_gpu7
for k in $(seq 1 30); do pgrep -f "release_gpu[67]" > /dev/null || break; sleep 5; done
echo "$(date -u '+%F %T') start $TAG on GPUs 6+7 (DDP): $*" >> logs/queue_log.txt
CUDA_VISIBLE_DEVICES=6,7 /dev_vepfs/xiyu/miniconda3/envs/lato2/bin/torchrun --nproc_per_node 2 --master_port 29567 tools/linkformer.py $C "$@" \
  --save ckpt/$TAG.pt --tag $TAG > logs/linkformer/$TAG.log 2>&1 < /dev/null; rc=$?
echo "$(date -u '+%F %T') done $TAG on GPUs 6+7 rc=$rc" >> logs/queue_log.txt; rm -f .busy_gpu6 .busy_gpu7
