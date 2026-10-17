#!/bin/bash
# One CrossClass linker training on one intern02 GPU (GPUs 0-3, owner OK 2026-10-16), same common recipe as queue_runner.sh.
# Usage: run_one.sh GPU TAG ARGS...
cd /dev_vepfs/xiyu/crossclass; G=$1; TAG=$2; shift 2; export CROSSCLASS=$PWD WANDB_MODE=offline
C="--data CrossClass_data_v2 --device cuda --threads 1 --amp --size XXL --dropout 0.1 --wd 0.01 --aug --cosine --workers 4 --train ktrain --bps 16 --size_ratio 1e9 --tok_budget 16384 --steps 37500 --lr 2e-4 --warmup 500 --eval kvalsub,clean_dev3 --eval_every 1250 --ckpt_every 0 --final_eval kval,dev3,all45,clean_test41,clean_dev3"
echo "$(date -u '+%F %T') start $TAG on GPU $G: $*" >> logs/queue_log.txt
CUDA_VISIBLE_DEVICES=$G /dev_vepfs/xiyu/miniconda3/envs/lato2/bin/python tools/linkformer.py $C "$@" --save ckpt/$TAG.pt --tag $TAG > logs/linkformer/$TAG.log 2>&1 < /dev/null
echo "$(date -u '+%F %T') done $TAG on GPU $G rc=$?" >> logs/queue_log.txt
