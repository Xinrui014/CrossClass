#!/usr/bin/env bash
# Qwen3.8-27B-FP8 on ONE L40 (default GPU 0) for the class-decision test (qwen_class.py): same vLLM recipe as
# serve_qwen38_bf16.sh but FP8 weights (29 GB) and TP=1, so it fits a single GPU while GPUs 1-3 carry other users' jobs.
# --max-num-seqs 64: with one GPU the default 256 exceeds the Mamba cache blocks that fit (229) and the engine refuses to start.
set -u
BIN="$HOME/miniconda3/envs/dsv4/bin"
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
exec "$BIN/vllm" serve Qwen/Qwen3.8-27B-FP8 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 64 \
  --served-model-name qwen3.8-27b --reasoning-parser qwen3 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 9393 --limit-mm-per-prompt "{\"image\": 1}"
