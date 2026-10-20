#!/usr/bin/env bash
# Causal (AR) link head, option A (owner 2026-10-19): frozen v2d2_I_noise05_100k backbone, train only tools/linkformer.py ARHead.
# 1) baseline: the same checkpoint with Hungarian decoding (1 step at lr 0, final eval);
# 2) AR head 20k steps (same data / augmentation as the backbone run, dropout 0), final eval kval / clean_test41 / clean_dev3;
# 3) soft links of the AR model on the 21PCB and WACV end-to-end boards (rule boxes) for the soft cascade C.
# Usage (L40): GPU=2 setsid nohup bash tools/ar_night.sh > /dev/null 2>&1 < /dev/null &      log: logs/ar_night.log
set -u
cd ~/projects/crossclass; export CROSSCLASS=$PWD; export CUDA_VISIBLE_DEVICES=${GPU:-2}; LOG=logs/ar_night.log; PY=~/venvs/gdino_fewshot/bin/python
COMMON="--train ktrain --size XXL --tokens E1 --cosine --dropout 0 --wd 0.01 --aug --data CrossClass_data_v2 --amp --workers 4 --head dot --geobias
 --noise 0.5 --txtfeat linkdata/kicad_textfeat --txtfeat_wacv linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz --bps 16 --size_ratio 1e9
 --tok_budget 16384 --fake 0 --init ckpt/intern02/v2d2_I_noise05_100k.pt --device cuda --final_eval kval,clean_test41,clean_dev3"
echo "$(date '+%F %T') baseline" >> $LOG
$PY tools/linkformer.py $COMMON --steps 1 --lr 0 --warmup 1 --eval "" --tag ar_baseline_noise05 >> $LOG 2>&1
echo "$(date '+%F %T') AR head training" >> $LOG
$PY tools/linkformer.py $COMMON --steps 20000 --lr 3e-4 --warmup 500 --arhead --freeze_backbone --eval kvalsub,clean_dev3 --eval_every 2500 \
  --save ckpt/v2d2_I_noise05_ar20k.pt --tag v2d2_I_noise05_ar20k >> $LOG 2>&1
echo "$(date '+%F %T') soft links" >> $LOG
$PY tools/e2e_infer_soft.py ckpt/v2d2_I_noise05_ar20k.pt data/21pcb/e2e/e2e_boards_infil.json data/21pcb/e2e/e2e_tf_infil.npz data/21pcb/e2e/soft_ar20k_infil.json >> $LOG 2>&1
$PY tools/e2e_infer_soft.py ckpt/v2d2_I_noise05_ar20k.pt /tmp/xinrui_e2e/e2e_boards_infil.json /tmp/xinrui_e2e/e2e_tf_infil.npz /tmp/xinrui_e2e/soft_ar20k_infil.json >> $LOG 2>&1
echo "$(date '+%F %T') AR_NIGHT_DONE" >> $LOG
