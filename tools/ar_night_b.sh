#!/usr/bin/env bash
# Option B of the causal (AR) link head (owner 2026-10-19 night): backbone + AR head fine-tuned together from v2d2_I_noise05_100k
# (link loss + AR loss, lr 1e-4, dropout 0.1 as the backbone run), 20k steps. Runs after tools/ar_night.sh (option A) has finished
# on the same GPU, then soft links on 21PCB / WACV and the C rows of both tables for A (ar20k) and B (arjoint20k).
# Usage (L40): GPU=2 setsid nohup bash tools/ar_night_b.sh > /dev/null 2>&1 < /dev/null &      log: logs/ar_night_b.log
set -u
cd ~/projects/crossclass; export CROSSCLASS=$PWD; export CUDA_VISIBLE_DEVICES=${GPU:-2}; LOG=logs/ar_night_b.log; PY=~/venvs/gdino_fewshot/bin/python
until grep -q "AR_NIGHT_DONE" logs/ar_night.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') option A finished; scoring A" >> $LOG
SOFT_TAG=ar20k $PY tools/results_two_sets.py >> $LOG 2>&1
echo "$(date '+%F %T') option B training" >> $LOG
$PY tools/linkformer.py --train ktrain --size XXL --tokens E1 --cosine --dropout 0.1 --wd 0.01 --aug --data CrossClass_data_v2 --amp --workers 4 \
  --head dot --geobias --noise 0.5 --txtfeat linkdata/kicad_textfeat --txtfeat_wacv linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz --bps 16 \
  --size_ratio 1e9 --tok_budget 16384 --fake 0 --init ckpt/intern02/v2d2_I_noise05_100k.pt --device cuda --arhead \
  --steps 20000 --lr 1e-4 --warmup 500 --eval kvalsub,clean_dev3 --eval_every 2500 --final_eval kval,clean_test41,clean_dev3 \
  --save ckpt/v2d2_I_noise05_arjoint20k.pt --tag v2d2_I_noise05_arjoint20k >> $LOG 2>&1
echo "$(date '+%F %T') soft links B" >> $LOG
$PY tools/e2e_infer_soft.py ckpt/v2d2_I_noise05_arjoint20k.pt data/21pcb/e2e/e2e_boards_infil.json data/21pcb/e2e/e2e_tf_infil.npz data/21pcb/e2e/soft_arjoint20k_infil.json >> $LOG 2>&1
$PY tools/e2e_infer_soft.py ckpt/v2d2_I_noise05_arjoint20k.pt /tmp/xinrui_e2e/e2e_boards_infil.json /tmp/xinrui_e2e/e2e_tf_infil.npz /tmp/xinrui_e2e/soft_arjoint20k_infil.json >> $LOG 2>&1
SOFT_TAG=arjoint20k $PY tools/results_two_sets.py >> $LOG 2>&1
echo "$(date '+%F %T') AR_NIGHT_B_DONE" >> $LOG
