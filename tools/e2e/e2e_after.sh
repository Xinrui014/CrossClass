#!/bin/bash
# End-to-end check of one intern02 checkpoint (2026-10-16): links on the WACV boards with all OCR boxes and with the rule-filtered boxes
# (infil), copied to the L40, then link accuracy (clean41 labelled parts) and mAP50 / F1 (42 test photos, PARSeq and Claude readers).
# Usage: e2e_after.sh TAG [GPU]
TAG=$1; G=${2:-6}; S=/tmp/claude-1000/-home-user-claude-backend-crossclass/a520309c-30b1-55be-9ddb-b64fff07fd55/scratchpad; RUN=$(echo $TAG | tr -d '_')
ssh intern02 "cd /dev_vepfs/xiyu/crossclass && export CROSSCLASS=\$PWD && for m in all infil; do CUDA_VISIBLE_DEVICES=$G /dev_vepfs/xiyu/miniconda3/envs/lato2/bin/python e2e/e2e_infer_tf.py ckpt/$TAG.pt e2e/e2e_boards_\$m.json e2e/e2e_tf_\$m.npz e2e/links_${TAG}_\$m.json 2>&1 | tail -1; done; sleep 3"
for m in all infil; do
  ssh intern02 "cat /dev_vepfs/xiyu/crossclass/e2e/links_${TAG}_$m.json; sleep 3" > $S/links_${TAG}_$m.json
  a=$(md5sum < $S/links_${TAG}_$m.json | cut -c1-32); b=$(ssh intern02 "md5sum /dev_vepfs/xiyu/crossclass/e2e/links_${TAG}_$m.json; sleep 3" | cut -c1-32)
  [ "$a" = "$b" ] || { echo "md5 mismatch $m"; exit 1; }
  scp -q $S/links_${TAG}_$m.json l40:/tmp/xinrui_e2e/links_${TAG}_${m}_raw.json
done
ssh l40 "cd /tmp/xinrui_e2e && ~/venvs/gdino_fewshot/bin/python -c \"
import json
B={b['id']:b for b in json.load(open('e2e_boards_infil.json'))}; L=json.load(open('links_${TAG}_infil_raw.json'))
json.dump({st:{i:[(B[st]['texts'][j]['orig'] if j is not None else None),p] for i,(j,p) in d.items()} for st,d in L.items()},open('e2e_tf_links_${RUN}infil_all.json','w'))
json.dump(json.load(open('links_${TAG}_all_raw.json')),open('e2e_tf_links_${RUN}_all.json','w'))\"
cd ~/projects/crossclass && for r in ${RUN} ${RUN}infil; do echo == \$r; ~/venvs/gdino_fewshot/bin/python /tmp/xinrui_e2e_linkacc.py /tmp/xinrui_e2e /tmp/xinrui_ce2e/linkrows_\$r.json all \$r 2>&1 | tail -4; ~/venvs/gdino_fewshot/bin/python /tmp/xinrui_e2e_score_claude.py /tmp/xinrui_e2e /tmp/xinrui_ce2e/todo.json /tmp/xinrui_ce2e/claude_reads.json \$r 2>&1 | grep -E '^(set|detector|parseq_|claude_)|class right'; done"
# top-3 classes per box (fair mAP format, 2026-10-16): detector vs prefix rule mix vs late fusion, on the rule-filtered links
ssh l40 "cd ~/projects/crossclass && nice -n 10 ~/venvs/gdino_fewshot/bin/python tools/late_fusion.py /tmp/xinrui_e2e /tmp/xinrui_ce2e/todo.json /tmp/xinrui_ce2e/claude_reads.json ${RUN}infil 2>&1 | tail -9"
