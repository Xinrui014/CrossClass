#!/usr/bin/env python3
"""Oracle-text end-to-end input (2026-10-15): parts = detector boxes (score >= 0.25, file order), texts = the annotated text boxes and
strings of WACV (linkdata/CrossClass_data_v1/wacv.json, old labels = the labels gt9 scores against), i.e. a perfect OCR. Writes
e2e_boards_gt.json in the e2e_build.py format. Usage: e2e_build_gt.py OUTDIR"""
import json, sys
from pathlib import Path
H = Path.home(); P = json.load(open(H/"projects/gdino_fewshot/err9/preds_wacv_1280.json"))
W = {b["id"]: b for b in json.load(open(H/"projects/crossclass/linkdata/CrossClass_data_v1/wacv.json"))}; OUT = Path(sys.argv[1])
boards = []
for st, dets in P.items():
    keep = [k for k, d in enumerate(dets) if d[4] >= 0.25]
    texts = [dict(box=t["box"], text=t["text"], angle=None) for t in W[st]["texts"]] if st in W else []
    boards.append(dict(id=st, det_index=keep, parts=[dict(box=dets[k][:4], des="", cls=int(dets[k][5])) for k in keep], texts=texts))
json.dump(boards, open(OUT/"e2e_boards_gt.json", "w")); print(len(boards), "boards,", sum(len(b["texts"]) for b in boards), "annotated texts")
