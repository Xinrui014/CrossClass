#!/usr/bin/env python3
"""End-to-end scores with PARSeq text-image features: a detection linked (p >= 0.5) to a text takes the class of the prefix PARSeq reads
on that text (assoc_lib.pfx_class); otherwise the detector class. Runs: I_100k (no string), GS_100k with PARSeq strings; texts: all OCR
boxes / pdes (PARSeq reads a designator) / gt (annotated boxes). Class-wise NMS 0.6, det_metrics, 42 test + 3 dev photos. Usage: DIR"""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1])
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
os.chdir(E); from analyze9 import iou
P = json.load(open(E/"preds_wacv_1280.json")); G = json.load(open(E/"gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
def nms_classwise(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def subset(name, imgs, preds):
    json.dump({s: preds[s] for s in imgs}, open(E/f"preds_{name}_1280.json", "w"))
    if not (E/f"gt9_{name}.json").exists(): json.dump({s: G[s] for s in imgs}, open(E/f"gt9_{name}.json", "w"))
test = [s for s in G if s not in DEV]; dev = [s for s in G if s in DEV]
sets_t = ["detector=wacv42", "result3_tree=wacvocrp05_test", "GS100k_ppocr_des05=wacvlf_des0542", "GS100k_ppocr_all05=wacvlf_all0542"]
sets_d = ["detector=wacvdev3", "GS100k_ppocr_des05=wacvlf_des05dev3", "GS100k_ppocr_all05=wacvlf_all05dev3"]
for run in ("I100k", "GS100kps"):
    for mode in ("all", "pdes", "gt"):
        f = LD/f"e2e_tf_links_{run}_{mode}.json"
        if not f.exists(): continue
        B = {b["id"]: b for b in json.load(open(LD/f"e2e_boards_{mode}.json"))}; Lk = json.load(open(f)); RD = json.load(open(LD/f"e2e_read_{mode}.json")); out = {}
        for st, dets in P.items():
            nb = [list(d) for d in dets]
            for i, (j, p) in Lk.get(st, {}).items():
                if j is None or p < 0.5: continue
                c = pfx_class(RD[st][j]); k = B[st]["det_index"][int(i)]
                if c is not None: nb[k][5] = c
            out[st] = nms_classwise(nb)
        name = f"tf_{run}_{mode}"; subset(f"wacv{name}42", test, out); subset(f"wacv{name}dev3", dev, out)
        sets_t.append(f"{name}=wacv{name}42"); sets_d.append(f"{name}=wacv{name}dev3")
for sets in (sets_t, sets_d): subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py")] + sets, cwd=E, check=True)
