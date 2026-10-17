#!/usr/bin/env python3
"""Score the transformer linker end to end (2026-10-15), the same way as result 3: a detection linked to an OCR text takes the class of
the text's prefix (assoc_lib.pfx_class, the table of result 3); unlinked detections and texts without a known prefix keep the detector
class; class-wise NMS 0.6; det_metrics.py on the 42 test images and the 3 dev images. Variants: texts des / all, and a link-probability
threshold (0 = every Hungarian link, 0.5). Writes data/preds_wacvlf_<v>{42,dev3}_1280.json and gt9 copies, prints the table.
Usage: e2e_score.py LINKS_DIR   (python: ~/venvs/gdino_fewshot on the L40; CROSSCLASS project root)"""
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
test = [s for s in G if s not in DEV]; dev = [s for s in G if s in DEV]; sets_t, sets_d = ["detector=wacv42", "result3_tree=wacvocrp05_test"], ["detector=wacvdev3"]
stats = {}
for mode in ("des", "all"):
    B = {b["id"]: b for b in json.load(open(LD/f"e2e_boards_{mode}.json"))}; Lk = json.load(open(LD/f"e2e_links_GS100k_{mode}.json"))
    for thr in (0.0, 0.5):
        out = {}; ch = linked = 0
        for st, dets in P.items():
            nb = [list(d) for d in dets]; b = B[st]
            for i, (j, p) in Lk.get(st, {}).items():
                if j is None or p < thr: continue
                linked += 1; c = pfx_class(b["texts"][j]["text"]); k = b["det_index"][int(i)]
                if c is not None and c != int(nb[k][5]): nb[k][5] = c; ch += 1
            out[st] = nms_classwise(nb)
        v = f"{mode}{int(thr*10):02d}"; stats[v] = (linked, ch)
        subset(f"wacvlf_{v}42", test, out); subset(f"wacvlf_{v}dev3", dev, out)
        sets_t.append(f"GS100k_{v}=wacvlf_{v}42"); sets_d.append(f"GS100k_{v}=wacvlf_{v}dev3")
print("links / class changes per variant:", stats)
for sets in (sets_t, sets_d):
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py")] + sets, cwd=E, check=True)
