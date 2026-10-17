#!/usr/bin/env python3
"""Which box score to use when the linked text decides the class (2026-10-16). The detector's score is its probability for ITS class;
after the text changes the class that number no longer fits. Modes: keep = detector score (all earlier runs); mult = detector score x
link probability when the text changes the class; mult_or = mult, and when the text agrees with the detector 1-(1-s)(1-p) (both say so).
Pipeline: I_100k links on the rule-filtered OCR boxes (RUN), reading = Claude where read else PARSeq, prefix class, NMS 0.6.
Metrics: mAP50, macro-F1, APr (SW / L / Q / Y), balanced class accuracy on found parts; dev3 (3 photos, for choosing) and the 42 test
photos (old WACV labels). Usage: E2EDIR TODO READS RUN"""
import json, os, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); RUN = sys.argv[4]
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))}
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
os.chdir(E); import analyze9 as A; from analyze9 import iou
P = json.load(open("preds_wacv_1280.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]; RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Lk = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json")); RD = json.load(open(LD/"e2e_read_all.json"))
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def build(mode, imgs):
    out = {}
    for st in imgs:
        nb = [list(d) for d in P[st]]
        for i, (j, p) in Lk.get(st, {}).items():
            if j is None or p < 0.5: continue
            c = pfx_class(CL.get((st, j), RD[st][j])); k = B[st]["det_index"][int(i)]
            if c is None: continue
            s, dc = nb[k][4], int(nb[k][5])
            if c != dc: nb[k][5] = c; nb[k][4] = s if mode == "keep" else s * p
            elif mode == "mult_or": nb[k][4] = 1 - (1 - s) * (1 - p)
        out[st] = nms(nb)
    return out
def metrics(name, imgs, out):
    json.dump(out, open(f"preds_{name}_1280.json", "w")); json.dump({s: G[s] for s in imgs}, open(f"gt9_{name}.json", "w"))
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={name}"], check=True, capture_output=True)
    dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(name)
    _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    pc = collections.defaultdict(lambda: [0, 0])
    for st in imgs:
        g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); d = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for m, x in enumerate(g):
            jj = int(np.argmax(I[m]))
            if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(d[jj, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); apr = np.mean([aps[n] for n in RARE if aps.get(n) is not None])
    return dm["map50"], dm["macro_f1"], apr, bal
test = [s for s in G if s not in DEV]
print(f"{'mode':10s} {'set':6s} mAP50  macroF1  APr   bal.acc")
for mode in ("keep", "mult", "mult_or"):
    for sn, imgs in (("dev3", DEV), ("test42", test)):
        m5, mf, apr, bal = metrics(f"conf_{mode}_{sn}", imgs, build(mode, imgs)); print(f"{mode:10s} {sn:6s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}")
