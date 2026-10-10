#!/usr/bin/env python3
"""Standard detection metrics for err9-style sets: precision / recall / F1 at conf 0.25 (class-aware, IoU 0.5; detections on
non-component labels are ignored, as in all err9 numbers), macro-F1 over the 9 classes, mAP50 and mAP50-95 (class-averaged).
Usage (in err9/): det_metrics.py name=set [name=set ...]"""
import sys, json, collections
import numpy as np
import analyze9 as A
NM = A.NM
rows = []
for arg in sys.argv[1:]:
    name, s = arg.split("=", 1); data = A.load(s)
    tp = fp = 0; ngt = 0; pc = collections.defaultdict(lambda: [0, 0, 0])   # tp, fp, gt per class
    for d in data:
        keep = d["ps"] >= A.THR; tpi, gm, I = A.match(d, keep); t, tgt, _ = A.pred_types(d, keep, tpi, gm, I)
        ngt += len(d["gc"])
        for c in d["gc"]: pc[int(c)][2] += 1
        for k in np.where(keep)[0]:
            if t[k] == "on_ignore": continue
            if t[k] == "correct": tp += 1; pc[int(d["pc"][k])][0] += 1
            else: fp += 1; pc[int(d["pc"][k])][1] += 1
    P = tp/max(tp+fp, 1); R = tp/max(ngt, 1); F = 2*P*R/max(P+R, 1e-9)
    f1s = []
    for c in range(9):
        a, b, g = pc[c]
        if g == 0: continue
        p = a/max(a+b, 1); r = a/g; f1s.append(2*p*r/max(p+r, 1e-9))
    base = [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data]
    maps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        A.FG = float(thr); maps.append(A.evaluate(data, base)[0])
    A.FG = 0.5
    rows.append((name, P, R, F, float(np.mean(f1s)), maps[0], float(np.mean(maps))))
print(f"{'set':42s} precision recall   F1    macroF1  mAP50  mAP50-95")
for n, P, R, F, mf, m5, m595 in rows: print(f"{n:42s} {P:8.3f} {R:6.3f} {F:6.3f} {mf:7.3f} {m5:6.3f} {m595:7.3f}")
json.dump([dict(set=n, precision=P, recall=R, f1=F, macro_f1=mf, map50=m5, map50_95=m595) for n, P, R, F, mf, m5, m595 in rows], open("det_metrics_q38.json", "w"), indent=1)
