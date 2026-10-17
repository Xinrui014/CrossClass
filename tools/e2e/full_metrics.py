# Full metric table (2026-10-17) for saved prediction sets on the 42 WACV test photos: precision / recall / F1 (conf 0.25, class-aware),
# macro-F1, mAP50, mAP50-95, APf (R, C) / APc (J, IC, D) / APr (SW, L, Q, Y), balanced class accuracy and accuracy on found parts.
import json, os, sys, subprocess, collections, numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/crossclass/tools")); os.chdir(os.path.expanduser("~/projects/crossclass/data")); import analyze9 as A
from analyze9 import iou
# groups by the detector's training-set frequency (v22): frequent > 4000, common 1000-2200, rare < 500 boxes
GRP = {"APf": ["Resistor", "Capacitor", "Connector"], "APc": ["IC", "Diode"], "APr": ["Switch", "Inductor", "Transistor", "Oscillator"]}
rows = []
for arg in sys.argv[1:]:
    label, name = arg.split("=", 1)
    subprocess.run([sys.executable, "../tools/det_metrics.py", f"x={name}"], check=True, capture_output=True); dm = json.load(open("det_metrics_q38.json"))[0]
    data = A.load(name); _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    G = json.load(open(f"gt9_{name}.json")); P = json.load(open(f"preds_{name}_1280.json")); pc = collections.defaultdict(lambda: [0, 0])
    for st, g in G.items():
        gg = [x for x in g["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in gg]).reshape(-1, 4); dd = np.array([x for x in P.get(st, []) if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(dd): continue
        I = iou(gb, dd[:, :4])
        for m, x in enumerate(gg):
            j = int(np.argmax(I[m]))
            if I[m, j] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(dd[j, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); acc = sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values())
    ga = {k: sum(pc[A.NM.index(n)][1] for n in v) / max(sum(pc[A.NM.index(n)][0] for n in v), 1) for k, v in GRP.items()}   # accuracy within each group
    rows.append((label, dm["precision"], dm["recall"], dm["f1"], dm["macro_f1"], dm["map50"], dm["map50_95"], *[np.mean([aps[n] for n in GRP[k]]) for k in ("APf", "APc", "APr")], bal, acc, ga['APf'], ga['APc'], ga['APr']))
print(f"{'setting':46s} prec  recall  F1    mF1   mAP50 mAP50-95  APf   APc   APr  bal.acc  acc  acc_f  acc_c  acc_r")
for r in rows: print(f"{r[0]:46s} {r[1]:.3f} {r[2]:.3f}  {r[3]:.3f} {r[4]:.3f} {r[5]:.3f}  {r[6]:.3f}   {r[7]:.3f} {r[8]:.3f} {r[9]:.3f}  {100*r[10]:5.1f}  {100*r[11]:5.1f} {100*r[12]:5.1f} {100*r[13]:5.1f} {100*r[14]:5.1f}")
