# Why mAP50 is low although class accuracy on found parts is ~90 %: counts, per-class AP50, TIDE-style fixes (42 test photos).
import sys, os, json, numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/crossclass/tools")); os.chdir(os.path.expanduser("~/projects/crossclass/data")); import analyze9 as A
from analyze9 import iou
G = json.load(open("gt9_wacv42.json")); ngt = sum(sum(1 for b in g["boxes"] if b[4] >= 0) for g in G.values())
P = json.load(open("preds_wacv42_1280.json")); nd = sum(sum(1 for p in v if p[4] >= 0.25) for v in P.values())
print("GT parts", ngt, "| detections conf>=0.25", nd)
for name in ("wacv42", "wacvclaude_v2d2I100kinfil_all42"):
    data = A.load(name); base = [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data]
    m, aps = A.evaluate(data, base); print(name, "mAP50", round(m, 3), {k: round(v, 2) for k, v in aps.items() if v is not None})
data = A.load("wacv42"); print("TIDE-style, each fix alone:", A.tide(data))
cnt = {c: [0, 0] for c in A.NM}
for st, g in G.items():
    gb = np.array([b[:4] for b in g["boxes"] if b[4] >= 0]).reshape(-1, 4); gc = [b[4] for b in g["boxes"] if b[4] >= 0]
    d = np.array([p for p in P.get(st, []) if p[4] >= 0.25]).reshape(-1, 6)
    if not len(gb): continue
    I = iou(gb, d[:, :4]) if len(d) else np.zeros((len(gb), 0))
    for k in range(len(gb)):
        cnt[A.NM[gc[k]]][0] += 1
        if I.shape[1] and I[k].max() >= 0.5: cnt[A.NM[gc[k]]][1] += 1
print("per class: GT parts / found (class-agnostic):", cnt)
right = {c: 0 for c in A.NM}; P2 = json.load(open("preds_wacvclaude_v2d2I100kinfil_all42_1280.json")); right2 = {c: 0 for c in A.NM}
for st, g in G.items():
    gb = np.array([b[:4] for b in g["boxes"] if b[4] >= 0]).reshape(-1, 4); gc = [b[4] for b in g["boxes"] if b[4] >= 0]
    for PP, R_ in ((P, right), (P2, right2)):
        d = np.array([p for p in PP.get(st, []) if p[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for k in range(len(gb)):
            j = int(np.argmax(I[k]))
            if I[k, j] >= 0.5 and int(d[j, 5]) == gc[k]: R_[A.NM[gc[k]]] += 1
print("class right among found: detector", right, "| ours", right2)
