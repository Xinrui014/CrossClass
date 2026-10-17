#!/usr/bin/env python3
"""Late fusion of detector class scores and the linked text (2026-10-16): a small classifier per detection decides the class from
(a) the detector's 9 class scores (data/preds_wacv_1280_cls9.json) and (b) the linked text: link / none probability, prefix class of the
reading, the prefix itself (top prefixes one-hot: lets the model learn e.g. DA -> diode, or L on a board whose L parts look like diodes).
Readings: Claude where read, else PARSeq; links: I_100k on the rule-filtered OCR boxes (RUN). Trained with 5-fold board cross-validation
on the 42 WACV test photos (+ the 3 dev photos always in training): every test photo is predicted by a model that never saw its board.
Box score = detector max score x fused probability of the chosen class; class-wise NMS 0.6. Control: the same fusion with detector scores
only (shows how much is WACV re-calibration and how much is the text). Metrics on the 42 test photos (old WACV labels).
Usage: late_fusion.py E2EDIR TODO READS RUN   (python: ~/venvs/gdino_fewshot, CPU)"""
import json, os, sys, subprocess, collections
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); RUN = sys.argv[4]
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))}
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
import re
os.chdir(E); import analyze9 as A; from analyze9 import iou
P9 = json.load(open("preds_wacv_1280_cls9.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Lk = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json")); RD = json.load(open(LD/"e2e_read_all.json"))
PRE = re.compile(r"^([A-Z]+)\d")
reads = {}
for st, d in Lk.items():                                     # linked reading per detection index
    for i, (j, p) in d.items():
        k = B[st]["det_index"][int(i)]; r = CL.get((st, j), RD[st][j]) if j is not None else None
        reads[(st, k)] = (r, j is not None and p >= 0.5, p if j is not None else 0.0, p if j is None else 1.0 - p)
cnt = collections.Counter(PRE.match(r).group(1) for r, *_ in reads.values() if r and PRE.match(r)); TOP = [p for p, _ in cnt.most_common(30)]
def feats(st, k, text=True):
    s = P9[st][k]; sc = [v if v is not None else 0.0 for v in s[6:15]]; f = sc + [s[4]]
    if not text: return f
    r, linked, pl, pn = reads.get((st, k), (None, False, 0.0, 1.0)); pc = pfx_class(r) if (r and linked) else None
    m = PRE.match(r) if (r and linked) else None; pre = m.group(1) if m else None
    oh = [0.0]*10; oh[pc if pc is not None else 9] = float(linked)
    po = [float(pre == p) for p in TOP] + [float(linked and pre is not None and pre not in TOP), float(linked and pre is None)]
    return f + [float(linked), pl, pn] + oh + po
imgs = list(G); X, Y, M = {}, {}, {}
for st in imgs:
    g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); db = np.array([d[:4] for d in P9[st]]).reshape(-1, 4)
    I = iou(db, gb) if len(gb) and len(db) else np.zeros((len(db), len(gb)))
    Y[st] = np.array([g[int(I[k].argmax())][4] if I.shape[1] and I[k].max() >= 0.5 else -1 for k in range(len(db))])
test = [s for s in imgs if s not in DEV]; boards = sorted({s.split("_")[0] for s in test}); fold = {b: i % 5 for i, b in enumerate(boards)}
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def metrics(name, out):
    json.dump(out, open(f"preds_{name}_1280.json", "w")); json.dump({s: G[s] for s in test}, open(f"gt9_{name}.json", "w"))
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={name}"], check=True, capture_output=True)
    dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(name)
    _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    pc = collections.defaultdict(lambda: [0, 0])
    for st in test:
        g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); d = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for m, x in enumerate(g):
            jj = int(np.argmax(I[m]))
            if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(d[jj, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); acc = sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values())
    return dm["map50"], dm["macro_f1"], np.mean([aps[n] for n in RARE if aps.get(n) is not None]), bal, acc
def cv(text, kind, score="prob"):
    out = {}
    for f in range(5):
        tr = DEV + [s for s in test if fold[s.split("_")[0]] != f]; te = [s for s in test if fold[s.split("_")[0]] == f]
        Xtr = np.array([feats(st, k, text) for st in tr for k in range(len(P9[st])) if Y[st][k] >= 0 and P9[st][k][4] >= 0.01])
        Ytr = np.array([Y[st][k] for st in tr for k in range(len(P9[st])) if Y[st][k] >= 0 and P9[st][k][4] >= 0.01])
        clf = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4, random_state=0) if kind == "gb"
               else LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr, Ytr)
        for st in te:
            pr = clf.predict_proba(np.array([feats(st, k, text) for k in range(len(P9[st]))])); full = np.zeros((len(pr), 9)); full[:, clf.classes_] = pr
            sc = lambda k: (P9[st][k][4]*full[k].max() if score == "prob" else P9[st][k][4] if score == "keep" else
                            P9[st][k][4] if int(full[k].argmax()) == int(P9[st][k][5]) else P9[st][k][4]*full[k].max())   # mixed
            if score == "multi":                       # every box gives its top-3 classes, score = detector box score x fused class probability
                out[st] = nms([[*P9[st][k][:4], P9[st][k][4]*full[k][c], int(c)] for k in range(len(P9[st])) for c in np.argsort(-full[k])[:3] if full[k][c] >= 0.02])
            else: out[st] = nms([[*P9[st][k][:4], sc(k), int(full[k].argmax())] for k in range(len(P9[st]))])
    return out
base = {st: nms([list(d[:6]) for d in P9[st]]) for st in test}
base_multi = {st: nms([[*d[:4], d[6+c], int(c)] for d in P9[st] if d[6] is not None for c in np.argsort(-np.array(d[6:15]))[:3] if d[6+c] >= 0.001]) for st in test}
print(f"{'setting':44s} mAP50  macroF1   APr   bal.acc   acc")
def rule_multi(text=True):
    """No learning: class distribution q = (1-p) d + p onehot(text class) when the text is linked (p >= 0.5) with a known prefix, else q = d
    (d = detector scores normalised to sum 1); each box gives its top-3 classes with score = detector box score x q(c)."""
    out = {}
    for st in test:
        rows = []
        for k, dd in enumerate(P9[st]):
            if dd[6] is None: rows.append(list(dd[:6])); continue
            d = np.array(dd[6:15]); q = d / max(d.sum(), 1e-9); r, linked, pl, pn = reads.get((st, k), (None, False, 0.0, 1.0))
            c = pfx_class(r) if (text and r and linked) else None
            if c is not None: q = (1 - pl) * q; q[c] += pl
            rows += [[*dd[:4], dd[4]*q[cc], int(cc)] for cc in np.argsort(-q)[:3] if q[cc] >= 0.02]
        out[st] = nms(rows)
    return out
RUNS = [("detector", lambda: base), ("detector, top-3 classes per box", lambda: base_multi), ("detector normalised, top-3", lambda: rule_multi(False)),
        ("prefix rule mix (no learning), top-3", lambda: rule_multi(True))] + [(f"fusion {kd.upper()}: {'detector + text' if tx else 'detector only (control)'}, score {sm}", (lambda tx=tx, kd=kd, sm=sm: cv(tx, kd, sm)))
        for kd in ("lr",) for tx in (False, True) for sm in ("prob", "multi")]
for name, fn in RUNS:
    out = fn()
    m5, mf, apr, bal, acc = metrics("lf_" + re.sub(r"\W+", "_", name)[:30], out); print(f"{name:44s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}")
