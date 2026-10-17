#!/usr/bin/env python3
"""End-to-end scores of a proposal-B linker (2026-10-17) from tools/e2e_infer_b.py output (class-head probabilities + soft links per
detector box). Modes, top-3 classes per box (score = detector box score x P(c)):
  head  : P = class head;
  C     : the soft cascade with B's own soft links: P = sum_j a_ij q_j(c) + (a_none + sum_j a_ij q_j(none)) d (q from PARSeq's character
          probabilities, d = detector class distribution);
  mix   : P = (head + C) / 2.
Plus 'head, 1 class per box' (class = argmax head, detector box score). Boxes the linker did not see (conf < 0.25) keep d.
Metrics: mAP50, macro-F1, APr (SW / L / Q / Y), balanced class accuracy and accuracy on found parts; dev3 and the 42 test photos.
Usage: score_b.py E2EDIR BOUT.json TAG"""
import json, os, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); BO = json.load(open(sys.argv[2])); TAG = sys.argv[3]
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
P9 = json.load(open("preds_wacv_1280_cls9.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]; test = [s for s in G if s not in DEV]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Z = np.load(LD/"e2e_tf_all.npz")
def q_soft(st, j):
    Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
    for p, c in PFX:
        if len(p) < 8: out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
    out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); return out / out.sum()
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def build(imgs, mode):
    out = {}
    for st in imgs:
        dpos = {k: i for i, k in enumerate(B[st]["det_index"])}; rows = []
        for k, dd in enumerate(P9[st]):
            if dd[6] is None: rows.append(list(dd[:6])); continue
            d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); P = d
            r = BO[st].get(str(dpos[k])) if k in dpos else None
            if r is not None:
                H = np.array(r["cls"]); Cd = (1.0 - sum(p for _, p in r["texts"])) * d
                for j, p in r["texts"]: q = q_soft(st, j); Cd = Cd + p * (q[:9] + q[9] * d)
                P = H if mode in ("head", "head1") else Cd if mode == "C" else (H + Cd) / 2
            if mode == "head1": rows.append([*dd[:5], int(P.argmax())]); continue
            rows += [[*dd[:4], dd[4] * P[c], int(c)] for c in np.argsort(-P)[:3] if P[c] >= 0.02]
        out[st] = nms(rows)
    return out
def metrics(name, imgs, out):
    json.dump(out, open(f"preds_{name}_1280.json", "w")); json.dump({s: G[s] for s in imgs}, open(f"gt9_{name}.json", "w"))
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={name}"], check=True, capture_output=True)
    dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(name)
    _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    pc = collections.defaultdict(lambda: [0, 0])
    for st in imgs:
        g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); dd = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(dd): continue
        I = iou(gb, dd[:, :4])
        for m, x in enumerate(g):
            jj = int(np.argmax(I[m]))
            if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(dd[jj, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); acc = sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values())
    return dm["map50"], dm["macro_f1"], np.mean([aps[n] for n in RARE if aps.get(n) is not None]), bal, acc
print(f"{TAG:34s} set    mAP50  macroF1   APr   bal.acc   acc")
for mode, nm in (("head1", "class head, 1 class per box"), ("head", "class head, top-3"), ("C", "soft cascade C with B's links, top-3"), ("mix", "(head + C) / 2, top-3")):
    for sn, imgs in (("dev3", DEV), ("test42", test)):
        m5, mf, apr, bal, acc = metrics(f"sb_{mode}_{sn}", imgs, build(imgs, mode))
        print(f"{nm:34s} {sn:6s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}", flush=True)
