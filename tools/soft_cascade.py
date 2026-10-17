#!/usr/bin/env python3
"""Soft cascade (proposal C, 2026-10-17): replace the hard decisions of the end-to-end pipeline by probabilities, no training.
  P(c | box i) = sum_j a_ij q_j(c) + (a_i,none + sum_j a_ij q_j(none)) d_i(c)
a_ij = the linker's softmax over the board's OCR text boxes + none (tools/e2e_infer_soft.py); q_j(c) = probability that text j names
class c, from PARSeq's per-position character probabilities (the 8 x 38 part of the text feature): P(prefix p) = prod_k q[k, p_k] x
P(digit at len(p)), summed over the prefixes of class c (assoc_lib.PFX); q_j(none) = 1 - sum; d_i = the detector's 9 class scores,
normalised. Each box gives its top-3 classes, score = detector box score x P(c); class-wise NMS 0.6. Boxes the linker did not see
(conf < 0.25) keep d_i. Variants switch one part back to the hard decision: hard link (decoded link, p >= 0.5) and / or hard reading
(the prefix of PARSeq's argmax reading, or Claude's reading where Claude read the text). Metrics on dev3 and the 42 test photos.
Usage: soft_cascade.py E2EDIR SOFTLINKS HARDLINKS_RUN TODO READS"""
import json, os, sys, subprocess, collections, re
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); SOFT = json.load(open(LD/sys.argv[2])); RUN = sys.argv[3]
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[4])), json.load(open(sys.argv[5])))}
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class, PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
P9 = json.load(open("preds_wacv_1280_cls9.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]; test = [s for s in G if s not in DEV]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Z = np.load(LD/"e2e_tf_all.npz"); RD = json.load(open(LD/"e2e_read_all.json"))
HARD = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json"))
CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
def q_soft(st, j):
    """(10,) = P(class 0..8 named by text j), P(no designator), from PARSeq's character probabilities."""
    Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
    for p, c in PFX:
        if len(p) >= 8: continue
        out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
    out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); return out / out.sum()
def q_hard(st, j, claude):
    r = CL.get((st, j), RD[st][j]) if claude else "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
    c = pfx_class(r); out = np.zeros(10); out[c if c is not None else 9] = 1.0; return out
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def build(imgs, soft_link=True, reading="soft"):
    out = {}
    for st in imgs:
        dpos = {k: i for i, k in enumerate(B[st]["det_index"])}; rows = []
        for k, dd in enumerate(P9[st]):
            if dd[6] is None: rows.append(list(dd[:6])); continue
            d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); P = d.copy()
            if k in dpos:
                i = str(dpos[k])
                if soft_link:
                    s = SOFT[st].get(i); links = [(j, p) for j, p in s["texts"]] if s else []
                else:
                    j, p = HARD[st].get(i, (None, 0.0)); links = [(j, p)] if (j is not None and p >= 0.5) else []
                rest = 1.0 - sum(p for _, p in links); P = rest * d
                for j, p in links:
                    q = q_soft(st, j) if reading == "soft" else q_hard(st, j, reading == "claude")
                    P = P + p * (q[:9] + q[9] * d)
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
print(f"{'setting (top-3 classes per box)':46s} set    mAP50  macroF1   APr   bal.acc   acc")
for name, kw in [("hard link + hard reading (PARSeq)", dict(soft_link=False, reading="parseq")),
                 ("hard link + hard reading (Claude)", dict(soft_link=False, reading="claude")),
                 ("hard link + soft reading", dict(soft_link=False, reading="soft")),
                 ("soft link + hard reading (PARSeq)", dict(soft_link=True, reading="parseq")),
                 ("soft link + soft reading = C", dict(soft_link=True, reading="soft"))]:
    for sn, imgs in (("dev3", DEV), ("test42", test)):
        m5, mf, apr, bal, acc = metrics("sc_" + re.sub(r"\W+", "_", name)[:24] + sn, imgs, build(imgs, **kw))
        print(f"{name:46s} {sn:6s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}", flush=True)
