#!/usr/bin/env python3
"""Board-level consistency prior for the text -> class step (2026-10-17, owner: naming is world knowledge). Knowledge: on one board the
same designator prefix nearly always means the same part type (but boards differ: L = LED on some boards). Evidence: what the detector
thinks the OTHER parts with that prefix on the same board are. For part i whose linked text has prefix p:
  pi_i(c) ∝ tau(c | p) * prod_{k != i, prefix p, same board} d_k(c) ** (w_k / T)
tau = prefix table prior (0.8 on the table class, rest spread; unknown prefix: uniform), d_k = detector class distribution, w_k = link
probability, T = temperature (higher = trust the detector less). Then q_i = (1 - p_i) d_i + p_i pi_i, each box gives its top-3 classes
with score = box score x q. Compared with the fixed table (pi = onehot table class). Links = RUN (decoded, p >= 0.5), reading = Claude
where read else PARSeq. T is chosen on dev3; all are printed for test.
Usage: board_prior.py E2EDIR RUN TODO READS [DECISIONS.json: per board|prefix class names from an LLM; adds the mode "llm"]"""
import json, os, sys, subprocess, collections, re
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); RUN = sys.argv[2]
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[3])), json.load(open(sys.argv[4])))}
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
TAB = dict(PFX); DEC = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None; PRE = re.compile(r"^([A-Z]+)\d")
P9 = json.load(open("preds_wacv_1280_cls9.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]; test = [s for s in G if s not in DEV]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; L = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json")); RD = json.load(open(LD/"e2e_read_all.json"))
def tau(p):
    t = np.full(9, 1/9) if p not in TAB or TAB[p] > 8 else np.full(9, 0.2/8)
    if p in TAB and TAB[p] <= 8: t[TAB[p]] = 0.8
    return t
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def build(imgs, T):
    out = {}; flips = collections.Counter()
    for st in imgs:
        dpos = {k: i for i, k in enumerate(B[st]["det_index"])}; info = {}
        for k, dd in enumerate(P9[st]):                              # linked detections: prefix, link prob, detector distribution
            if k not in dpos or dd[6] is None: continue
            j, p = L[st].get(str(dpos[k]), (None, 0.0))
            if j is None or p < 0.5: continue
            m = PRE.match(CL.get((st, j), RD[st][j]) or "")
            if m: d = np.array(dd[6:15]); info[k] = (m.group(1), p, d / max(d.sum(), 1e-9))
        bypre = collections.defaultdict(list)
        for k, (pre, p, d) in info.items(): bypre[pre].append(k)
        rows = []
        for k, dd in enumerate(P9[st]):
            if dd[6] is None: rows.append(list(dd[:6])); continue
            d = np.array(dd[6:15]); q = d / max(d.sum(), 1e-9)
            if k in info:
                pre, p, _ = info[k]
                if T == "llm":                                       # an LLM's per-board decision for this prefix
                    c_ = DEC.get(f"{st}|{pre}"); pi = None
                    if c_ is not None: pi = np.zeros(9); pi[A.NM.index(c_)] = 1.0
                elif T is None:                                      # fixed table
                    if pre in TAB and TAB[pre] <= 8: pi = np.zeros(9); pi[TAB[pre]] = 1.0
                    else: pi = None
                else:
                    lg = np.log(tau(pre))
                    for kk in bypre[pre]:
                        if kk != k: lg = lg + info[kk][1] / T * np.log(np.maximum(info[kk][2], 1e-6))
                    pi = np.exp(lg - lg.max()); pi /= pi.sum()
                    if pre in TAB and TAB[pre] <= 8 and pi.argmax() != TAB[pre]: flips[(pre, A.NM[int(pi.argmax())])] += 1
                    if pre not in TAB: flips[(pre + " (not in table)", A.NM[int(pi.argmax())])] += 1
                if pi is not None: q = (1 - p) * q + p * pi
            rows += [[*dd[:4], dd[4] * q[c], int(c)] for c in np.argsort(-q)[:3] if q[c] >= 0.02]
        out[st] = nms(rows)
    return out, flips
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
    return dm["map50"], dm["macro_f1"], np.mean([aps[n] for n in RARE if aps.get(n) is not None]), bal, acc, aps
print(f"{'setting (top-3 classes per box)':34s} set    mAP50  macroF1   APr   bal.acc   acc")
for T in ((None, "llm") if DEC else (None, 1.0, 2.0, 4.0)):
    nm = "fixed prefix table" if T is None else "LLM (Claude) per-board prefix -> class" if T == "llm" else f"board prior, T = {T}"
    for sn, imgs in (("dev3", DEV), ("test42", test)):
        out, fl = build(imgs, T); m5, mf, apr, bal, acc, aps = metrics(f"bp_{T}_{sn}", imgs, out)
        print(f"{nm:34s} {sn:6s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}", flush=True)
        if sn == "test42": print("    AP50 per class:", {n[:5]: round(v, 2) for n, v in aps.items() if v is not None}, "| mapping changes (prefix -> class: parts):", dict(fl.most_common(8)))
