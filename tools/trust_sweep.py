#!/usr/bin/env python3
"""How much should the class decision trust the detector? (owner question 2026-10-19). WACV (42 test photos) and 21PCB (28 photos),
standard output: ONE class per box, score = detector box score x P(class); class-wise NMS 0.6; same metrics as the main tables.
Per detector box (score >= 0.25 boxes seen by the linker; the others keep the detector class):
  d      = detector 9-class scores, normalised
  head_d = proposal-B class head with the detector scores in the part tokens (as deployed so far)
  T      = proposal-B class head with NO detector scores (zeros, as in the 30 % dropout training): text + geometry + context only
  Ct     = soft cascade with B's links, fallback T instead of d: sum_j a_ij q_j(c) + (a_none + sum_j a_ij q_j(none)) T(c)
  X      = (T + Ct) / 2 : the detector-free class decision
  sweep  : P = (1 - w) X + w d, w = 0 .. 1 (w = share of the detector)
  pcb21 inputs: data/21pcb/e2e/{preds_cls9, e2e_boards_all, e2e_tf_all.npz, b_drop03_infil, b_drop03_infil_nodet}
  wacv inputs : data/preds_wacv_1280_cls9.json, /tmp/xinrui_e2e/{e2e_boards_all, e2e_tf_all.npz, bout_v2d2_B1_det_fp_drop03_100k, bout_drop03_nodet}
  trust_sweep.py   (env ~/venvs/gdino_fewshot, CPU) -> results/pcb21/trust_sweep.json"""
import json, os, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]; DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
SETS = {"21pcb": dict(P9=E/"21pcb/e2e/preds_cls9.json", G=E/"gt9_21pcb.json", B=E/"21pcb/e2e/e2e_boards_all.json", Z=E/"21pcb/e2e/e2e_tf_all.npz",
                      BD=E/"21pcb/e2e/b_drop03_infil.json", BT=E/"21pcb/e2e/b_drop03_infil_nodet.json"),
        "wacv": dict(P9=E/"preds_wacv_1280_cls9.json", G=E/"gt9_wacv.json", B=Path("/tmp/xinrui_e2e/e2e_boards_all.json"), Z=Path("/tmp/xinrui_e2e/e2e_tf_all.npz"),
                     BD=Path("/tmp/xinrui_e2e/bout_v2d2_B1_det_fp_drop03_100k.json"), BT=Path("/tmp/xinrui_e2e/bout_drop03_nodet.json"))}

def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]

res = {}
for sn, S in SETS.items():
    P9 = json.load(open(S["P9"])); G = json.load(open(S["G"])); Bd = {b["id"]: b for b in json.load(open(S["B"]))}; Z = np.load(S["Z"])
    BD = json.load(open(S["BD"])); BT = json.load(open(S["BT"])); imgs = sorted(s for s in G if s not in DEV); qc = {}
    def q_soft(st, j):
        if (st, j) in qc: return qc[(st, j)]
        Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
        for p, c in PFX:
            if len(p) < 8: out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
        out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); qc[(st, j)] = out / out.sum(); return qc[(st, j)]
    def C(st, r, fb):
        P = (1.0 - sum(p for _, p in r["texts"])) * fb
        for j, p in r["texts"]: q = q_soft(st, j); P = P + p * (q[:9] + q[9] * fb)
        return P
    cache = {}
    for st in imgs:                                     # per detection: (d, head_d, X) or None
        dpos = {k: i for i, k in enumerate(Bd[st]["det_index"])}; rows = []
        for k, dd in enumerate(P9[st]):
            if dd[6] is None: rows.append(None); continue
            d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); i = dpos.get(k)
            rd, rt = (BD[st].get(str(i)), BT[st].get(str(i))) if i is not None else (None, None)
            if rd is None or rt is None: rows.append((d, d, d, d)); continue
            T = np.array(rt["cls"]); X = (T + C(st, rt, T)) / 2; Hd = np.array(rd["cls"]); mixd = (Hd + C(st, rd, d)) / 2
            rows.append((d, Hd, mixd, X, T))
        cache[st] = rows
    def run(name, fn):
        out = {}
        for st in imgs:
            rows = []
            for dd, r in zip(P9[st], cache[st]):
                if r is None: rows.append(list(dd[:6])); continue
                P = fn(*r) if len(r) == 5 else r[0]; c = int(np.argmax(P)); rows.append([*dd[:4], dd[4] * P[c], c])
            out[st] = nms(rows)
        import re as _re; tag = "ts_" + sn + "_" + _re.sub(r"[^A-Za-z0-9]+", "_", name); json.dump(out, open(f"preds_{tag}_1280.json", "w")); json.dump({s: G[s] for s in imgs}, open(f"gt9_{tag}.json", "w"))
        subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={tag}"], check=True, capture_output=True)
        dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(tag)
        _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
        pc = collections.defaultdict(lambda: [0, 0])
        for st in imgs:
            g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); dd = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
            if not len(gb) or not len(dd): continue
            I = iou(gb, dd[:, :4])
            for m, x in enumerate(g):
                jj = int(np.argmax(I[m]))
                if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(dd[jj, 5]) == x[4]
        r = dict(map50=dm["map50"], map50_95=dm["map50_95"], macro_f1=dm["macro_f1"], APr=float(np.mean([aps[n] for n in RARE if aps.get(n) is not None])),
                 bal=float(np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]])), acc=sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values()))
        res[f"{sn}|{name}"] = r
        print(f"{sn:6s} {name:34s} {r['map50']:.3f}  {r['map50_95']:.3f}  {r['macro_f1']:.3f}  {r['APr']:.3f}  {100*r['bal']:5.1f}  {100*r['acc']:5.1f}", flush=True)
    print(f"{'set':6s} {'setting (1 class / box)':34s} mAP50  50-95   mF1    APr   bal.acc  acc")
    run("detector", lambda d, Hd, mixd, X, T: d)
    run("B head (with detector)", lambda d, Hd, mixd, X, T: Hd)
    run("B mix 1:1 (with detector)", lambda d, Hd, mixd, X, T: mixd)
    run("T = B head, no detector", lambda d, Hd, mixd, X, T: T)
    run("X = (T + Ct)/2, no detector", lambda d, Hd, mixd, X, T: X)
    for w in (0.1, 0.2, 0.3, 0.5, 0.7):
        run(f"(1-w) X + w d, w = {w}", lambda d, Hd, mixd, X, T, w=w: (1 - w) * X + w * d)
json.dump(res, open(ROOT/"results/pcb21/trust_sweep.json", "w"), indent=1)
