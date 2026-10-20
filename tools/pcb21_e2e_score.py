#!/usr/bin/env python3
"""End-to-end class results on the 21PCB test set (28 photos with IC annotation), same methods and metrics as the WACV ladder:
  detector, 1 class / box and top-3 classes / box (v22 9-class scores, data/21pcb/e2e/preds_cls9.json);
  hard pipeline  = prefix rule mix (tools/late_fusion.py rule_multi): linked (p >= 0.5) text with a known prefix ->
                   q = (1 - p) d + p onehot(prefix class), else q = d; links of v2d2_I_noise05_100k on the rule boxes; reading = PARSeq
                   argmax (no Claude reads on 21PCB);
  soft cascade C = sum_j a_ij q_j(c) + (a_none + sum_j a_ij q_j(none)) d (tools/soft_cascade.py), soft links of the same linker;
  proposal B     = v2d2_B1_det_fp_drop03_100k: class head (1 class and top-3), C with B's links, (head + C) / 2 (tools/score_b.py).
Top-3 rows: each box gives up to 3 classes with score = detector box score x P(c) (P(c) >= 0.02); class-wise NMS 0.6.
Boxes the linker did not see (score < 0.25) keep the detector scores. Metrics as tools/e2e/full_metrics.py (groups by the
detector's v22 training frequency: APf R C J, APc IC D, APr SW L Q Y). No dev split: every rule and threshold comes from WACV dev3.
  pcb21_e2e_score.py   (env ~/venvs/gdino_fewshot, CPU) -> results/pcb21/e2e_results.json + printed table"""
import json, os, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = E/"21pcb/e2e"
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class, PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
P9 = json.load(open(LD/"preds_cls9.json")); G = json.load(open(E/"gt9_21pcb.json")); imgs = sorted(G)
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; BI = {b["id"]: b for b in json.load(open(LD/"e2e_boards_infil.json"))}
Z = np.load(LD/"e2e_tf_all.npz")
HARD = json.load(open(LD/"links_noise05_infil_raw.json")); SOFT = json.load(open(LD/"soft_noise05_infil.json")); BO = json.load(open(LD/"b_drop03_infil.json"))
CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
GRP = {"APf": ["Resistor", "Capacitor", "Connector"], "APc": ["IC", "Diode"], "APr": ["Switch", "Inductor", "Transistor", "Oscillator"]}

def read(st, j): return "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
# blind LLM per-board prefix -> class decisions (tools/pcb21_llm_prep.py + a blind agent; null = not a component / ignored class)
NM9 = ["Resistor", "Capacitor", "Inductor", "Connector", "Diode", "Switch", "Transistor", "IC", "Oscillator"]
LLM = json.load(open(LD/"llm_naming/blind_decisions.json")); USE_LLM = False
def table_for(st):
    """prefix -> class index (None = names no class) for photo st: the fixed table, overridden by the LLM's decisions when USE_LLM."""
    t = {p: c for p, c in PFX}
    if USE_LLM:
        for key, v in LLM.items():
            b, p = key.split("|", 1)
            if b == st: t[p] = NM9.index(v) if v in NM9 else None
    return t
def pfx_cls(st, r):
    import re as _re
    m = _re.match(r"([A-Z]+)\d", r or ""); return table_for(st).get(m.group(1)) if m else None
_QC = {}
def q_soft(st, j):
    key = (st, j, USE_LLM)
    if key in _QC: return _QC[key]
    Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
    for p, c in table_for(st).items():
        if c is not None and len(p) < 8 and p.isalpha(): out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
    out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); _QC[key] = out / out.sum(); return _QC[key]
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
TOPK = 3      # 1 = one class per box (the ultralytics / single-label output), 3 = top-3 class hypotheses per box
def top3(dd, P):
    if TOPK == 1: c = int(np.argmax(P)); return [[*dd[:4], dd[4] * P[c], c]]
    return [[*dd[:4], dd[4] * P[c], int(c)] for c in np.argsort(-P)[:3] if P[c] >= 0.02]

def build(mode):
    out = {}
    for st in imgs:
        dpos = {k: i for i, k in enumerate(B[st]["det_index"])}; rows = []
        for k, dd in enumerate(P9[st]):
            d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); i = dpos.get(k)
            if mode == "det1": rows.append(list(dd[:6])); continue
            P = d
            if mode == "hard" and i is not None and str(i) in HARD[st]:
                j, p = HARD[st][str(i)]
                if j is not None and p >= 0.5:
                    c = pfx_cls(st, read(st, BI[st]["texts"][j]["orig"]))
                    if c is not None: P = (1 - p) * d; P[c] += p
            if mode == "C" and i is not None and str(i) in SOFT[st]:
                r = SOFT[st][str(i)]; P = (1.0 - sum(p for _, p in r["texts"])) * d
                for j, p in r["texts"]: q = q_soft(st, j); P = P + p * (q[:9] + q[9] * d)
            if mode in ("head1", "head", "BC", "mix") and i is not None and str(i) in BO[st]:
                r = BO[st][str(i)]; H = np.array(r["cls"]); Cd = (1.0 - sum(p for _, p in r["texts"])) * d
                for j, p in r["texts"]: q = q_soft(st, j); Cd = Cd + p * (q[:9] + q[9] * d)
                P = H if mode in ("head1", "head") else Cd if mode == "BC" else (H + Cd) / 2
                if mode == "head1": rows.append([*dd[:5], int(P.argmax())]); continue
            elif mode == "head1": rows.append(list(dd[:6])); continue
            rows += top3(dd, P)
        out[st] = nms(rows) if mode not in ("det1",) else rows
    return out

def metrics(name, out):
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
    grp = {k: float(np.mean([aps[n] for n in v if aps.get(n) is not None])) for k, v in GRP.items()}
    gacc = {k: sum(pc[A.NM.index(n)][1] for n in v) / max(sum(pc[A.NM.index(n)][0] for n in v), 1) for k, v in GRP.items()}
    return dict(precision=dm["precision"], recall=dm["recall"], f1=dm["f1"], macro_f1=dm["macro_f1"], map50=dm["map50"], map50_95=dm["map50_95"],
                **grp, bal_acc=bal, acc=acc, acc_f=gacc["APf"], acc_c=gacc["APc"], acc_r=gacc["APr"], ap_per_class={k: aps.get(k) for k in A.NM})

RUNS = [("det1", "detector, 1 class / box"), ("det3", "detector, top-3"), ("head1", "B class head, 1 class / box"),
        ("hard", "hard pipeline (noise05 + rule boxes + prefix table), top-3"), ("C", "soft cascade C (noise05), top-3"),
        ("head", "B class head, top-3"), ("BC", "B: C with B's links, top-3"), ("mix", "B: (head + C) / 2, top-3")]
RUNS += [(k, "TOP1: " + l.replace("top-3", "1 class / box")) for k, l in RUNS if k in ("hard", "C", "BC", "mix")]
RUNS += [("hard", "+LLM: hard pipeline, LLM prefix table, top-3"), ("C", "+LLM: soft cascade C, LLM prefix table, top-3"),
         ("mix", "+LLM: B (head + C) / 2, LLM prefix table, top-3")]
res = {}
print(f"{'setting':58s} prec  recall  F1    mF1   mAP50 mAP50-95  APf   APc   APr  bal.acc  acc  acc_f  acc_c  acc_r")
for key, label in RUNS:
    USE_LLM = label.startswith("+LLM"); TOPK = 1 if label.startswith("TOP1") else 3
    key_name = key + ("_llm" if USE_LLM else "") + ("_top1" if TOPK == 1 else "")
    # detector top-3 as on WACV (late_fusion.py base_multi): the raw class scores are the box scores
    r = metrics(f"p21_{key_name}", build(key) if key != "det3" else
                {st: nms([[*dd[:4], dd[6+c], int(c)] for dd in P9[st] for c in np.argsort(-np.array(dd[6:15]))[:3] if dd[6+c] >= 0.001]) for st in imgs})
    res[label] = r
    print(f"{label:58s} {r['precision']:.3f} {r['recall']:.3f}  {r['f1']:.3f} {r['macro_f1']:.3f} {r['map50']:.3f}  {r['map50_95']:.3f}   "
          f"{r['APf']:.3f} {r['APc']:.3f} {r['APr']:.3f}  {100*r['bal_acc']:5.1f}  {100*r['acc']:5.1f} {100*r['acc_f']:5.1f} {100*r['acc_c']:5.1f} {100*r['acc_r']:5.1f}", flush=True)
json.dump(res, open(ROOT/"results/pcb21/e2e_results.json", "w"), indent=1)
