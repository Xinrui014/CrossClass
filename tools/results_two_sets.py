#!/usr/bin/env python3
"""Main results table for the report (2026-10-19): WACV (42 test photos) and 21PCB (28 photos), ONE class per box, four methods:
  Detector     : v22 9-class scores (argmax), score = box score
  C            : soft cascade, soft links of v2d2_I_noise05_100k on the rule boxes x PARSeq prefix probabilities, fallback = detector
  C + LLM      : C with the blind LLM's per-board prefix -> class table instead of the fixed table (inference only)
  B-head       : class head of v2d2_B1_det_fp_drop03_100k (detector scores in the part tokens)
Score of a box = detector box score x P(chosen class); class-wise NMS 0.6. Metrics as tools/e2e/full_metrics.py.
  results_two_sets.py   (env ~/venvs/gdino_fewshot, CPU) -> results/pcb21/results_two_sets.json"""
import json, os, re, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import PFX
os.chdir(E); import analyze9 as A; from analyze9 import iou
NM9 = ["Resistor", "Capacitor", "Inductor", "Connector", "Diode", "Switch", "Transistor", "IC", "Oscillator"]
GRP = {"APf": ["Resistor", "Capacitor", "Connector"], "APc": ["IC", "Diode"], "APr": ["Switch", "Inductor", "Transistor", "Oscillator"]}
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]; T = Path("/tmp/xinrui_e2e")
SETS = {"WACV": dict(P9=E/"preds_wacv_1280_cls9.json", G=E/"gt9_wacv.json", B=T/"e2e_boards_all.json", Z=T/"e2e_tf_all.npz",
                     SOFT=T/"soft_noise05_infil.json", BO=T/"bout_v2d2_B1_det_fp_drop03_100k.json", LLM=ROOT/"tools/e2e/llm_naming/blind_decisions.json"),
        "21PCB": dict(P9=E/"21pcb/e2e/preds_cls9.json", G=E/"gt9_21pcb.json", B=E/"21pcb/e2e/e2e_boards_all.json", Z=E/"21pcb/e2e/e2e_tf_all.npz",
                      SOFT=E/"21pcb/e2e/soft_noise05_infil.json", BO=E/"21pcb/e2e/b_drop03_infil.json", LLM=E/"21pcb/e2e/llm_naming/blind_decisions.json")}

def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]

TAG = os.environ.get("SOFT_TAG", "")              # e.g. ar20k: soft links of another linker (soft_<tag>_infil.json); only the C rows
if TAG:
    SETS["WACV"]["SOFT"] = T/f"soft_{TAG}_infil.json"; SETS["21PCB"]["SOFT"] = E/f"21pcb/e2e/soft_{TAG}_infil.json"
res = {}
for sn, S in SETS.items():
    P9 = json.load(open(S["P9"])); G = json.load(open(S["G"])); Bd = {b["id"]: b for b in json.load(open(S["B"]))}; Z = np.load(S["Z"])
    SOFT = json.load(open(S["SOFT"])); BO = json.load(open(S["BO"])); LLM = json.load(open(S["LLM"])); imgs = sorted(s for s in G if s not in DEV)
    def table(st, llm):
        t = {p: c for p, c in PFX}
        if llm:
            for key, v in LLM.items():
                b, p = key.split("|", 1)
                if b == st: t[p] = NM9.index(v) if v in NM9 else None
        return t
    qc = {}
    def q_soft(st, j, llm):
        if (st, j, llm) in qc: return qc[(st, j, llm)]
        Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
        for p, c in table(st, llm).items():
            if c is not None and p.isalpha() and len(p) < 8: out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
        out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); qc[(st, j, llm)] = out / out.sum(); return qc[(st, j, llm)]
    def decide(mode, st, k, dd):
        d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); i = {kk: ii for ii, kk in enumerate(Bd[st]["det_index"])}.get(k)
        if mode == "det" or i is None: return d
        if mode == "head":
            r = BO[st].get(str(i)); return np.array(r["cls"]) if r else d
        r = SOFT[st].get(str(i))
        if not r: return d
        llm = mode == "C+LLM"; P = (1.0 - sum(p for _, p in r["texts"])) * d
        for j, p in r["texts"]: q = q_soft(st, j, llm); P = P + p * (q[:9] + q[9] * d)
        return P
    for mode, label in ((("C", "C (soft cascade)"), ("C+LLM", "C + LLM prefix table")) if TAG else (("det", "Detector"), ("C", "C (soft cascade)"), ("C+LLM", "C + LLM prefix table"), ("head", "B-head"))):
        out = {}
        for st in imgs:
            rows = []
            for k, dd in enumerate(P9[st]):
                if dd[6] is None: rows.append(list(dd[:6])); continue
                P = decide(mode, st, k, dd); c = int(np.argmax(P)); rows.append([*dd[:4], dd[4] * (P[c] if mode != "det" else 1.0) if mode != "det" else dd[4], c])
            out[st] = nms(rows)
        tag = f"rt_{sn}_{mode.replace('+', '_')}{'_' + TAG if TAG else ''}"; json.dump(out, open(f"preds_{tag}_1280.json", "w")); json.dump({s: G[s] for s in imgs}, open(f"gt9_{tag}.json", "w"))
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
        r = dict(map50=dm["map50"], map50_95=dm["map50_95"], precision=dm["precision"], recall=dm["recall"], f1=dm["f1"], macro_f1=dm["macro_f1"],
                 **{k: float(np.mean([aps[n] for n in v if aps.get(n) is not None])) for k, v in GRP.items()},
                 bal=float(np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]])), acc=sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values()))
        res[f"{sn}|{label}"] = r
        print(f"{sn:6s} {label:22s} mAP50 {r['map50']:.3f}  50-95 {r['map50_95']:.3f}  P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}  mF1 {r['macro_f1']:.3f}  "
              f"APf {r['APf']:.3f} APc {r['APc']:.3f} APr {r['APr']:.3f}  bal {100*r['bal']:.1f}  acc {100*r['acc']:.1f}", flush=True)
json.dump(res, open(ROOT/f"results/pcb21/results_two_sets{'_' + TAG if TAG else ''}.json", "w"), indent=1)
