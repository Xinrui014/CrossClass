#!/usr/bin/env python3
"""Upper bound with a perfect OCR (2026-10-17, owner question): detector boxes + the annotated text boxes and strings (old WACV labels,
e2e_boards_gt.json) + the real linker (links on those texts, p >= 0.5). Class of a linked detection: (a) prefix rule on the true string
(naming errors stay, = 'perfect strings'); (b) perfect text -> class mapping: the annotated class of the part that owns the text (same
designator), so every right link gives the right class; a wrong link gives the class of the other part. Unlinked detections keep the
detector class. Output in both formats: one class per box (detector score) and top-3 classes per box (q = (1-p) d + p onehot, score =
box score x q). Metrics on the 42 test photos (old WACV labels). Usage: oracle_text_e2e.py E2EDIR LINKS_RUN"""
import json, os, sys, subprocess, collections, re
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); RUN = sys.argv[2]
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
from kicad_links import PFX as PFX_EXT
PRE = re.compile(r'([A-Z]+)\d')
def pfx_ext(s):
    m = PRE.match(s or ''); c = PFX_EXT.get(m.group(1)) if m else None
    return c if c is not None and c <= 8 else None
os.chdir(E); import analyze9 as A; from analyze9 import iou
P9 = json.load(open("preds_wacv_1280_cls9.json")); G = json.load(open("gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]; test = [s for s in G if s not in DEV]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_gt.json"))}; L = json.load(open(LD/f"e2e_tf_links_{RUN}_gt.json"))
W = {b["id"]: b for b in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv.json"))}
owner = {st: {p["des"]: p["cls"] for p in W[st]["parts"] if p.get("des")} for st in W}
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
def build(mode, top3):
    out = {}
    for st in test:
        dpos = {k: i for i, k in enumerate(B[st]["det_index"])}; rows = []
        for k, dd in enumerate(P9[st]):
            c_txt, p = None, 0.0
            if k in dpos:
                j, p = L[st].get(str(dpos[k]), (None, 0.0))
                if j is not None and p >= 0.5:
                    s = B[st]["texts"][j]["text"]; c_txt = owner[st].get(s, pfx_class(s)) if mode == "owner" else pfx_ext(s) if mode == "ext" else pfx_class(s)
                    if c_txt is not None and not 0 <= c_txt <= 8: c_txt = None
            if not top3 or dd[6] is None:
                rows.append([*dd[:5], c_txt if c_txt is not None else int(dd[5])]); continue
            d = np.array(dd[6:15]); q = d / max(d.sum(), 1e-9)
            if c_txt is not None: q = (1 - p) * q; q[c_txt] += p
            rows += [[*dd[:4], dd[4] * q[c], int(c)] for c in np.argsort(-q)[:3] if q[c] >= 0.02]
        out[st] = nms(rows)
    return out
def metrics(name, out):
    json.dump(out, open(f"preds_{name}_1280.json", "w")); json.dump({s: G[s] for s in test}, open(f"gt9_{name}.json", "w"))
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={name}"], check=True, capture_output=True)
    dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(name)
    _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    pc = collections.defaultdict(lambda: [0, 0])
    for st in test:
        g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); dd = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(dd): continue
        I = iou(gb, dd[:, :4])
        for m, x in enumerate(g):
            jj = int(np.argmax(I[m]))
            if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(dd[jj, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); acc = sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values())
    return dm["map50"], dm["macro_f1"], np.mean([aps[n] for n in RARE if aps.get(n) is not None]), bal, acc
print(f"{'setting':58s} mAP50  macroF1   APr   bal.acc   acc")
for top3 in (False, True):
    fmt = "top-3" if top3 else "1 class"
    for mode, nm in (("prefix", "perfect OCR, prefix rule (naming errors stay)"), ("ext", "perfect OCR, extended prefix table (kicad_links)"), ("owner", "perfect OCR + perfect text->class")):
        m5, mf, apr, bal, acc = metrics(f"or_{mode}_{int(top3)}", build(mode, top3)); print(f"{nm + ', ' + fmt:58s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}")
# all found parts right (class ceiling) in both formats
for top3 in (False, True):
    out = {}
    for st in test:
        g = [x for x in G[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); rows = []
        for dd in P9[st]:
            I = iou(np.array([dd[:4]]), gb)[0] if len(gb) else np.zeros(0)
            c = g[int(I.argmax())][4] if len(I) and I.max() >= 0.5 else int(dd[5]); rows.append([*dd[:5], c])
        out[st] = nms(rows)
    m5, mf, apr, bal, acc = metrics(f"or_ceiling_{int(top3)}", out)
    print(f"{'ceiling: every found part right, 1 class':58s} {m5:.3f}  {mf:.3f}   {apr:.3f}  {100*bal:5.1f}   {100*acc:5.1f}"); break
