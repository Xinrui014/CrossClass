#!/usr/bin/env python3
"""End-to-end scores when Claude reads the linked texts. Same links as tf_I100k_all (I_100k, all OCR candidate boxes, p >= 0.5). For the
1,411 linked texts where PARSeq gives no class or a class different from the detector (ce2e/todo.json), the reading is Claude's
(ce2e/claude_reads.json, "-" = no text); the other linked texts keep the PARSeq reading. Class = pfx_class(reading), else detector class.
Class-wise NMS 0.6, det_metrics on the 42 test photos, plus class accuracy on found parts. Usage: DIR TODO READS"""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1])
RUN = sys.argv[4] if len(sys.argv) > 4 else "I100k"; todo = [tuple(t) for t in json.load(open(sys.argv[2]))]; reads = json.load(open(sys.argv[3])); assert len(todo) == len(reads)
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
os.chdir(E); from analyze9 import iou
P = json.load(open(E/"preds_wacv_1280.json")); G = json.load(open(E/"gt9_wacv.json")); DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
def nms_classwise(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
test = [s for s in G if s not in DEV]
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Lk = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json"))
RD = json.load(open(LD/"e2e_read_all.json")); CL = {k: ("" if r == "-" else r) for k, r in zip(todo, reads)}
def build(CL):
  out, n_changed = {}, 0
  for st in test:
    nb = [list(d) for d in P[st]]
    for i, (j, p) in Lk.get(st, {}).items():
        if j is None or p < 0.5: continue
        rd = CL.get((st, j), RD[st][j]); n_changed += (st, j) in CL
        c = pfx_class(rd); k = B[st]["det_index"][int(i)]
        if c is not None: nb[k][5] = c
    out[st] = nms_classwise(nb)
  return out, n_changed
names = []
for who, cl in (("parseq", {}), ("claude", CL)):
    out, n_changed = build(cl); name = f"wacv{who}_{RUN}_all42"; names.append(name)
    json.dump(out, open(E/f"preds_{name}_1280.json", "w")); json.dump({s: G[s] for s in test}, open(E/f"gt9_{name}.json", "w")); print(who, "texts read by Claude:", n_changed)
subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), "detector=wacv42", "tf_I100k_all=wacvtf_I100k_all42",
                "claude_I100k_all=wacvclaude_I100k_all42"] + [f"{n[4:]}={n}" for n in names], cwd=E, check=True)
name = names[-1]
Gt = json.load(open("gt9_wacv42.json"))
for nm in ["wacv42", "wacvtf_I100k_all42", "wacvclaude_I100k_all42"] + names:
    pr = json.load(open(f"preds_{nm}_1280.json")); f = r = 0
    for st, g in Gt.items():
        gb = np.array([b[:4] for b in g["boxes"] if b[4] >= 0]).reshape(-1, 4); gc = [b[4] for b in g["boxes"] if b[4] >= 0]
        d = np.array([q for q in pr.get(st, []) if q[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for k in range(len(gb)):
            j = int(np.argmax(I[k]))
            if I[k, j] >= 0.5: f += 1; r += int(d[j, 5]) == gc[k]
    print(f"{nm:26s} class right on found parts: {r}/{f} = {100*r/f:.1f} %")
