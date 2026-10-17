#!/usr/bin/env python3
"""Fix / break count of the class changes from the linked text (I_100k links, all OCR boxes, p >= 0.5, 42 test photos): for each detection
(conf >= 0.25) matched to a GT part (IoU >= 0.5), compare detector class and text class with the GT class. PARSeq vs Claude readings.
Usage: DIR TODO READS"""
import json, os, sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1])
todo = [tuple(t) for t in json.load(open(sys.argv[2]))]; reads = json.load(open(sys.argv[3]))
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
os.chdir(E); from analyze9 import iou
P = json.load(open("preds_wacv_1280.json")); G = json.load(open("gt9_wacv42.json"))
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Lk = json.load(open(LD/"e2e_tf_links_I100k_all.json"))
RD = json.load(open(LD/"e2e_read_all.json")); CL = {k: ("" if r == "-" else r) for k, r in zip(todo, reads)}
cnt = {"parseq": Counter(), "claude": Counter()}; cr = Counter(); brk = []
for st, g in G.items():
    gb = np.array([b[:4] for b in g["boxes"] if b[4] >= 0]).reshape(-1, 4); gc = [b[4] for b in g["boxes"] if b[4] >= 0]
    dets = P[st]; I = iou(np.array([d[:4] for d in dets]).reshape(-1, 4), gb)
    for i, (j, p) in Lk.get(st, {}).items():
        if j is None or p < 0.5: continue
        k = B[st]["det_index"][int(i)]; d = dets[k]
        if d[4] < 0.25 or not len(gb): continue
        m = int(np.argmax(I[k]));
        if I[k, m] < 0.5: continue
        t, dc = gc[m], int(d[5])
        for who, rd in (("parseq", RD[st][j]), ("claude", CL.get((st, j), RD[st][j]))):
            c = pfx_class(rd)
            if c is None or c == dc: cnt[who]["kept det, " + ("right" if dc == t else "wrong")] += 1
            elif c == t: cnt[who]["changed: fix"] += 1
            elif dc == t: cnt[who]["changed: break"] += 1; (who == "claude") and brk.append((rd, dc, t, c, (st, j) in CL))
            else: cnt[who]["changed: wrong to wrong"] += 1
        if (st, j) in CL:
            c = pfx_class(CL[(st, j)]); cr["no class" if c is None else ("= GT" if c == t else "!= GT")] += 1
for who, c in cnt.items(): print(who, dict(sorted(c.items())))
print("Claude-read texts linked to a found part, text class vs GT:", dict(cr))
NM = G[next(iter(G))].get("names") or json.load(open("gt9_wacv.json"))[next(iter(G))].get("names")
print("names", NM)
for b in brk: print(b)
