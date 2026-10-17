#!/usr/bin/env python3
"""What the OCR candidate boxes are (clean41 test photos): annotated label (covers / lies in >= 50 % of an annotated text box), inside a
detector box (>= 50 % of the OCR box), or elsewhere; and the same split for the 459 wrong links to non-designator boxes. Usage: DIR ROWS"""
import json, os, re, sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); LD = Path(sys.argv[1]); rows = json.load(open(sys.argv[2]))
DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}; DES = re.compile(r"^[A-Z]+\d+[A-Z]?$")
W = {w["id"]: w for w in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json"))}
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}
def frac_in(a, b):  # fraction of box a inside box b
    iw = max(0, min(a[2], b[2])-max(a[0], b[0])); ih = max(0, min(a[3], b[3])-max(a[1], b[1])); return iw*ih/max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
def kind(o, st):
    w, b = W[st], B[st]
    if any(frac_in(t["box"], o) >= 0.5 or frac_in(o, t["box"]) >= 0.5 for t in w["texts"]): return "annotated text"
    if any(frac_in(o, p["box"]) >= 0.5 for p in b["parts"]): return "inside a detector box"
    return "elsewhere (pads, vias, other print)"
c = Counter()
for st in W:
    if st in DEV or st not in B: continue
    for t in B[st]["texts"]: c[kind(t["box"], st)] += 1
print("all OCR boxes:", sum(c.values()), dict(c))
junk = [r for r in rows if r["stage"] == "linked to a wrong text" and not DES.match(r["ltext"] or "")]
cj = Counter(); own = 0
for r in junk:
    cj[kind(r["lbox"], r["img"])] += 1; own += frac_in(r["lbox"], r["dbox"]) >= 0.5
print("459 junk links:", dict(cj), "| inside the part's own detector box:", own)
