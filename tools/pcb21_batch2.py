#!/usr/bin/env python3
"""Batch 2 of the 21PCB labelling: the 29 photos without design-file correction. Owner rules (2026-10-18): check component
boxes and classes only (no text boxes, no designator strings needed); no test points, no mounting holes / fiducials.
From data/21pcb/labelme/<stem>.json (pcb21_labelme.py export): drop text boxes, drop TP* / H* / FID* / MH* components, and
remove duplicate boxes of one part (class-agnostic: of two component boxes with IoU > 0.7 keep the one with a designator,
else the higher detector score, read from prelabel.json).
  pcb21_batch2.py   -> data/21pcb/labelme_batch2/ + data/21pcb/labelme_batch2.zip"""
import json, os, re, shutil, zipfile
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; LM = D/"labelme"; OUT = D/"labelme_batch2"
MECH = re.compile(r"^(TP|H|FID|MH)\d+")

def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)

PL = json.load(open(D/"prelabel.json"))["photos"]; FIX = json.load(open(D/"design_fixed.json"))
stems = sorted(s for s in PL if s not in FIX); OUT.mkdir(exist_ok=True); tot = 0
for st in stems:
    d = json.load(open(LM/f"{st}.json")); sc = d["imageWidth"]/PL[st]["W"]
    score = {tuple(round(v*sc, 1) for v in p["box"]): p["score"] for p in PL[st]["parts"]}
    sh = []
    for s in d["shapes"]:
        desc = (s.get("description") or "").strip()
        if s["label"] == "text" or MECH.match(desc): continue
        (x1, y1), (x2, y2) = s["points"][:2]; b = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        sh.append((bool(desc), score.get(tuple(round(v, 1) for v in b), 0.0), b, s))
    sh.sort(key=lambda z: (-z[0], -z[1])); keep = []
    for z in sh:
        if all(iou(z[2], k[2]) <= 0.7 for k in keep): keep.append(z)
    d["shapes"] = [z[3] for z in keep]; tot += len(keep)
    json.dump(d, open(OUT/f"{st}.json", "w"), indent=1); shutil.copy2(LM/f"{st}.jpg", OUT/f"{st}.jpg")
    print(f"{st}: {len(keep)} components (dropped {len(sh)-len(keep)} duplicates)")
shutil.copy2(LM/"labels.txt", OUT/"labels.txt")
with zipfile.ZipFile(D/"labelme_batch2.zip", "w", zipfile.ZIP_STORED) as z:
    for f in sorted(OUT.iterdir()): z.write(f, f"pcb21_batch2/{f.name}")
print(len(stems), "photos,", tot, "components ->", D/"labelme_batch2.zip")
