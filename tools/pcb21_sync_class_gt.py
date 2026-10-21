#!/usr/bin/env python3
"""Sync the 21PCB class ground truth with the owner's link-tool edits (owner 2026-10-21: part boxes added / deleted / resized and
classes changed in the link tool are final). The component boxes of data/21pcb/final_v1/gt9_21pcb.json are replaced by the parts
of data/21pcb/link_gt_v1.json (tools/pcb21_link_merge.py; same photos, native pixels, designators kept, 'ignore' = class -1).
The text boxes of final_v1 are kept. The previous file is kept once as final_v1/gt9_21pcb_prelink.json.
Then run tools/pcb21_e2e_prep.py (scorer GT data/gt9_21pcb.json) and the result scripts.
  pcb21_sync_class_gt.py   (any python3)"""
import json, os, shutil
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
F = D/"final_v1/gt9_21pcb.json"; B = D/"final_v1/gt9_21pcb_prelink.json"
if not B.exists(): shutil.copy2(F, B)
G = json.load(open(B)); L = json.load(open(D/"link_gt_v1.json")); n0 = n1 = 0
for st, g in G.items():
    old = g["boxes"]; new = [p["box"] + [CLS.index(p["cls"]) if p["cls"] in CLS else -1, p["cls"], p.get("des", "")] for p in L[st]["parts"]]
    if len(old) != len(new): print(f"{st}: {len(old)} -> {len(new)} boxes")
    g["boxes"] = new; n0 += len(old); n1 += len(new)
from collections import Counter
c0 = Counter(b[5] for g in json.load(open(B)).values() for b in g["boxes"]); c1 = Counter(b[5] for g in G.values() for b in g["boxes"])
print("per class (before -> after):", {k: f"{c0[k]} -> {c1[k]}" for k in CLS + ["ignore"] if c0[k] != c1[k]})
json.dump(G, open(F, "w")); print(f"class GT: {n0} -> {n1} component boxes ({F}); previous file: {B}")
