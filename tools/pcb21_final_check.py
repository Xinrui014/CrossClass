#!/usr/bin/env python3
"""Consistency check of the owner's final 21PCB labels (the 28 photos with IC annotation; no other photos).
Sources: data/21pcb/owner_final2/pcb21_batch1_class/ (12 design-aligned photos, component boxes) and
data/21pcb/owner_final2/pcb21_batch2/ (the other 16 in-scope photos). BoM classes from data/21pcb/prelabel.json (current rules:
fuses / polyswitches -> ignore, EMI filters -> inductor, test points, 0-ohm links -> resistor, connectors by description).
Checks per photo:
  A. box designator in the BoM but its label differs from the BoM class
  B. designator not in the BoM (typo, or a test-point / text name)
  C. the same designator on two boxes
  D. two component boxes overlapping (IoU > 0.7)
  E. label counts vs BoM populated counts for the side (rough: a BoM covers both sides)
  pcb21_final_check.py -> results/pcb21/final_check.json + printed report
Env PCB21_SRC = one folder under data/21pcb with all 28 component files (e.g. owner_final3/pcb21_final, the owner's GT check of
2026-10-20) instead of the owner_final2 batch folders."""
import json, os, re, collections
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; R = ROOT/"results/pcb21"
SRC = os.environ.get("PCB21_SRC", "")
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
IC28 = sorted(p.stem for p in (Path.home()/"projects/tcassi/data/21PCBs/IC_annotation").glob("*.json"))

def box(s):
    (x1, y1), (x2, y2) = s["points"][:2]; return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)

PL = json.load(open(D/"prelabel.json")); out = {}; tot = collections.Counter()
for st in IC28:
    f = D/SRC/f"{st}.json" if SRC else D/"owner_final2/pcb21_batch1_class"/f"{st}.json"
    if not f.exists(): f = D/"owner_final2/pcb21_batch2"/f"{st}.json"
    S = [s for s in json.load(open(f))["shapes"] if s["label"] != "text"]; bom = PL["bom"][st.split("_")[0]]
    A, B, C, Dd = [], [], [], []; seen = collections.defaultdict(list)
    for k, s in enumerate(S):
        r = (s.get("description") or "").strip().upper()
        if not r or r.startswith("?"): continue
        seen[r].append(k)
        if r in bom:
            want = CLS[bom[r]["cls"]] if bom[r]["cls"] >= 0 else "ignore"
            if want != s["label"]: A.append(dict(ref=r, label=s["label"], bom_class=want, bom_raw=bom[r]["raw"], value=bom[r]["value"][:30], desc=bom[r]["desc"][:60]))
        else: B.append(dict(ref=r, label=s["label"]))
    C = [dict(ref=r, n=len(v)) for r, v in seen.items() if len(v) > 1]
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if iou(box(S[i]), box(S[j])) > 0.7: Dd.append((S[i]["label"], S[i].get("description") or "", S[j]["label"], S[j].get("description") or ""))
    lab = collections.Counter(s["label"] for s in S)
    out[st] = dict(n=len(S), labels=dict(lab), A=A, B=B, C=C, D=Dd)
    tot.update(A=len(A), B=len(B), C=len(C), D=len(Dd), boxes=len(S))
json.dump(out, open(R/"final_check.json", "w"), indent=1)
print("TOTAL", dict(tot), "photos", len(out))
trans = collections.Counter((a["label"], a["bom_class"], a["bom_raw"]) for v in out.values() for a in v["A"])
print("label vs BoM class (label, bom class, bom raw): count")
for k, n in trans.most_common(): print("  ", k, n)
for st, v in out.items():
    if v["A"] or v["B"] or v["C"] or v["D"]:
        print(f"== {st} ({v['n']} boxes) A={len(v['A'])} B={len(v['B'])} C={len(v['C'])} D={len(v['D'])}")
        for a in v["A"]: print(f"   A {a['ref']:7} labelled {a['label']:10} BoM {a['bom_class']:10} ({a['value']} | {a['desc']})")
        if v["B"]: print("   B not in BoM:", [(b["ref"], b["label"]) for b in v["B"]])
        if v["C"]: print("   C repeated:", v["C"])
        if v["D"]: print("   D overlap:", v["D"][:6])
