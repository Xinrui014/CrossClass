#!/usr/bin/env python3
"""Apply reviewed corrections (data/21pcb/fix_batch2/<stem>.ops.json) to the owner's current labelme file, without overwriting
the owner's own edits. Base = data/21pcb/owner_edits/pcb21_batch2_owner/<stem>.json if the owner edited the photo, else
data/21pcb/labelme_batch2/<stem>.json. Each op is matched to a base shape by box IoU > 0.8 (labelme coordinates); it is applied
only if that shape still has the op's old_label and old_desc (the owner did not change it), else it is reported as skipped.
Actions: relabel (class only), relabel_clear_desc (class + designator cleared: wrong designator), clear_desc, delete.
  pcb21_apply_ops.py STEM [STEM ...]   -> data/21pcb/fix_batch2/<stem>.json + printed summary with base / result md5"""
import json, hashlib, os, sys
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; FX = D/"fix_batch2"

def box(s):
    (x1, y1), (x2, y2) = s["points"][:2]; return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for st in sys.argv[1:]:
    own = D/"owner_edits/pcb21_batch2_owner"/f"{st}.json"; base = own if own.exists() else D/"labelme_batch2"/f"{st}.json"
    doc = json.load(open(base)); S = doc["shapes"]; ops = json.load(open(FX/f"{st}.ops.json"))["ops"]
    done, skipped, dels = {}, [], set()
    for op in ops:
        b = op["box"]; b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
        cand = [(iou(b, box(s)), j) for j, s in enumerate(S) if s["label"] != "text"]
        v, j = max(cand) if cand else (0, None)
        if j is None or v < 0.8: skipped.append((op["action"], op.get("old_desc"), "box gone")); continue
        s = S[j]
        if s["label"] != op["old_label"] or (s.get("description") or "") != (op.get("old_desc") or ""):
            skipped.append((op["action"], op.get("old_desc"), f"owner changed it to {s['label']}/{s.get('description') or '-'}")); continue
        a = op["action"]
        if a == "delete": dels.add(j)
        if a in ("relabel", "relabel_clear_desc"): s["label"] = op["new_label"]
        if a in ("clear_desc", "relabel_clear_desc"): s["description"] = ""
        done[a] = done.get(a, 0) + 1
    doc["shapes"] = [s for j, s in enumerate(S) if j not in dels]
    out = FX/f"{st}.json"; json.dump(doc, open(out, "w"), indent=1)
    print(f"{st}: base {'owner' if base == own else 'original'} md5 {md5(base)} -> fix md5 {md5(out)}; applied {done}; skipped {len(skipped)}")
    for x in skipped: print("    skipped", x)
