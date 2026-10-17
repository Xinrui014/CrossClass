#!/usr/bin/env python3
"""Rule test input: drop every OCR candidate box that lies >= 50 % inside a detector box (the KiCad data build drops labels >= 50 %
inside a footprint body, so the linker never saw text boxes on parts). Keeps each text's original index in "orig". Writes
e2e_boards_infil.json and e2e_tf_infil.npz next to the inputs. Usage: DIR"""
import json, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); B = json.load(open(D/"e2e_boards_all.json")); Z = np.load(D/"e2e_tf_all.npz"); out, feats, n0, n1 = [], {}, 0, 0
def frac_in(a, b):
    iw = max(0, min(a[2], b[2])-max(a[0], b[0])); ih = max(0, min(a[3], b[3])-max(a[1], b[1])); return iw*ih/max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
for b in B:
    keep = [j for j, t in enumerate(b["texts"]) if not any(frac_in(t["box"], p["box"]) >= 0.5 for p in b["parts"])]
    n0 += len(b["texts"]); n1 += len(keep)
    out.append(dict(b, texts=[dict(b["texts"][j], orig=j) for j in keep]))
    if b["id"] in Z.files: feats[b["id"]] = Z[b["id"]][keep]
json.dump(out, open(D/"e2e_boards_infil.json", "w")); np.savez(D/"e2e_tf_infil.npz", **feats); print(f"texts {n0} -> {n1}")
