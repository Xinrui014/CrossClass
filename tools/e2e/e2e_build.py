#!/usr/bin/env python3
"""End-to-end input for the transformer linker (2026-10-15): per WACV photo, parts = detector boxes (preds_wacv_1280.json, score >= 0.25,
in file order) and texts = PP-OCRv6 candidates (assoc/ocr_wacvcand.json, the same texts the tree linker of result 3 saw). mode "des": only
designator-like strings (^[A-Z]+\\d+[A-Z]?$, i.e. those that can give a class); mode "all": every candidate. Writes e2e_boards_<mode>.json.
Usage: e2e_build.py OUTDIR   (python: any python3 on the L40)"""
import json, re, sys
from pathlib import Path
H = Path.home(); E = H/"projects/gdino_fewshot/err9"; A = H/"projects/gdino_fewshot/assoc"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
P = json.load(open(E/"preds_wacv_1280.json")); O = json.load(open(A/"ocr_wacvcand.json")); DES = re.compile(r"^([A-Z]+)(\d+)([A-Z]?)$")
for mode in ("des", "all"):
    boards = []
    for st, dets in P.items():
        keep = [k for k, d in enumerate(dets) if d[4] >= 0.25]
        texts = [dict(box=t["box"], text=t["text"], angle=None) for t in O.get(st, []) if mode == "all" or DES.match(t["text"])]
        boards.append(dict(id=st, det_index=keep, parts=[dict(box=dets[k][:4], des="", cls=int(dets[k][5])) for k in keep], texts=texts))
    json.dump(boards, open(OUT/f"e2e_boards_{mode}.json", "w"))
    print(mode, len(boards), "boards,", sum(len(b["parts"]) for b in boards), "parts,", sum(len(b["texts"]) for b in boards), "texts")
