#!/usr/bin/env python3
"""End-to-end inputs for the 21PCB test set (28 photos), same format as the WACV run (tools/e2e/e2e_build.py):
- data/gt9_21pcb.json: GT in the scorer format {stem: {W, H, boxes: [[x1,y1,x2,y2, cls9 (-1 ignore), label]]}} from
  data/21pcb/final_v1/gt9_21pcb.json (owner's final labels);
- data/21pcb/e2e/e2e_boards_all.json: per photo, parts = detector boxes with score >= 0.25 (data/21pcb/e2e/preds_cls9.json, file order,
  det_index = their index), texts = every PP-OCRv6 box of data/21pcb/ocr.json (native pixels).
Then: text_features.py boxes --imgdir data/21pcb/images (PARSeq features) and tools/e2e/e2e_filter.py (rule boxes).
  pcb21_e2e_prep.py   (any python3)"""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; E = D/"e2e"
G = json.load(open(D/"final_v1/gt9_21pcb.json"))
json.dump({st: dict(W=v["W"], H=v["H"], boxes=[b[:6] for b in v["boxes"]]) for st, v in G.items()}, open(ROOT/"data/gt9_21pcb.json", "w"))
P = json.load(open(E/"preds_cls9.json")); O = json.load(open(D/"ocr.json")); boards = []
for st in sorted(G):
    keep = [k for k, d in enumerate(P[st]) if d[4] >= 0.25]
    boards.append(dict(id=st, det_index=keep, parts=[dict(box=P[st][k][:4], des="", cls=int(P[st][k][5])) for k in keep],
                       texts=[dict(box=t["box"], text=t["text"], angle=None) for t in O[st]]))
json.dump(boards, open(E/"e2e_boards_all.json", "w"))
print(len(boards), "boards,", sum(len(b["parts"]) for b in boards), "parts,", sum(len(b["texts"]) for b in boards), "texts;",
      sum(len(v["boxes"]) for v in G.values()), "GT boxes")
