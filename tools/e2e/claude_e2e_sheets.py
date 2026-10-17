#!/usr/bin/env python3
"""Sheets for Claude to read the end-to-end linked texts (I_100k links, OCR candidate boxes, p >= 0.5, 42 test photos) where PARSeq gives no
class or a class different from the detector. Crops as-is (box + 15 %), 60 per sheet (6 x 10), numbered; key = (image, text index).
Usage: claude_e2e_sheets.py E2EDIR OUTDIR"""
import json, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import RAW, pfx_class
D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}
P = json.load(open(os.path.expanduser("~/projects/gdino_fewshot/err9/preds_wacv_1280.json")))
B = {b["id"]: b for b in json.load(open(D/"e2e_boards_all.json"))}; L = json.load(open(D/"e2e_tf_links_I100k_all.json")); R = json.load(open(D/"e2e_read_all.json"))
todo = []
for st in sorted(L):
    if st in DEV: continue
    for i, (j, p) in L[st].items():
        if j is None or p < 0.5: continue
        rc = pfx_class(R[st][j]); dc = int(P[st][B[st]["det_index"][int(i)]][5])
        if rc is None or rc != dc: todo.append((st, j))
photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
font = ImageFont.load_default(); CW, CH, cols, per = 260, 80, 6, 60; cache = {}
for s in range((len(todo)+per-1)//per):
    sheet = Image.new("RGB", (cols*CW, 10*CH), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for k, (st, j) in enumerate(todo[s*per:(s+1)*per]):
        if st not in cache: cache = {st: Image.open(photo(st)).convert("RGB")}
        x0, y0, x1, y1 = B[st]["texts"][j]["box"]; w, h = x1-x0, y1-y0
        c = cache[st].crop((int(x0-0.15*w), int(y0-0.15*h), int(x1+0.15*w+1), int(y1+0.15*h+1)))
        sc = min((CW-40)/c.width, (CH-8)/c.height); c = c.resize((max(1, int(c.width*sc)), max(1, int(c.height*sc))), Image.LANCZOS)
        x, y = (k % cols)*CW, (k//cols)*CH; sheet.paste(c, (x+36, y+4)); d.text((x+4, y+CH//2-6), f"{s*per+k}", fill=(200, 0, 0), font=font)
    sheet.save(OUT/f"e{s:02d}.png")
json.dump(todo, open(OUT/"todo.json", "w")); print(len(todo), "texts,", (len(todo)+per-1)//per, "sheets")
