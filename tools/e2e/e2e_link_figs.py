#!/usr/bin/env python3
"""Example grids of end-to-end links (I_100k, all OCR candidate boxes, p >= 0.5; clean41 labelled parts) from e2e_linkacc.py rows.
Blue = detector box, green = annotated label box, red = the text box the linker chose (orange when it is the right one). Caption: GT
designator, reading of the chosen box (Claude's where Claude read it, else PARSeq), link probability. Usage: E2EDIR ROWS TODO READS OUTDIR"""
import json, os, sys, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import RAW
D = Path(sys.argv[1]); rows = json.load(open(sys.argv[2])); OUT = Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[3])), json.load(open(sys.argv[4])))}
RD = json.load(open(D/"e2e_read_all.json"))
photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
try: font = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError: font = ImageFont.load_default()
CW, CH, TH, cols = 300, 230, 40, 6; cache = {}
def grid(name, sel):
    n = len(sel); R = (n+cols-1)//cols; sheet = Image.new("RGB", (cols*CW, R*(CH+TH)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for k, r in enumerate(sel):
        if r["img"] not in cache: cache.clear(); cache[r["img"]] = Image.open(photo(r["img"])).convert("RGB")
        im = cache[r["img"]]; bxs = [b for b in (r["pbox"], r.get("dbox"), r["tbox"], r.get("lbox")) if b]
        x0, y0 = min(b[0] for b in bxs), min(b[1] for b in bxs); x1, y1 = max(b[2] for b in bxs), max(b[3] for b in bxs)
        pw, ph = max(x1-x0, 20), max(y1-y0, 20); x0 -= 0.25*pw; y0 -= 0.25*ph; x1 += 0.25*pw; y1 += 0.25*ph
        c = im.crop((int(x0), int(y0), int(x1)+1, int(y1)+1)); s = min(CW/c.width, CH/c.height); c = c.resize((max(1, int(c.width*s)), max(1, int(c.height*s))), Image.LANCZOS)
        cd = ImageDraw.Draw(c); T = lambda b: [(b[0]-int(x0))*s, (b[1]-int(y0))*s, (b[2]-int(x0))*s, (b[3]-int(y0))*s]
        if r.get("dbox"): cd.rectangle(T(r["dbox"]), outline=(30, 110, 255), width=3)
        cd.rectangle(T(r["tbox"]), outline=(0, 200, 60), width=3)
        if r.get("lbox"): cd.rectangle(T(r["lbox"]), outline=(255, 150, 0) if r["stage"] == "linked to the right text" else (230, 20, 20), width=3)
        X, Y = (k % cols)*CW, (k//cols)*(CH+TH); sheet.paste(c, (X+(CW-c.width)//2, Y))
        if r.get("link") is not None:
            key = (r["img"], r["link"]); rd = CL.get(key, RD[r["img"]][r["link"]]); who = "Claude" if key in CL else "PARSeq"
            cap2 = f"chosen box reads \"{rd or '-'}\" ({who}), p {r['prob']:.2f}"
        elif r.get("raw") is not None: cap2 = f"best box p {r['prob']:.2f} < 0.5 -> none"
        else: cap2 = "linker: none" if r.get("dbox") else "no detector box"
        d.text((X+6, Y+CH+3), f"GT {r['des']}  ({r['img']})", fill=(0, 0, 0), font=font); d.text((X+6, Y+CH+20), cap2, fill=(60, 60, 60), font=font)
    sheet.save(OUT/f"{name}.jpg", quality=88)
random.seed(0); by = {}
for r in rows: by.setdefault(r["stage"], []).append(r)
import re; DES = re.compile(r"^[A-Z]+\d+[A-Z]?$")
wr = by["linked to a wrong text"]; junk = [r for r in wr if not DES.match(r["ltext"] or "")]; desw = [r for r in wr if DES.match(r["ltext"] or "")]
for name, pool, n in (("right", by["linked to the right text"], 12), ("wrong_junk", junk, 24), ("wrong_des", desw, 12),
                      ("none", by["linked to none"], 18), ("ocr_missed", by["OCR has no box on the label"], 12)):
    grid(name, random.sample(pool, min(n, len(pool)))); print(name, len(pool))
