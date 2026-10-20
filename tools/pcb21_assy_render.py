#!/usr/bin/env python3
"""Render the vendors' own assembly drawings / layout PDFs (from the design packages) to JPG pages for viewing next to labelme
(owner wants the original design drawings, not our overlays). Long side capped at MAXS px.
  pcb21_assy_render.py   (env with PyMuPDF, e.g. ~/miniconda3/envs/ocr_ab)  -> page/ref21/assy/<Card>_<k>.jpg + pages.json"""
import json, os, re
from pathlib import Path
import fitz
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); DF = ROOT/"designfiles/21pcb"; OUT = ROOT/"page/ref21/assy"
U = DF/"unz"; MAXS = 5000
SRC = {
    "Card5": [DF/"Card5_TPS65982EVM/slvc618/Schematic and BOM/HVL117D_Assy.PDF"],
    "Card6": [U/"cn0540-DesignSupport/CN0540-DesignSupport/CN0540 CAD Files/assy/01-059358-01-b.pdf"],
    "Card7": [U/"cn0395-DesignSupport/CN0395-DesignSupport/EVAL-CN0395-ARDZ-TopAssembly-RevB.pdf",
              U/"cn0395-DesignSupport/CN0395-DesignSupport/EVAL-CN0395-ARDZ-BottomAssembly-RevB.pdf"],
    "Card8": [U/"cn0506-DesignSupport/cn0506-designsupport/CN0506 assembly files/01-054610-01-b.pdf"],
    "Card9": [U/"cn0216-sdp-designsupport/CN0216-Design Support/EVAL-CN0216-SDPZ-AssemblyDrawing-RevA.pdf"],
    "Card16": [U/"CN0272-DesignSupport/CN0272-DesignSupport/EVAL-CN0272-SDPZ-AssemblyDrawing-RevC.pdf"],
    "Card17": [U/"CN0321-DesignSupport/CN0321-DesignSupport/EVAL-CN0321-SDPZ Layout.pdf"],
    "Card19": [DF/"Card19_TMDSIDK437X/tidrc76a/Revision 1p4A/AM437X_INDUSTRIAL_EVM_3K0008_PCB_REV1_4A_ASSEMBLY_TOP.pdf",
               DF/"Card19_TMDSIDK437X/tidrc76a/Revision 1p4A/AM437X_INDUSTRIAL_EVM_3K0008_PCB_REV1_4A_ASSEMBLY_BOTTOM.pdf"],
}
OUT.mkdir(parents=True, exist_ok=True); pages = {}
for card, files in SRC.items():
    k = 0; pages[card] = []
    for f in files:
        doc = fitz.open(f)
        for pg in doc:
            # crop to the board: the box around all designator-like words (vector text), + 6 % margin; else the whole page
            w = [x for x in pg.get_text("words") if re.fullmatch(r"[A-Z]{1,4}[0-9]{1,4}[A-Z]?", x[4])]
            clip = pg.rect
            if len(w) >= 10:   # robust: drop words far from the main cluster (assembly notes, title block)
                cx = sorted((x[0]+x[2])/2 for x in w); cy = sorted((x[1]+x[3])/2 for x in w); n = len(w)
                lo_x, hi_x, lo_y, hi_y = cx[n//20], cx[-1-n//20], cy[n//20], cy[-1-n//20]
                sx, sy = 0.25*(hi_x-lo_x), 0.25*(hi_y-lo_y)
                w = [x for x in w if lo_x-sx <= (x[0]+x[2])/2 <= hi_x+sx and lo_y-sy <= (x[1]+x[3])/2 <= hi_y+sy]
                r = fitz.Rect(min(x[0] for x in w), min(x[1] for x in w), max(x[2] for x in w), max(x[3] for x in w))
                m = 0.08 * max(r.width, r.height); clip = fitz.Rect(r.x0-m, r.y0-m, r.x1+m, r.y1+m) & pg.rect
            z = MAXS / max(clip.width, clip.height); pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip, alpha=False)
            name = f"{card}_{k}.jpg"; pix.save(str(OUT/name))
            pages[card].append(dict(file=name, src=f.name, page=pg.number + 1, cropped=clip != pg.rect)); k += 1
        print(card, f.name, len(doc), "pages", flush=True)
json.dump(pages, open(OUT/"pages.json", "w"), indent=1)
