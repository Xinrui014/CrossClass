#!/usr/bin/env python3
"""Reference images for checking classes in labelme (owner request 2026-10-18): each photo (same downscaled jpg as the labelme
files) with every component box drawn in its class colour and labelled "<designator> <BoM value>". Sources: the design check
(data/21pcb/design_fixed.json) where a photo was aligned to its design files, else the pre-labels. Also writes a labelme config
with the same fixed class colours.
  pcb21_refimg.py [stem,stem]   -> data/21pcb/labelme/ref/<stem>_ref.jpg, data/21pcb/labelme/labelme_colors.yaml"""
import json, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; LM = D/"labelme"
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
COL = {"resistor": (255, 60, 60), "capacitor": (40, 140, 255), "inductor": (255, 160, 0), "connector": (170, 90, 255),
       "diode": (0, 220, 110), "switch": (255, 0, 200), "transistor": (0, 230, 230), "IC": (255, 255, 0), "oscillator": (255, 255, 255),
       "ignore": (140, 140, 140), "text": (255, 120, 60)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def main(only):
    PL = json.load(open(D/"prelabel.json")); FIX = json.load(open(D/"design_fixed.json")); (LM/"ref").mkdir(parents=True, exist_ok=True)
    for st, v in PL["photos"].items():
        if only and st not in only: continue
        bom = PL["bom"].get(st.split("_")[0], {})
        im = Image.open(LM/f"{st}.jpg").convert("RGB"); s = im.size[0] / v["W"]; d = ImageDraw.Draw(im)
        parts = FIX[st]["parts"] if st in FIX else v["parts"]
        sizes = sorted(max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1])*s for p in parts) or [40]
        fs = max(9, min(22, int(sizes[len(sizes)//2]*0.28))); font = ImageFont.truetype(FONT, fs)
        for p in parts:
            name = CLS[p["cls"]] if p["cls"] >= 0 else "ignore"; c = COL[name]; b = [x*s for x in p["box"]]
            d.rectangle(b, outline=c, width=max(2, fs//6))
            ref = p.get("ref"); val = (bom.get(ref, {}).get("value") or "")[:12] if ref else ""
            lab = f"{ref or '?'} {val}".strip() if ref else ("?extra" if p.get("src") == "extra" else "?")
            tw = d.textlength(lab, font=font); d.rectangle([b[0], b[3], b[0]+tw+4, b[3]+fs+3], fill=(0, 0, 0))
            d.text((b[0]+2, b[3]+1), lab, fill=c, font=font)
        lf = ImageFont.truetype(FONT, 28); y = 10                                            # legend
        d.rectangle([8, 8, 260, 14+34*len(CLS)+34], fill=(0, 0, 0))
        for name in CLS + ["ignore"]:
            d.rectangle([16, y+6, 40, y+28], outline=COL[name], width=4); d.text((50, y+2), name, fill=COL[name], font=lf); y += 34
        im.save(LM/"ref"/f"{st}_ref.jpg", quality=85); print(st, "design" if st in FIX else "prelabel", flush=True)
    with open(LM/"labelme_colors.yaml", "w") as f:
        f.write("# labelme --config labelme_colors.yaml : fixed colour per class\nlabel_colors:\n")
        for k, c in COL.items(): f.write(f"  {k}: [{c[0]}, {c[1]}, {c[2]}]\n")

if __name__ == "__main__":
    main([s for s in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if s])
