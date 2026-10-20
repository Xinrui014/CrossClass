#!/usr/bin/env python3
"""Class suggestions for a photo whose board has only a layout PDF (no readable placement), e.g. Card13 (CN0254, PADS binary).
1. Read every designator word and its position from the PDF silkscreen page (pypdfium2 char boxes; PyMuPDF crashes on it).
2. Align PDF silkscreen to the photo with a homography from the photo OCR texts that read a BoM designator (same silkscreen
   text in both), RANSAC.
3. For every component box of the photo labelme file, the nearest projected PDF designator (centre distance <= 1.5 x box size)
   gives a suggested designator and its BoM class; boxes whose current label disagrees are listed (suggestions only: the
   nearest label is not always the part's own label).
  pcb21_pdf_silk.py PDF PAGE STEM LABELME_JSON   (env with pypdfium2 + cv2, e.g. ~/miniconda3/envs/ravio)
  -> results/pcb21/pdf_silk_<stem>.json"""
import json, os, re, sys
from pathlib import Path
import numpy as np, cv2, pypdfium2 as pdfium
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
pdf_f, page, st, lm_f = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

def words(pg):
    tp = pg.get_textpage(); out = []; cur, box = "", None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch.strip():
            if cur: out.append((cur, box))
            cur, box = "", None; continue
        l, b, r, t = tp.get_charbox(i)
        if cur and box and l - box[2] > 0.6*(box[3]-box[1]) and abs(b - box[1]) > 0.5*(box[3]-box[1]):
            out.append((cur, box)); cur, box = "", None
        cur += ch; box = [l, b, r, t] if box is None else [min(box[0], l), min(box[1], b), max(box[2], r), max(box[3], t)]
    if cur: out.append((cur, box))
    H = pg.get_size()[1]
    return [(w, [bx[0], H-bx[3], bx[2], H-bx[1]]) for w, bx in out]            # y down like the photo

PL = json.load(open(D/"prelabel.json")); card = st.split("_")[0]; bom = PL["bom"][card]; v = PL["photos"][st]
W = {w: b for w, b in words(pdfium.PdfDocument(pdf_f)[page-1]) if w in bom}
src, dst = [], []
for t in v["texts"]:
    if t["ref"] in W:
        b = W[t["ref"]]; src.append([(b[0]+b[2])/2, (b[1]+b[3])/2]); dst.append([(t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2])
src, dst = np.float32(src), np.float32(dst)
for mirror in (False, True):                       # bottom-side prints may be mirrored
    s2 = src.copy()
    if mirror: s2[:, 0] *= -1
    H, inl = cv2.findHomography(s2, dst, cv2.RANSAC, 60.0, maxIters=20000)
    if H is not None and inl.sum() >= max(8, 0.5*len(src)): break
res = np.linalg.norm(cv2.perspectiveTransform(s2[inl.ravel() == 1][None], H)[0] - dst[inl.ravel() == 1], axis=1)
names = list(W); P = np.float32([[(W[n][0]+W[n][2])/2 * (-1 if mirror else 1), (W[n][1]+W[n][3])/2] for n in names])
Pp = cv2.perspectiveTransform(P[None], H)[0]
lm = json.load(open(lm_f)); sc = lm["imageWidth"] / v["W"]; out = []
for s in lm["shapes"]:
    if s["label"] == "text": continue
    (x1, y1), (x2, y2) = s["points"][:2]; b = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]) / sc
    c = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]); sz = max(b[2]-b[0], b[3]-b[1], 80)
    d = np.linalg.norm(Pp - c, axis=1); k = int(d.argmin())
    sug = names[k] if d[k] <= 1.5*sz else None; bc = CLS[bom[sug]["cls"]] if sug and bom[sug]["cls"] >= 0 else ("ignore" if sug else None)
    out.append(dict(box=[round(float(x), 1) for x in b], label=s["label"], desc=s.get("description") or "", nearest=sug,
                    nearest_dist=round(float(d[k]/sz), 2), bom_class=bc, bom_raw=bom[sug]["raw"] if sug else None,
                    disagree=bool(sug and bc != s["label"])))
summ = dict(photo=st, pdf_words=len(W), pairs=len(src), inliers=int(inl.sum()), mirrored=mirror, residual_px=round(float(np.median(res)), 1),
            boxes=len(out), with_nearest=sum(o["nearest"] is not None for o in out), disagree=sum(o["disagree"] for o in out))
json.dump(dict(summary=summ, boxes=out), open(ROOT/"results/pcb21"/f"pdf_silk_{st}.json", "w"), indent=1)
print(json.dumps(summ))
for o in out:
    if o["disagree"]: print(f"  {o['label']:>10} desc={o['desc'] or '-':6} nearest {o['nearest']:6} ({o['bom_raw']}, d={o['nearest_dist']})")
