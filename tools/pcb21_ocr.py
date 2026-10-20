#!/usr/bin/env python3
"""PP-OCRv6 texts on 21PCB photos (pre-annotation). Same protocol as gdino_fewshot/assoc/ocr_wacv.py (det + rec + text-line
orientation, 1280-px tiles, 256 overlap, two passes: as is and rotated 90 deg, de-dup IoU 0.5), but the photo is first
downscaled by SCALE (21PCB is 87.8 px/mm, silkscreen text ~70-90 px tall native; x0.4 gives ~30 px, close to WACV's 27 px).
Each text is then matched to the board's BoM designators (both sides, DNI rows included because they can be printed):
exact match after normalisation, else a unique designator at edit distance 1 -> field "ref".
  pcb21_ocr.py gpu:2 [Card6_,Card19_]   (env ~/miniconda3/envs/ppocr6) -> data/21pcb/ocr.json
  {stem: [{"box": [x1,y1,x2,y2] native px, "text", "score", "pass": 0|90, "ref": str|null}]}"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
SCALE, T, OV = 0.4, 1280, 256
from paddleocr import PaddleOCR
ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det", text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                device=sys.argv[1] if len(sys.argv) > 1 else "gpu:0")
only = [s for s in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if s]

def bom_refs():
    refs = {}
    for f in ("tabular_rows.json", "pdf_rows.json"):
        for card, v in json.load(open(D/f)).items():
            refs.setdefault(card, set()).update(r["ref"].strip().upper() for r in v["rows"] if r.get("ref"))
    return refs

def lev(a, b):
    p = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        q = [i]
        for j, cb in enumerate(b, 1): q.append(min(p[j]+1, q[j-1]+1, p[j-1]+(ca != cb)))
        p = q
    return p[-1]

def match(t, refs):
    s = re.sub(r"[^A-Z0-9]", "", t.upper().replace("O", "0") if re.fullmatch(r"[A-Z]+[0-9O]+", t.upper()) else t.upper())
    if not s or not re.match(r"[A-Z]", s): return None
    if s in refs: return s
    c = [r for r in refs if abs(len(r)-len(s)) <= 1 and lev(r, s) == 1]
    return c[0] if len(c) == 1 else None

def run(arr):
    H, W = arr.shape[:2]; out = []
    for y0 in range(0, max(H-OV, 1), T-OV):
        for x0 in range(0, max(W-OV, 1), T-OV):
            tile = arr[y0:y0+T, x0:x0+T]
            if tile.shape[0] < 32 or tile.shape[1] < 32: continue
            for page in ocr.predict(tile):
                d = page if isinstance(page, dict) else getattr(page, "json", {}).get("res", {})
                for t, s, poly in zip(d.get("rec_texts", []), d.get("rec_scores", []), d.get("rec_polys", d.get("dt_polys", []))):
                    p = np.array(poly, float); out.append(([p[:, 0].min()+x0, p[:, 1].min()+y0, p[:, 0].max()+x0, p[:, 1].max()+y0], str(t), float(s)))
    return out

def iou1(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)

REFS = bom_refs(); out_f = D/"ocr.json"
res = json.load(open(out_f)) if out_f.exists() else {}; t0 = time.time()
for ip in sorted((D/"images").glob("*.jpg")):
    st = ip.stem
    if only and not any(st.startswith(o) for o in only): continue
    im = Image.open(ip).convert("RGB"); W0, H0 = im.size
    im = np.asarray(im.resize((round(W0*SCALE), round(H0*SCALE)), Image.LANCZOS))[:, :, ::-1].copy(); H, W = im.shape[:2]
    a = [(b, t, s, 0) for b, t, s in run(im)]
    r = np.ascontiguousarray(np.rot90(im, k=-1))            # 90 deg clockwise: (x, y) -> (H-1-y, x)
    for (bx1, by1, bx2, by2), t, s in run(r): a.append(([by1, H-1-bx2, by2, H-1-bx1], t, s, 90))
    a.sort(key=lambda z: -z[2]); keep = []
    for z in a:
        if all(iou1(z[0], k[0]) <= 0.5 for k in keep): keep.append(z)
    refs = REFS.get(st.split("_")[0], set())
    res[st] = [dict(box=[round(v/SCALE, 1) for v in b], text=t, score=round(s, 3), ref=match(t, refs), **{"pass": p}) for b, t, s, p in keep]
    n_ref = sum(x["ref"] is not None for x in res[st])
    print(f"{st}: {len(keep)} texts, {n_ref} match a BoM designator, {len(refs)} BoM refs ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out_f, "w"))
print("OCR_DONE", sum(map(len, res.values())), round(time.time()-t0))
