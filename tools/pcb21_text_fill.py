#!/usr/bin/env python3
"""Find the missing designator text boxes on photos aligned to their design files (no learned linker involved).
For every design part (data/21pcb/design_fixed.json) whose designator has no OCR text box yet, crop the photo around the part
(native resolution, window = part box + 3 x max(part size, 150 px) each side), run PP-OCRv6 on the crop as is and rotated by
90 / 180 / 270 deg, and keep a detection whose string equals the designator (after normalisation; edit distance 1 allowed for
designators of 3+ characters, and the designator as a whole token inside a longer string). The nearest such box wins.
  pcb21_text_fill.py gpu:2 [skip_stem,skip_stem]   (env ~/miniconda3/envs/ppocr6) -> data/21pcb/text_fill.json {stem: [{"box", "ref", "read", "score"}]}"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
from paddleocr import PaddleOCR
ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det", text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                device=sys.argv[1] if len(sys.argv) > 1 else "gpu:0")
norm = lambda s: re.sub(r"[^A-Z0-9_/]", "", s.upper())

def lev(a, b):
    p = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        q = [i]
        for j, cb in enumerate(b, 1): q.append(min(p[j]+1, q[j-1]+1, p[j-1]+(ca != cb)))
        p = q
    return p[-1]

def ok(read, ref, others):
    r = norm(read)
    if not r: return False
    if r == ref: return True
    if r in others: return False                    # it reads another part's designator (e.g. R123 or R13 for R12)
    if len(ref) >= 3 and abs(len(r)-len(ref)) <= 1 and lev(r, ref) <= 1: return True
    return ref in re.split(r"[^A-Z0-9_/]+", read.upper())

def boxes(arr):
    out = []
    for page in ocr.predict(arr):
        d = page if isinstance(page, dict) else getattr(page, "json", {}).get("res", {})
        for t, s, poly in zip(d.get("rec_texts", []), d.get("rec_scores", []), d.get("rec_polys", d.get("dt_polys", []))):
            p = np.array(poly, float); out.append(([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], str(t), float(s)))
    return out

def rot_back(b, k, W, H):
    """box in the crop rotated k x 90 deg clockwise -> box in the crop (W x H)"""
    x1, y1, x2, y2 = b
    if k == 0: return b
    if k == 1: return [y1, H-x2, y2, H-x1]          # rot90 cw: (x, y) -> (H-1-y, x)
    if k == 2: return [W-x2, H-y2, W-x1, H-y1]
    return [W-y2, x1, W-y1, x2]                     # rot270 cw

FIX = json.load(open(D/"design_fixed.json")); PL = json.load(open(D/"prelabel.json"))["photos"]
skip = set(s for s in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if s)     # photos the owner already edited
res = json.load(open(D/"text_fill.json")) if (D/"text_fill.json").exists() else {}; t0 = time.time()
for st, fx in FIX.items():
    if st in skip: continue
    have = {t["ref"] for t in PL[st]["texts"] if t["ref"]}
    card = st.split("_")[0]; plc = ROOT/"designfiles/21pcb/placement"/f"{card}.json"
    others = {p["ref"].upper() for p in json.load(open(plc))["parts"]} if plc.exists() else set()
    others |= {p["ref"] for p in fx["parts"] if p.get("ref")}
    im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); W0, H0 = im.size; found = []
    for p in fx["parts"]:
        ref = p.get("ref")
        if not ref or ref in have: continue
        b = p["box"]; sz = max(b[2]-b[0], b[3]-b[1], 150); m = 3*sz
        x0, y0, x1, y1 = int(max(0, b[0]-m)), int(max(0, b[1]-m)), int(min(W0, b[2]+m)), int(min(H0, b[3]+m))
        crop = im.crop((x0, y0, x1, y1)); s = min(1.0, 1600/max(crop.size))
        if s < 1: crop = crop.resize((round(crop.size[0]*s), round(crop.size[1]*s)), Image.LANCZOS)
        arr = np.asarray(crop)[:, :, ::-1]; Hc, Wc = arr.shape[:2]; cand = []
        for k in range(4):
            for bb, t, sc in boxes(np.ascontiguousarray(np.rot90(arr, k=-k))):
                if ok(t, ref, others):
                    q = rot_back(bb, k, Wc, Hc); q = [q[0]/s+x0, q[1]/s+y0, q[2]/s+x0, q[3]/s+y0]
                    d = np.hypot((q[0]+q[2])/2-(b[0]+b[2])/2, (q[1]+q[3])/2-(b[1]+b[3])/2); cand.append((d, q, t, sc))
        if cand:
            d, q, t, sc = min(cand, key=lambda c: c[0])
            found.append(dict(box=[round(v, 1) for v in q], ref=ref, read=t, score=round(sc, 3), dist_part_sizes=round(d/sz, 2)))
    res[st] = found
    n_miss = sum(1 for p in fx["parts"] if p.get("ref") and p["ref"] not in have)
    print(f"{st}: {len(found)} of {n_miss} missing designator texts found ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(D/"text_fill.json", "w"))
print("FILL_DONE")
