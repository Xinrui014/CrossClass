#!/usr/bin/env python3
"""Read every batch-1 text box of 21PCB with PP-OCRv6 recognition (the box is given, no detection) for the owner's string check
(owner 2026-10-20: check the text-box strings first, links later). No design file, no BoM, no linker involved.
Boxes: data/21pcb/owner_final2/pcb21_batch1/<stem>.json, shapes with label "text" (12 design-aligned photos). Each box is cropped
from the native photo (data/21pcb/images, padding 15 % of the short side + 4 px) and read at the orientations its shape allows
(wide: 0 / 180 deg, tall: 90 / 270 deg, near square: all four). A designator-like reading (DES: 1-5 letters + digits, e.g. R174,
TP6) with score >= 0.5 beats any other reading (upside-down text often reads as confident digits, e.g. '1219'), then the score.
Labels: "text_check" when the score is < CHECK_SCORE, the reading is empty, or it disagrees with the string the box already had
(then the description is "<new reading>|<old string>", the owner keeps the right one); else "text_other" when the reading is not
designator-like (pin numbers, function names, chip markings: the owner skims them); else "text".
Outputs: data/21pcb/text_check_b1/ (12 labelme json with the text boxes only + jpg, labels.txt, labelme_colors.yaml, README.txt,
MD5.txt), data/21pcb/text_check_b1.zip, data/21pcb/text_check_b1_ocr.json (per box: native box, old, read, score, rotation).
  pcb21_text_ocr.py [cpu|gpu:N]   (env ~/miniconda3/envs/ppocr6)"""
import hashlib, json, os, re, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
SRC = D/"owner_final2/pcb21_batch1"; OUT = D/"text_check_b1"; CHECK_SCORE = 0.90
from paddleocr import TextRecognition
rec = TextRecognition(model_name="PP-OCRv6_medium_rec", device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
norm = lambda s: re.sub(r"[^A-Z0-9_/+.\-]", "", (s or "").upper())
DES = re.compile(r"^[A-Z]{1,5}[0-9]{1,4}[A-Z]?$"); des_like = lambda s: bool(DES.match(norm(s)))

OUT.mkdir(parents=True, exist_ok=True); side = {}; t0 = time.time(); tot = {"boxes": 0, "check": 0}
for f in sorted(SRC.glob("*.json")):
    st = f.stem; doc = json.load(open(f)); texts = [s for s in doc["shapes"] if s["label"] == "text"]
    im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); W0, H0 = im.size; sc = doc["imageWidth"] / W0
    crops, owner = [], []
    for i, s in enumerate(texts):
        (x1, y1), (x2, y2) = s["points"][:2]
        b = [min(x1, x2)/sc, min(y1, y2)/sc, max(x1, x2)/sc, max(y1, y2)/sc]; w, h = b[2]-b[0], b[3]-b[1]; p = 0.15*min(w, h) + 4
        c = np.asarray(im.crop((int(max(0, b[0]-p)), int(max(0, b[1]-p)), int(min(W0, b[2]+p)), int(min(H0, b[3]+p)))))[:, :, ::-1]
        r = w / max(h, 1e-6); ks = (0, 2) if r >= 1.4 else (1, 3) if r <= 0.7 else (0, 1, 2, 3)
        for k in ks: crops.append(np.ascontiguousarray(np.rot90(c, k=-k))); owner.append((i, k))
        s["_b"] = [round(v, 1) for v in b]
    best = {}
    for (i, k), r in zip(owner, rec.predict(input=crops, batch_size=32)):
        d = r if isinstance(r, dict) else r.json.get("res", {}); t, q = str(d.get("rec_text", "")).strip(), float(d.get("rec_score", 0.0))
        key = (des_like(t) and q >= 0.5, q)
        if i not in best or key > best[i][3]: best[i] = (t, q, k, key)
    shapes, rows = [], []
    for i, s in enumerate(texts):
        t, q, k, _ = best.get(i, ("", 0.0, 0, None)); old = (s.get("description") or "").strip(); o = old.lstrip("?")
        why = [w for w, c in (("low score", q < CHECK_SCORE), ("empty read", not norm(t)), ("not designator-like", norm(t) and not des_like(t)),
                              ("differs from old", o and norm(o) != norm(t))) if c]
        desc = f"{t}|{old}" if "differs from old" in why else (t or old)
        lab = "text_other" if why == ["not designator-like"] else "text_check" if why else "text"
        shapes.append(dict(label=lab, points=s["points"], group_id=s.get("group_id"), description=desc,
                           shape_type=s.get("shape_type", "rectangle"), flags=s.get("flags") or {}, mask=None))
        rows.append(dict(i=i, box=s["_b"], old=old, read=t, score=round(q, 4), rot=90*k, check=why))
    json.dump(dict(version=doc.get("version", "5.4.1"), flags={}, shapes=shapes, imagePath=f"{st}.jpg", imageData=None,
                   imageHeight=doc["imageHeight"], imageWidth=doc["imageWidth"]), open(OUT/f"{st}.json", "w"), indent=1)
    shutil.copy2(D/"labelme"/f"{st}.jpg", OUT/f"{st}.jpg"); side[st] = rows
    n = sum(1 for r in rows if r["check"] and r["check"] != ["not designator-like"]); tot["boxes"] += len(rows); tot["check"] += n
    print(f"{st}: {len(rows)} text boxes, {n} to check ({time.time()-t0:.0f}s)", flush=True)
json.dump(side, open(D/"text_check_b1_ocr.json", "w"), indent=1)
(OUT/"labels.txt").write_text("text\ntext_other\ntext_check\n")
(OUT/"labelme_colors.yaml").write_text("# labelme --config labelme_colors.yaml\nlabel_colors:\n  text: [0, 220, 110]\n  text_other: [40, 140, 255]\n  text_check: [255, 60, 60]\n")
(OUT/"README.txt").write_text(
    "21PCB batch 1: text-box strings to check (text boxes only; components are not in these files).\n"
    "Open:  labelme pcb21_text_b1 --labels pcb21_text_b1/labels.txt --config pcb21_text_b1/labelme_colors.yaml --nodata --autosave\n"
    "description = what is printed in the box, read by PP-OCRv6.\n"
    "label text_check (red) = please check: low OCR score, empty reading, or the new reading differs from the old string;\n"
    "  in the last case the description is NEW|OLD: keep the right one (or type the printed text), then set the label to text.\n"
    "label text_other (blue) = OCR is confident but the text is not designator-like (GND, RESET, pin numbers, chip markings): skim.\n"
    "label text (green) = designator-like, OCR is confident and agrees with the old string (spot checks welcome).\n"
    "Delete a box that is not text. Close labelme before the upload.\n")
files = sorted(p for p in OUT.iterdir() if p.name != "MD5.txt")
(OUT/"MD5.txt").write_text("".join(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files if p.suffix == ".json"))
with zipfile.ZipFile(D/"text_check_b1.zip", "w") as z:
    for p in sorted(OUT.iterdir()): z.write(p, f"pcb21_text_b1/{p.name}")
print("TOTAL", tot, "->", OUT)
