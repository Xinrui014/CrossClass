#!/usr/bin/env python3
"""Part boxes on 21PCB photos with the fixed PCB-Dense v22 YOLOv8l-P2 (pre-annotation, no training).
21PCB photos are 4.5k-14.6k px wide (87.8 px/mm), far above the detector's scale, so a whole-image pass is useless.
Scale estimate as in gdino_fewshot/err9/detect_scaled.py (scaleE): resize so the median resistor/capacitor long side = 29.2 px
(training median at the 1280 input). Pass 1 at a guessed scale (0402 ~ 1 mm -> x0.3), median R/C from its detections
(score >= 0.25), pass 2 at the corrected scale. Tiles 1280 px, 256 overlap, boxes touching an inner tile border dropped,
class-wise NMS 0.6, conf 0.001, max 3000 / photo.
  pcb21_detect.py --gpu 2 [--only Card6_,Card19_]   (env ~/venvs/pcbid) -> data/21pcb/preds_det.json {stem: [[x1,y1,x2,y2,score,cls]]}"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image
from ultralytics import YOLO
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
W = Path.home()/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/weights/best.pt"
TARGET, T, OV, GUESS = 29.2, 1280, 256, 0.3
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="0"); ap.add_argument("--only", default="")
ap.add_argument("--out", default=str(D/"preds_det.json")); a = ap.parse_args()
m = YOLO(str(W))

def pred(arrs):
    out = []
    for i in range(0, len(arrs), 8):
        for r in m.predict(source=arrs[i:i+8], imgsz=T, conf=0.001, iou=0.6, max_det=3000, verbose=False, device=a.gpu):
            b = r.boxes; out.append(np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()] if len(b) else np.zeros((0, 6)))
    return out

def tiles(arr, scale):
    """arr already resized by `scale`; returns boxes in ORIGINAL pixels."""
    Hh, Ww = arr.shape[:2]; boxes, crops, offs = [], [], []
    for y0 in range(0, max(Hh-OV, 1), T-OV):
        for x0 in range(0, max(Ww-OV, 1), T-OV):
            c = arr[y0:y0+T, x0:x0+T]
            if c.shape[0] >= 32 and c.shape[1] >= 32: crops.append(np.ascontiguousarray(c)); offs.append((x0, y0, c.shape[1], c.shape[0]))
    for (x0, y0, w, h), p in zip(offs, pred(crops)):
        if not len(p): continue
        inner = np.zeros(len(p), bool)
        if x0 > 0: inner |= p[:, 0] <= 3
        if y0 > 0: inner |= p[:, 1] <= 3
        if x0 + w < Ww: inner |= p[:, 2] >= w - 3
        if y0 + h < Hh: inner |= p[:, 3] >= h - 3
        p = p[~inner].copy(); p[:, [0, 2]] += x0; p[:, [1, 3]] += y0; p[:, :4] /= scale; boxes.append(p)
    p = np.vstack(boxes) if boxes else np.zeros((0, 6))
    if not len(p): return p
    k = torchvision.ops.batched_nms(torch.tensor(p[:, :4]), torch.tensor(p[:, 4]), torch.tensor(p[:, 5]).long(), 0.6).numpy()[:3000]
    return p[k]

def run(im, scale):
    Wn, Hn = im.size
    return tiles(np.asarray(im.resize((round(Wn*scale), round(Hn*scale)), Image.LANCZOS))[:, :, ::-1], scale)

def median_rc(p):
    rc = [max(b[2]-b[0], b[3]-b[1]) for b in p if b[4] >= 0.25 and int(b[5]) in (0, 1)]
    return float(np.median(rc)) if len(rc) >= 5 else None

only = [s for s in a.only.split(",") if s]
out = json.load(open(a.out)) if os.path.exists(a.out) else {}; t0 = time.time()
info = json.load(open(D/"preds_det_info.json")) if (D/"preds_det_info.json").exists() else {}
# photos with too few R/C to estimate (sparse bottoms) take the scale of the other side of the same board (same rig, same parts);
# run those last so the sibling's scale is known
imgs = sorted((D/"images").glob("*.jpg"), key=lambda p: (p.stem in info and info[p.stem]["median_rc_native"] is None, p.stem))
for ip in imgs:
    st = ip.stem
    if only and not any(st.startswith(o) for o in only): continue
    im = Image.open(ip).convert("RGB")
    p1 = run(im, GUESS); med = median_rc(p1)
    sib = [v["scale"] for k, v in info.items() if k != st and k.split("_")[0] == st.split("_")[0] and v["median_rc_native"]]
    scale = float(np.clip(TARGET/med, 0.05, 1.0)) if med else (sib[0] if sib else 0.12)
    p2 = p1 if abs(scale/GUESS - 1) < 0.1 else run(im, scale)
    out[st] = [[float(v) for v in row[:5]] + [int(row[5])] for row in p2]
    info[st] = dict(median_rc_native=med, scale=scale, scale_from="own" if med else ("sibling" if sib else "default"), n=len(p2), n25=int((p2[:, 4] >= 0.25).sum()) if len(p2) else 0)
    print(st, info[st], f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(a.out, "w")); json.dump(info, open(D/"preds_det_info.json", "w"), indent=1)
print("DET_DONE", round(time.time()-t0))
