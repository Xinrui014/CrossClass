#!/usr/bin/env python3
"""Detector baseline on the 28 21PCB test photos with the full 9-class score vector per box (needed by the soft cascade and the
proposal-B linker, like data/preds_wacv_1280_cls9.json for WACV). Fixed PCB-Dense v22 YOLOv8l-P2, no training.
Same scale protocol as tools/pcb21_detect.py: resize each photo by the scale in data/21pcb/preds_det_info.json (median R/C long side
-> 29.2 px), 1280-px tiles with 256 overlap (edge tiles padded with grey 114, not rescaled), raw head output (xywh + 9 sigmoid scores),
per tile: conf = max class score >= 0.001, class-wise NMS 0.6 on the argmax class; tile boxes touching an inner tile border dropped;
all boxes class-wise NMS 0.6, max 3000 per photo.
  pcb21_detect9.py --gpu 2   (env ~/venvs/pcbid)
  -> data/21pcb/e2e/preds_cls9.json {stem: [[x0,y0,x1,y1,score,cls,s0..s8]]} native pixels (s = Resistor .. Oscillator)"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image
from ultralytics import YOLO
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; OUT = D/"e2e"
W = Path.home()/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/weights/best.pt"
T, OV = 1280, 256
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="0"); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); net = YOLO(str(W)).model.to(dev).eval()
IC28 = sorted(p.stem for p in (Path.home()/"projects/tcassi/data/21PCBs/IC_annotation").glob("*.json"))
INFO = json.load(open(D/"preds_det_info.json"))

def tile_preds(arr):
    """arr: resized RGB uint8 (H, W, 3) -> (N, 4+9) boxes in resized pixels."""
    Hh, Ww = arr.shape[:2]; out = []
    origins = [(x0, y0) for y0 in range(0, max(Hh-OV, 1), T-OV) for x0 in range(0, max(Ww-OV, 1), T-OV)]
    for i in range(0, len(origins), 8):
        batch, meta = [], []
        for x0, y0 in origins[i:i+8]:
            c = arr[y0:y0+T, x0:x0+T]
            if c.shape[0] < 32 or c.shape[1] < 32: continue
            pad = np.full((T, T, 3), 114, np.uint8); pad[:c.shape[0], :c.shape[1]] = c; batch.append(pad); meta.append((x0, y0, c.shape[1], c.shape[0]))
        if not batch: continue
        x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            y = net(x); y = y[0] if isinstance(y, (list, tuple)) else y           # (B, 4+9, N)
        for (x0, y0, w, h), p in zip(meta, y.float().transpose(1, 2)):
            s = p[:, 4:]; conf, cls = s.max(1); k = conf >= 0.001
            p, conf, cls = p[k], conf[k], cls[k]
            xyxy = torch.cat([p[:, :2] - p[:, 2:4]/2, p[:, :2] + p[:, 2:4]/2], 1)
            keep = torchvision.ops.batched_nms(xyxy, conf, cls, 0.6)[:3000]; xyxy, s = xyxy[keep], p[keep, 4:]
            inner = torch.zeros(len(xyxy), dtype=torch.bool, device=dev)
            if x0 > 0: inner |= xyxy[:, 0] <= 3
            if y0 > 0: inner |= xyxy[:, 1] <= 3
            if x0 + w < Ww: inner |= xyxy[:, 2] >= w - 3
            if y0 + h < Hh: inner |= xyxy[:, 3] >= h - 3
            xyxy, s = xyxy[~inner], s[~inner]
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] + x0).clamp(0, Ww); xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] + y0).clamp(0, Hh)
            out.append(torch.cat([xyxy, s], 1).cpu())
    return torch.cat(out) if out else torch.zeros((0, 13))

res = {}; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
for st in IC28:
    im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); sc = INFO[st]["scale"]
    arr = np.asarray(im.resize((round(im.size[0]*sc), round(im.size[1]*sc)), Image.LANCZOS))
    p = tile_preds(arr)
    if len(p):
        conf, cls = p[:, 4:].max(1); keep = torchvision.ops.batched_nms(p[:, :4], conf, cls, 0.6)[:3000]; p, conf, cls = p[keep], conf[keep], cls[keep]
        p[:, :4] /= sc
    res[st] = [[round(float(v), 2) for v in r[:4]] + [round(float(c), 5), int(k)] + [round(float(v), 5) for v in r[4:]] for r, c, k in zip(p, conf, cls)]
    print(st, f"scale {sc:.3f}", len(res[st]), "boxes", sum(r[4] >= 0.25 for r in res[st]), ">=0.25", f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(OUT/"preds_cls9.json", "w")); print("DET9_DONE")
