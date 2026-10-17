#!/usr/bin/env python3
"""Recover the full 9-class score vector of every detection in gdino_fewshot/err9/preds_wacv_1280.json.

What: preds_wacv_1280.json keeps only the top class and its score per box. It was made by gdino_fewshot/err9/dump_all9.py
(--set wacv, default imgsz 1280): PCB-Dense v22 YOLOv8l-P2 (yolov8l_p2_notile/weights/best.pt), ultralytics predict on
one photo per call from data/wacv_eval/images, imgsz 1280, conf 0.001, iou 0.6, max_det 3000, no TTA, no tiling.
How: re-run the same model with the same predict call on the CPU, capture the raw head output before NMS (per anchor:
xywh in letterbox pixels + 9 sigmoid class scores), map the boxes to original photo pixels with the predictor's own
xywh2xyxy + scale_boxes (the same steps the predictor uses), then for each stored detection take the raw candidate with the
same top class and the highest IoU (must be >= 0.9) and attach its 9 scores. Check: the candidate's score of the detection's
own class should equal the stored score (within 1e-3). The re-run NMS output is also compared with the stored file.
Output: data/preds_wacv_1280_cls9.json = {image id: [[x0, y0, x1, y1, score, cls, s0, ..., s8], ...]}, same keys and order
as the input, the first 6 values unchanged, s0..s8 = Resistor, Capacitor, Inductor, Connector, Diode, Switch, Transistor,
IC, Oscillator; null for the 9 scores of an unmatched detection.
Usage (CPU only, the L40 GPUs are shared):
  CUDA_VISIBLE_DEVICES="" nice -n 15 ~/venvs/pcbid/bin/python tools/det_cls9.py [--threads 16] [--limit N --out /tmp/x.json]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torchvision
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops

H = Path.home()
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass"))
IMG = H/"projects/gdino_fewshot/data/wacv_eval/images"            # symlinks to RAW/<id>/<id>.jpg (same files dump_all9.py read)
SRC = H/"projects/gdino_fewshot/err9/preds_wacv_1280.json"
W = H/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/weights/best.pt"
NC, IOU_MIN, TOL, KEEP_MIN = 9, 0.9, 1e-3, 5e-4                  # KEEP_MIN: raw candidates below this top score are dropped (stored min = 0.001)

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=str(ROOT/"data/preds_wacv_1280_cls9.json")); ap.add_argument("--weights", default=str(W))
a = ap.parse_args()
torch.set_num_threads(a.threads)

CAP = {}
_orig_pp = DetectionPredictor.postprocess
def _capture_pp(self, preds, img, orig_imgs, **kw):
    """Keep a copy of the raw head output (NMS edits the tensor in place) and the two image shapes, then run the normal path."""
    p = preds[0] if isinstance(preds, (list, tuple)) else preds
    CAP["raw"] = p.detach().float().cpu().clone(); CAP["img_shape"] = tuple(img.shape[2:]); CAP["orig_shape"] = orig_imgs[0].shape
    return _orig_pp(self, preds, img, orig_imgs, **kw)
DetectionPredictor.postprocess = _capture_pp

P = json.load(open(SRC)); m = YOLO(a.weights); print(m.names, flush=True)
imgs = {p.stem: p for p in IMG.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
keys = list(P)[:a.limit] if a.limit else list(P)
out, t0 = {}, time.time()
n_det = n_match = n_agree = n_agree4 = 0; diffs, ious, rerun_cnt = [], [], []
for st in keys:
    r = m.predict(source=str(imgs[st]), imgsz=1280, conf=0.001, iou=0.6, max_det=3000, verbose=False, device="cpu")[0]
    raw = CAP["raw"]; assert raw.shape[0] == 1 and raw.shape[1] == 4 + NC, raw.shape
    raw = raw[0].T                                                   # (anchors, 4 + 9)
    sc = raw[:, 4:]; top, tcls = sc.max(1)
    k = top >= KEEP_MIN; sc, tcls = sc[k], tcls[k]
    cb = ops.scale_boxes(CAP["img_shape"], ops.xywh2xyxy(raw[k, :4]), CAP["orig_shape"])
    det = torch.tensor([row[:4] for row in P[st]], dtype=torch.float32).reshape(-1, 4)
    dcls = torch.tensor([int(row[5]) for row in P[st]]); dsc = np.array([row[4] for row in P[st]])
    best_iou = torch.zeros(len(det)); best_j = torch.full((len(det),), -1, dtype=torch.long)
    for c in range(NC):
        di = (dcls == c).nonzero().flatten(); ci = (tcls == c).nonzero().flatten()
        if not len(di) or not len(ci): continue
        for s in range(0, len(di), 256):
            d = di[s:s+256]; iou = torchvision.ops.box_iou(det[d], cb[ci]); v, j = iou.max(1)
            best_iou[d] = v; best_j[d] = ci[j]
    rows = []
    for i, row in enumerate(P[st]):
        n_det += 1
        if best_iou[i] >= IOU_MIN:
            s9 = sc[best_j[i]].numpy().astype(np.float64); n_match += 1; ious.append(float(best_iou[i]))
            dd = abs(s9[int(row[5])] - dsc[i]); diffs.append(dd); n_agree += dd <= TOL; n_agree4 += dd <= 1e-4
            rows.append(list(row) + [round(float(x), 7) for x in s9])
        else:
            rows.append(list(row) + [None]*NC)
    out[st] = rows
    rerun_cnt.append((st, len(P[st]), len(r.boxes)))
    print(f"{st}: stored {len(P[st])} rerun-NMS {len(r.boxes)} raw-kept {int(k.sum())} letterbox {CAP['img_shape']} "
          f"orig {CAP['orig_shape'][:2]}  {time.time()-t0:.0f}s", flush=True)

diffs = np.array(diffs); ious = np.array(ious)
print(f"detections {n_det}  matched (IoU>={IOU_MIN}) {n_match} = {100*n_match/max(n_det,1):.2f} %")
print(f"own-class score within {TOL}: {n_agree} = {100*n_agree/max(n_match,1):.2f} % of matched; within 1e-4: "
      f"{100*n_agree4/max(n_match,1):.2f} %; |diff| median {np.median(diffs):.2e} p99 {np.quantile(diffs, .99):.2e} max {diffs.max():.2e}")
print(f"matched IoU: median {np.median(ious):.6f} min {ious.min():.4f};  images with the same NMS count as stored: "
      f"{sum(a_ == b_ for _, a_, b_ in rerun_cnt)}/{len(rerun_cnt)}")
json.dump(out, open(a.out, "w"))
print("CLS9_DONE", a.out, f"{time.time()-t0:.0f}s", flush=True)
