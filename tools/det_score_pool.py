#!/usr/bin/env python3
"""Pool of real detector class-score vectors per TRUE class (to simulate detector outputs on KiCad boards in training).

What: run the PCB-Dense v22 YOLOv8l-P2 (yolov8l_p2_notile/weights/best.pt) with the dump_all9.py predict setting
(imgsz 1280, conf 0.001, NMS iou 0.6, max_det 3000, one image per call, no TTA / tiling) on a labelled image set, keep the
9 raw sigmoid class scores of every box kept by NMS, match boxes with score >= 0.25 to the GT boxes (class-agnostic,
IoU >= 0.5, greedy by score, one detection per GT) and collect, per true class, [box_score, s0..s8] of the matched detection.
How the 9 scores are taken: a wrapper around ultralytics' non_max_suppression keeps a copy of the raw head output and
calls NMS with return_idxs=True, so each kept box is tied to its own anchor (exact, no box matching needed).
Sets (label class ids are mapped to the model ids 0..8 = Resistor, Capacitor, Inductor, Connector, Diode, Switch,
Transistor, IC, Oscillator):
  val9    gdino_fewshot/data/val9 (= tcassi val/optical_new, 9 large photos); labels are single-class (tcassi yaml
          "nc: 1, names: 0: IC") -> label 0 is mapped to IC (7). Only the IC pool is filled.
  v22val  PCB-Dense v22 dataset/images/val (515 images, the split the detector used for validation / best.pt selection)
  v22test PCB-Dense v22 dataset/images/test (402 images, held out); both v22 sets use dataset_v22.yaml, same names as the model
Output JSON: {"classes", "pool": {"0": [[box_score, s0..s8], ...], ...}, "gt_per_class", "matched_per_class", "source", ...}
Usage (CPU only, the L40 GPUs are shared):
  CUDA_VISIBLE_DEVICES="" nice -n 15 ~/venvs/pcbid/bin/python tools/det_score_pool.py --set val9 [--out F] [--threads 16]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torchvision
from ultralytics import YOLO
from ultralytics.utils import nms

H = Path.home()
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass"))
W = H/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/weights/best.pt"
V22 = H/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/dataset"
SETS = {"val9": (H/"projects/gdino_fewshot/data/val9/images", H/"projects/gdino_fewshot/data/val9/labels", {0: 7}),
        "v22val": (V22/"images/val", V22/"labels/val", {i: i for i in range(9)}),
        "v22test": (V22/"images/test", V22/"labels/test", {i: i for i in range(9)})}
NC, CONF_MATCH, IOU_MATCH = 9, 0.25, 0.5
SETTING = "ultralytics predict imgsz=1280 conf=0.001 iou=0.6 max_det=3000 (as gdino_fewshot/err9/dump_all9.py), CPU fp32"

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="val9", choices=SETS); ap.add_argument("--out", default="")
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--weights", default=str(W)); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
img_dir, lbl_dir, LMAP = SETS[a.set]; out_path = a.out or str(ROOT/f"data/det_score_pool_{a.set}.json")

CAP = {}
_orig_nms = nms.non_max_suppression
def _nms_keep_idx(prediction, *args, **kw):
    """Keep a copy of the raw head output (NMS edits it in place) and the anchor index of every kept box."""
    p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    CAP["raw"] = p.detach().float().cpu().clone()
    want = kw.pop("return_idxs", False)
    out, keep = _orig_nms(prediction, *args, return_idxs=True, **kw)
    CAP["keep"] = [k.cpu() for k in keep]
    return (out, keep) if want else out
nms.non_max_suppression = _nms_keep_idx

m = YOLO(a.weights); names = [m.names[i] for i in range(NC)]; print(names, flush=True)
imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
if a.limit: imgs = imgs[:a.limit]
pool = {str(c): [] for c in range(NC)}; gt_n = np.zeros(NC, int); mt_n = np.zeros(NC, int); t0 = time.time(); max_chk = 0.0
for ip in imgs:
    r = m.predict(source=str(ip), imgsz=1280, conf=0.001, iou=0.6, max_det=3000, verbose=False, device="cpu")[0]
    raw, keep = CAP["raw"][0], CAP["keep"][0].long().view(-1)        # raw: (4 + 9, anchors)
    s9 = raw[4:, keep].T.numpy().astype(np.float64)                   # (kept boxes, 9), same order as r.boxes
    conf = r.boxes.conf.numpy().astype(np.float64); cls = r.boxes.cls.numpy().astype(int); xyxy = r.boxes.xyxy.numpy()
    assert len(s9) == len(conf) and (len(conf) == 0 or (s9.argmax(1) == cls).all())
    if len(conf): max_chk = max(max_chk, float(np.abs(s9[np.arange(len(cls)), cls] - conf).max()))
    Hh, Ww = r.orig_shape
    lf = lbl_dir/f"{ip.stem}.txt"; g = np.loadtxt(lf, ndmin=2) if lf.exists() and lf.stat().st_size else np.zeros((0, 5))
    gcls = np.array([LMAP[int(c)] for c in g[:, 0]], int)
    gbox = np.c_[(g[:, 1]-g[:, 3]/2)*Ww, (g[:, 2]-g[:, 4]/2)*Hh, (g[:, 1]+g[:, 3]/2)*Ww, (g[:, 2]+g[:, 4]/2)*Hh]
    np.add.at(gt_n, gcls, 1)
    sel = np.where(conf >= CONF_MATCH)[0]; sel = sel[np.argsort(-conf[sel], kind="stable")]
    used = np.zeros(len(gbox), bool); nm = 0
    if len(sel) and len(gbox):
        iou = torchvision.ops.box_iou(torch.tensor(xyxy[sel], dtype=torch.float64), torch.tensor(gbox, dtype=torch.float64)).numpy()
        for row, d in enumerate(sel):                                 # greedy by detection score, one detection per GT
            v = np.where(used, -1.0, iou[row]); j = int(v.argmax())
            if v[j] >= IOU_MATCH:
                used[j] = True; mt_n[gcls[j]] += 1; nm += 1
                pool[str(gcls[j])].append([round(float(conf[d]), 6)] + [round(float(x), 6) for x in s9[d]])
    print(f"{ip.stem}: {Ww}x{Hh} gt {len(gbox)} det>=0.25 {len(sel)} matched {nm}  {time.time()-t0:.0f}s", flush=True)

acc = {}
for c in range(NC):
    P = np.array(pool[str(c)]).reshape(-1, 1 + NC)
    acc[names[c]] = round(float((P[:, 1:].argmax(1) == c).mean()), 4) if len(P) else None
res = {"classes": names, "pool": pool, "gt_per_class": {names[c]: int(gt_n[c]) for c in range(NC)},
       "matched_per_class": {names[c]: int(mt_n[c]) for c in range(NC)}, "argmax_acc_per_class": acc,
       "source": f"{a.weights}; {SETTING}; set {a.set} = {img_dir} ({len(imgs)} images), labels {lbl_dir}, label map {LMAP}; "
                 f"match: box score >= {CONF_MATCH}, IoU >= {IOU_MATCH}, class-agnostic greedy by score, one detection per GT"}
json.dump(res, open(out_path, "w"))
print("gt", res["gt_per_class"]); print("matched", res["matched_per_class"]); print("argmax acc", acc)
print(f"own-score check (kept-box conf vs raw score of its class): max |diff| {max_chk:.2e}")
print("POOL_DONE", out_path, f"{time.time()-t0:.0f}s", flush=True)
