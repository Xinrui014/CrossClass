#!/usr/bin/env python3
"""Detector baseline trained on the KiCad renders (task 37, owner 2026-10-25 21:46): ultralytics YOLOv8l-P2 with EXACTLY the v22 recipe
(~/projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/args.yaml on the L40, ultralytics 8.4.37); only the
data (data/det_kicad_v1_1280: kicad_render/out_paste_v1, classes of linkdata/kicad_fptype_v3, split CrossClass_data_v2 ktrain / kval,
long side <= 1280), the number of epochs and close_mosaic (scaled to the epochs) change. One GPU: batch 8 (v22: 8 over 4 GPUs), nbs 64.
Selection as v22: best.pt = best val fitness (mAP50-95) on kval. Every epoch is saved (save_period 1).
v2 (2026-10-26, cros-det) = train_yolo_kicad.py + --val, for the Real+Syn detector (data/det_realsyn_v1: list files with KiCad + v2.2 real
images); all defaults unchanged (val images/val), so a KiCad-only call trains as before. A copy, because a running step uses v1.
v3 (2026-10-28, cros-det; main 16:12, owner "fix the code first, then re-run"): = v2 + the FP32 trainer tools/det/fp32_trainer.py
(val during training in fp32, EMA saved in fp32, final strip / val in fp32; the train steps keep AMP as before; data, epochs, imgsz,
lr, aug, batch unchanged). best.pt = best fitness (mAP50-95) on the run's own val (--val). --smoke = the code test of that fix:
2 epochs, val on, save every epoch, no plots (use a small --train list). DDP (--device 0,1): the batch is the TOTAL batch
(ultralytics splits it per GPU); ultralytics has no SyncBatchNorm (BN statistics per GPU).
Usage (cluster02, in an allocation): python tools/det/train_yolo_kicad_v3.py --name NAME --epochs E [--close_mosaic C] [--fraction F] [--model YAML --weights PT]
  [--data DIR --train LIST --val LIST]"""
import argparse, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"xrssd/crossclass"))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--close_mosaic", type=int, default=None, help="default: round(10 / 60 x epochs), v22's share, at least 1")
ap.add_argument("--data", default="data/det_kicad_v1_1280"); ap.add_argument("--weights", default="weights/yolov8l.pt")
ap.add_argument("--save_period", type=int, default=1, help="save epochN.pt every N epochs (owner 2026-10-26 14:10: 2 for new runs; the scoring is every 2nd epoch)")
ap.add_argument("--train", default="images/train", help="train images: a folder or a list file, relative to --data (e.g. train_drop_ge1000.txt)")
ap.add_argument("--val", default="images/val", help="val images: a folder or a list file, relative to --data (best.pt = best fitness on it)")
ap.add_argument("--model", default="yolov8l-p2.yaml", help="model yaml: an ultralytics name or a path, e.g. configs/det/yolo11l-p2.yaml (scale from the file name)")
ap.add_argument("--fraction", type=float, default=1.0, help="share of the train images (timing runs only)")
ap.add_argument("--workers", type=int, default=3); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="0")
ap.add_argument("--smoke", action="store_true", help="code test of the fp32 fix: 2 epochs, val on, save_period 1, no plots (with a small --train list)")
ap.add_argument("--imgsz", type=int, default=1280, help="1280 = the recipe; other values only for the CPU code test")
a = ap.parse_args()
import hashlib, sys
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))                       # fp32_trainer: here and in the DDP subprocess
os.environ["PYTHONPATH"] = str(HERE) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
from ultralytics import YOLO
from fp32_trainer import FP32DetectionTrainer
print("train_yolo_kicad_v3", {f: hashlib.md5((HERE/f).read_bytes()).hexdigest()[:8] for f in ("train_yolo_kicad_v3.py", "fp32_trainer.py")},
      "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "DET_SMOKE_INFLATE_EPOCH", os.environ.get("DET_SMOKE_INFLATE_EPOCH"), flush=True)
D = (ROOT/a.data).resolve(); Y = ROOT/"runs/det_kicad"/f"{a.name}.data.yaml"; Y.parent.mkdir(parents=True, exist_ok=True)
Y.write_text(f"path: {D}\ntrain: {a.train}\nval: {a.val}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(["R", "C", "L", "J", "D", "SW", "Q", "IC", "Y"])))
if a.smoke: a.epochs, a.save_period = 2, 1
cm = a.close_mosaic if a.close_mosaic is not None else max(1, round(10 / 60 * a.epochs))
model = YOLO(str(ROOT/a.model) if (ROOT/a.model).exists() else a.model).load(str(ROOT/a.weights))
model.train(trainer=FP32DetectionTrainer, data=str(Y), project=str(ROOT/"runs/det_kicad"), name=a.name, exist_ok=False, epochs=a.epochs, close_mosaic=cm, fraction=a.fraction,
            # ---- v22 recipe (args.yaml), unchanged ----
            patience=20, batch=a.batch, imgsz=a.imgsz, save_period=a.save_period, workers=a.workers, seed=0, deterministic=True, amp=True, rect=False,
            multi_scale=0.0, cache=False, nbs=64, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.8, dfl=1.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            bgr=0.0, mosaic=1.0, mixup=0.1, cutmix=0.0, copy_paste=0.3, copy_paste_mode="flip", auto_augment="randaugment", erasing=0.4,
            conf=0.001, iou=0.6, device=a.device, plots=not a.smoke, val=True)
