#!/usr/bin/env python3
"""Detector baseline trained on the KiCad renders (task 37, owner 2026-10-25 21:46): ultralytics YOLOv8l-P2 with EXACTLY the v22 recipe
(~/projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/args.yaml on the L40, ultralytics 8.4.37); only the
data (data/det_kicad_v1_1280: kicad_render/out_paste_v1, classes of linkdata/kicad_fptype_v3, split CrossClass_data_v2 ktrain / kval,
long side <= 1280), the number of epochs and close_mosaic (scaled to the epochs) change. One GPU: batch 8 (v22: 8 over 4 GPUs), nbs 64.
Selection as v22: best.pt = best val fitness (mAP50-95) on kval. Every epoch is saved (save_period 1).
Usage (cluster02, in an allocation): python tools/det/train_yolo_kicad.py --name NAME --epochs E [--close_mosaic C] [--fraction F] [--model YAML --weights PT]"""
import argparse, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"xrssd/crossclass"))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--close_mosaic", type=int, default=None, help="default: round(10 / 60 x epochs), v22's share, at least 1")
ap.add_argument("--data", default="data/det_kicad_v1_1280"); ap.add_argument("--weights", default="weights/yolov8l.pt")
ap.add_argument("--save_period", type=int, default=1, help="save epochN.pt every N epochs (owner 2026-10-26 14:10: 2 for new runs; the scoring is every 2nd epoch)")
ap.add_argument("--train", default="images/train", help="train images: a folder or a list file, relative to --data (e.g. train_drop_ge1000.txt)")
ap.add_argument("--model", default="yolov8l-p2.yaml", help="model yaml: an ultralytics name or a path, e.g. configs/det/yolo11l-p2.yaml (scale from the file name)")
ap.add_argument("--fraction", type=float, default=1.0, help="share of the train images (timing runs only)")
ap.add_argument("--workers", type=int, default=3); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="0")
ap.add_argument("--smoke", action="store_true", help="code test: 1 epoch, fraction 0.002, no val, no plots")
a = ap.parse_args()
from ultralytics import YOLO
D = (ROOT/a.data).resolve(); Y = ROOT/"runs/det_kicad"/f"{a.name}.data.yaml"; Y.parent.mkdir(parents=True, exist_ok=True)
Y.write_text(f"path: {D}\ntrain: {a.train}\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(["R", "C", "L", "J", "D", "SW", "Q", "IC", "Y"])))
if a.smoke: a.epochs, a.fraction = 1, 0.002
cm = a.close_mosaic if a.close_mosaic is not None else max(1, round(10 / 60 * a.epochs))
model = YOLO(str(ROOT/a.model) if (ROOT/a.model).exists() else a.model).load(str(ROOT/a.weights))
model.train(data=str(Y), project=str(ROOT/"runs/det_kicad"), name=a.name, exist_ok=False, epochs=a.epochs, close_mosaic=cm, fraction=a.fraction,
            # ---- v22 recipe (args.yaml), unchanged ----
            patience=20, batch=a.batch, imgsz=1280, save_period=a.save_period, workers=a.workers, seed=0, deterministic=True, amp=True, rect=False,
            multi_scale=0.0, cache=False, nbs=64, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.8, dfl=1.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            bgr=0.0, mosaic=1.0, mixup=0.1, cutmix=0.0, copy_paste=0.3, copy_paste_mode="flip", auto_augment="randaugment", erasing=0.4,
            conf=0.001, iou=0.6, device=a.device, plots=not a.smoke, val=not a.smoke)
