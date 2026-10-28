#!/usr/bin/env python3
"""Part IMAGE features for LinkFormer part tokens (owner 2026-10-23: detector RoI features, E0 = trunk + part image feature).
Extractor = the one of tools/partfeat_check.py (tasks 2 / 3): PCB-Dense v22 YOLOv8l-P2 (yolov8l_p2_notile/weights/best.pt), ultralytics
predict imgsz 1280 (det_score_pool.py setting), the 4 Detect-head inputs P2..P5 (strides 4 / 8 / 16 / 32, 128 / 256 / 512 / 512 ch),
torchvision roi_align 4 x 4 (aligned, 2 samples) at each part box mapped into the letterboxed input, averaged per level, the levels
concatenated -> 1,408 values per part. Self-check per image: the kept detections' boxes mapped with the same letterbox transform equal
the head's own raw boxes (both clipped to the image), else the image fails.
  photo   -> data/partfeat_photo/{wacv_annot, wacv_det, pcb21_gt, pcb21_det}.npz: one array (n parts, 1408) per board id, in the parts
             order of the boards file (wacv_clean_v1.json, wacv_e2e/e2e_boards_all.json, 21pcb/link_eval/boards.json,
             21pcb/e2e/e2e_boards_all.json); one detector pass per image serves both box sets of that image. + run_meta sidecar.
  render --src kicad_render/out_paste_v1 [--out data/partfeat_render/out_paste_v1] [--limit N] [--ids FILE]
          -> <out>/<side id>.npz {feat (n kicad parts, 1408) float16, ok (n) bool} in the side's kicad.json order (part field "i" of the
             kicad_render_3d.py json); ok = in_image and status != no_image; parts not in the json get ok False. Resumable.
  boxes --boards F --out F.npz [--pcb21]  -> {board id: (n parts, 1408)} for the part boxes of any boards JSON (e.g. raw WACV wacv.json)
(env ~/venvs/pcbid, CPU: CUDA_VISIBLE_DEVICES="" taskset -c 0-63 nice -n 15; --device cuda:N for a GPU)"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
H = Path.home(); ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass"))
W = Path(os.environ.get("ROI_WEIGHTS", H/"projects/PCB_structure_layout/V2.2_DINO_detection/stage1_yolo/yolov8l_p2_notile/weights/best.pt"))
RAWW = Path(os.environ.get("ROI_WACV_PHOTOS", H/"projects/GLIGEN/igligen/dataset/pcb_wacv_2019"))      # <id>/<id>.jpg or .png
PCB21 = Path(os.environ.get("ROI_PCB21_PHOTOS", ROOT/"data/21pcb/images"))                           # <id>.jpg
OUTP = Path(os.environ.get("ROI_PHOTO_OUT", ROOT/"data/partfeat_photo"))   # env defaults = L40 paths; intern02 sets ROI_* (task 9)

class RoI:
    """Detector neck RoI features at given boxes (pixel x0, y0, x1, y1 of the original image)."""
    def __init__(self, device="cpu", threads=16):
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import nms
        torch.set_num_threads(threads); self.dev = device; self.CAP = CAP = {}; _orig = nms.non_max_suppression
        def _nms(prediction, *args, **kw):            # as det_score_pool.py: raw head output + anchor index of every kept box
            p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            CAP["raw"] = p.detach().float().cpu().clone(); want = kw.pop("return_idxs", False)
            out, keep = _orig(prediction, *args, return_idxs=True, **kw); CAP["keep"] = [k.cpu() for k in keep]
            return (out, keep) if want else out
        nms.non_max_suppression = _nms
        self.m = YOLO(str(W)); net = self.m.model; head = net.model[-1]; self.strides = [int(s) for s in head.stride.tolist()]
        net.model[0].register_forward_pre_hook(lambda mod, inp: CAP.__setitem__("x", tuple(inp[0].shape)))
        head.register_forward_pre_hook(lambda mod, inp: CAP.__setitem__("maps", [f.detach().float() for f in inp[0]]))
    def __call__(self, ip, box_sets):
        """ip: image path; box_sets: list of (n, 4) arrays -> list of (n, 1408) float32 arrays, letterbox self-check error (px)."""
        import torch, torchvision
        r = self.m.predict(source=str(ip), imgsz=1280, conf=0.001, iou=0.6, max_det=3000, verbose=False, device=self.dev)[0]
        CAP = self.CAP; Hh, Ww = r.orig_shape; raw, keep = CAP["raw"][0], CAP["keep"][0].long().view(-1)
        rr = min(1280 / Hh, 1280 / Ww); nw, nh = int(round(Ww * rr)), int(round(Hh * rr)); hin, win = CAP["x"][2], CAP["x"][3]
        left, top = int(round((win - nw) / 2 - 0.1)), int(round((hin - nh) / 2 - 0.1)); tf = lambda b: b * rr + np.array([left, top, left, top])
        err = 0.0
        if len(keep):
            rb = raw[:4, keep].T.numpy(); rb = np.c_[rb[:, :2] - rb[:, 2:]/2, rb[:, :2] + rb[:, 2:]/2]
            rb = np.clip(rb, [left, top, left, top], [left + nw, top + nh, left + nw, top + nh])
            err = float(np.abs(tf(r.boxes.xyxy.cpu().numpy()) - rb).max()); assert err < 2.0, f"{ip}: letterbox transform error {err:.2f} px"
        maps = [f.to("cpu") for f in CAP["maps"]]; out = []
        for b in box_sets:
            b = np.asarray(b, float).reshape(-1, 4)
            if not len(b): out.append(np.zeros((0, sum(f.shape[1] for f in maps)), np.float32)); continue
            bi = torch.tensor(tf(b), dtype=torch.float32)
            out.append(torch.cat([torchvision.ops.roi_align(f, [bi], output_size=4, spatial_scale=1.0/s, sampling_ratio=2, aligned=True).mean((2, 3))
                                  for f, s in zip(maps, self.strides)], 1).numpy())
        return out, err

def photo(threads=16):
    """WACV (annotated clean boxes + e2e detector boxes) and 21PCB (link_eval GT boxes + e2e detector boxes)."""
    from PIL import Image
    sys.path.insert(0, str(ROOT/"tools")); import run_meta as RM
    L = lambda p: {b["id"]: b for b in json.load(open(p))}
    sets = {"wacv_annot": L(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json"), "wacv_det": L(ROOT/"data/wacv_e2e/e2e_boards_all.json"),
            "pcb21_gt": L(ROOT/"data/21pcb/link_eval/boards.json"), "pcb21_det": L(ROOT/"data/21pcb/e2e/e2e_boards_all.json")}
    img = lambda st, wacv: (RAWW/st/f"{st}.jpg" if (RAWW/st/f"{st}.jpg").exists() else RAWW/st/f"{st}.png") if wacv else PCB21/f"{st}.jpg"
    ex = RoI(threads=threads); OUTP.mkdir(parents=True, exist_ok=True); res = {k: {} for k in sets}; info = dict(images=0, seconds=0.0, max_letterbox_err_px=0.0)
    for pair, wacv in ((("wacv_annot", "wacv_det"), True), (("pcb21_gt", "pcb21_det"), False)):
        for st in sorted(set(sets[pair[0]]) | set(sets[pair[1]])):
            ip = img(st, wacv); Ww, Hh = Image.open(ip).size; ks = [k for k in pair if st in sets[k]]
            for k in ks:
                b = sets[k][st]
                if "W" in b: assert (b["W"], b["H"]) == (Ww, Hh), (k, st, b["W"], b["H"], Ww, Hh)
            t0 = time.time(); feats, err = ex(ip, [[p["box"] for p in sets[k][st]["parts"]] for k in ks])
            for k, f in zip(ks, feats): res[k][st] = f.astype(np.float32)
            info["images"] += 1; info["seconds"] += time.time() - t0; info["max_letterbox_err_px"] = max(info["max_letterbox_err_px"], err)
            print(f"{st}: {Ww}x{Hh} " + " ".join(f"{k} {len(f)}" for k, f in zip(ks, feats)) + f"  {time.time()-t0:.1f}s  err {err:.2f}px", flush=True)
    for k, d in res.items():
        out = OUTP/f"{k}.npz"; np.savez(out, **d); n = sum(len(v) for v in d.values())
        RM.sidecar(out, [], None, {"boards": {"wacv_annot": ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json", "wacv_det": ROOT/"data/wacv_e2e/e2e_boards_all.json",
                   "pcb21_gt": ROOT/"data/21pcb/link_eval/boards.json", "pcb21_det": ROOT/"data/21pcb/e2e/e2e_boards_all.json"}[k], "detector": W},
                   decode="none (features)", none_bias=0, boards=len(d), parts=n, feature="YOLOv8l-P2 P2..P5 roi_align 4x4 mean, 1408-d")
        print(f"-> {out}: {len(d)} boards, {n} parts", flush=True)
    print(json.dumps(info), flush=True)

def boxes(boards, out, wacv=True, threads=16):
    """Part boxes of any boards JSON (id, W, H, parts[].box) on the WACV photos (or 21PCB with wacv False) -> out npz {board id: (n parts,
    1408)} in the parts order, + run_meta sidecar (2026-10-23: the raw WACV boxes of wacv.json, dev3 / all45 evals)."""
    from PIL import Image
    sys.path.insert(0, str(ROOT/"tools")); import run_meta as RM
    B = json.load(open(boards)); ex = RoI(threads=threads); res = {}; info = dict(images=0, seconds=0.0, max_letterbox_err_px=0.0)
    for b in B:
        st = b["id"]; ip = (RAWW/st/f"{st}.jpg" if (RAWW/st/f"{st}.jpg").exists() else RAWW/st/f"{st}.png") if wacv else PCB21/f"{st}.jpg"
        Ww, Hh = Image.open(ip).size
        if "W" in b: assert (b["W"], b["H"]) == (Ww, Hh), (st, b["W"], b["H"], Ww, Hh)
        t0 = time.time(); feats, err = ex(ip, [[p["box"] for p in b["parts"]]]); res[st] = feats[0].astype(np.float32)
        info["images"] += 1; info["seconds"] += time.time() - t0; info["max_letterbox_err_px"] = max(info["max_letterbox_err_px"], err)
        print(f"{st}: {Ww}x{Hh} parts {len(res[st])}  {time.time()-t0:.1f}s  err {err:.2f}px", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True); np.savez(out, **res); n = sum(len(v) for v in res.values())
    RM.sidecar(out, [], None, {"boards": Path(boards), "detector": W}, decode="none (features)", none_bias=0, boards=len(res), parts=n,
               feature="YOLOv8l-P2 P2..P5 roi_align 4x4 mean, 1408-d")
    print(f"-> {out}: {len(res)} boards, {n} parts", flush=True); print(json.dumps(info), flush=True)

def render(src, out, limit=0, ids=None, device="cpu", threads=16):
    """3D-render sides (kicad_render_3d.py json + jpg): RoI features at the part boxes, in kicad.json order, ok flags."""
    src, out = Path(src), Path(out); out.mkdir(parents=True, exist_ok=True)
    sides = json.load(open(ids)) if ids else sorted(f.stem for f in src.glob("*.json") if f.name not in ("sides.json", "bad_sides.json"))
    ex = RoI(device=device, threads=threads); t_all = time.time(); done = 0
    for sid in sides[: limit or None]:
        o = out/f"{sid}.npz"
        if o.exists() or not (src/f"{sid}.jpg").exists(): continue
        s = json.load(open(src/f"{sid}.json"))
        if s.get("status") != "ok": continue
        P = s["parts"]; n = max([p["i"] for p in P], default=-1) + 1; feat = np.zeros((n, 1408), np.float16); ok = np.zeros(n, bool)
        use = [p for p in P if p.get("in_image") and p.get("status") != "no_image"]
        t0 = time.time(); (f,), err = ex(src/f"{sid}.jpg", [[p["box"] for p in use]])
        for p, v in zip(use, f): feat[p["i"]] = v; ok[p["i"]] = True
        np.savez(o, feat=feat, ok=ok); done += 1
        print(f"{done} {sid}: parts {n}, with image {int(ok.sum())}  {time.time()-t0:.1f}s  err {err:.2f}px  ({time.time()-t_all:.0f}s)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["photo", "render", "boxes"]); ap.add_argument("--boards", default=""); ap.add_argument("--pcb21", action="store_true"); ap.add_argument("--src", default=str(ROOT/"kicad_render/out_paste_v1"))
    ap.add_argument("--out", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ids", default="")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=16); a = ap.parse_args()
    if a.cmd == "photo": photo(a.threads)
    elif a.cmd == "boxes": boxes(a.boards, a.out, not a.pcb21, a.threads)
    else: render(a.src, a.out or str(ROOT/"data/partfeat_render"/Path(a.src).name), a.limit, a.ids or None, a.device, a.threads)
