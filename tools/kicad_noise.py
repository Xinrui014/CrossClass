#!/usr/bin/env python3
"""Noise boxes for linker training (2026-10-16). On WACV photos about half of the OCR candidate boxes are not labels; the KiCad training
data had none. This script takes, per KiCad side of CrossClass_data_v2, the places where a photo's OCR finds text-like marks that are not
designator labels, straight from the design file:
  print: silkscreen text that is not a designator label (board text such as "GND" / "3V3" / names, footprint user texts, visible Value
         fields, visible Reference fields that do not look like designators)
  via:   vias on the side's copper
  pad:   pads of non-component footprints (test points, mounting holes, fiducials, logos, ...)
Left out: boxes >= 50 % inside a part body (at inference the OCR boxes inside detector boxes are dropped by rule) and boxes that overlap
a label. At most max(20, 2 x labels) boxes per side: all print first, then a random sample of vias / pads.
Frame: the side's kicad.json frame (mm, relative to the union of its boxes, bottom mirrored), found as in kicad_render_texts.py from the
absolute reference-text boxes; pixels: a linear fit kicad.json text boxes -> render json text boxes. Only sides with an aligned render
(offset_spread_mm <= 0.05) and text features. Writes linkdata/kicad_noise/<side>.json {id, boxes (mm), px, kind, text}.
Usage: /usr/bin/python3 kicad_noise.py --worker I --nworkers N [--limit K]   (CROSSCLASS project root)"""
import argparse, json, os, random, statistics, sys, time
from pathlib import Path
import numpy as np
import pcbnew
sys.path.insert(0, str(Path(__file__).parent))
from kicad_links import DES, is_component, box_mm, inside_frac
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); OUT = ROOT/"linkdata/kicad_noise"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
K = json.load(open(ROOT/"linkdata/CrossClass_data_v2/kicad.json"))
if a.limit: K = sorted(K, key=lambda b: b["id"])[:a.limit]
by_key = {}
for b in K: by_key.setdefault(b["id"].rsplit("_", 1)[0], []).append(b)

def shown(t):
    try: return t.GetShownText(False)
    except Exception: return t.GetText()

def fit(rel, px):
    """Per axis px = c + s * mm from matching box coordinates (x0, x1 / y0, y1 of every label)."""
    R, P = np.array(rel, float), np.array(px, float)
    cx = np.polyfit(np.r_[R[:, 0], R[:, 2]], np.r_[P[:, 0], P[:, 2]], 1); cy = np.polyfit(np.r_[R[:, 1], R[:, 3]], np.r_[P[:, 1], P[:, 3]], 1)
    return lambda b: [round(cx[1] + cx[0]*b[0], 1), round(cy[1] + cy[0]*b[1], 1), round(cx[1] + cx[0]*b[2], 1), round(cy[1] + cy[0]*b[3], 1)]

for n, (key, sides) in enumerate(sorted(by_key.items())):
    if n % a.nworkers != a.worker: continue
    todo = []
    for s in sides:
        rj = ROOT/f"linkdata/kicad_render/{s['id']}.json"
        if (OUT/f"{s['id']}.json").exists() or not rj.exists() or not (ROOT/f"linkdata/kicad_textfeat/{s['id']}.npy").exists(): continue
        r = json.load(open(rj))
        if r.get("offset_spread_mm", 1) <= 0.05 and len(r["texts"]) == len(s["texts"]): todo.append((s, r))
    if not todo: continue
    t0 = time.time()
    try:
        brd = pcbnew.LoadBoard(str(ROOT/f"kicad_scan/boards/{key}.kicad_pcb"))
        for s, r in todo:
            bottom = s["side"] == "bottom"; cu, silk = (pcbnew.B_Cu, pcbnew.B_SilkS) if bottom else (pcbnew.F_Cu, pcbnew.F_SilkS)
            mir = (lambda b: [-b[2], b[1], -b[0], b[3]]) if bottom else (lambda b: b)
            ref, cand = {}, []                                   # cand: (kind, absolute mirrored mm box, text)
            for fp in brd.GetFootprints():
                rf = fp.Reference(); des = fp.GetReference()
                if rf.GetLayer() == silk and rf.IsVisible():
                    ref.setdefault(des, mir(box_mm(rf.GetBoundingBox())))
                    if not DES.match(des): cand.append(("print", mir(box_mm(rf.GetBoundingBox())), des))
                v = fp.Value()
                if v.GetLayer() == silk and v.IsVisible(): cand.append(("print", mir(box_mm(v.GetBoundingBox())), shown(v)))
                for g in fp.GraphicalItems():
                    if g.Type() == pcbnew.PCB_TEXT_T and g.GetLayer() == silk and g.IsVisible(): cand.append(("print", mir(box_mm(g.GetBoundingBox())), shown(g)))
                if not is_component(fp, des):
                    for p in fp.Pads():
                        if p.IsOnLayer(cu): cand.append(("pad", mir(box_mm(p.GetBoundingBox())), ""))
            for d in brd.GetDrawings():
                if d.Type() in (pcbnew.PCB_TEXT_T, pcbnew.PCB_TEXTBOX_T) and d.GetLayer() == silk: cand.append(("print", mir(box_mm(d.GetBoundingBox())), shown(d)))
            for t in brd.GetTracks():
                if t.Type() == pcbnew.PCB_VIA_T and t.IsOnLayer(cu): cand.append(("via", mir(box_mm(t.GetBoundingBox())), ""))
            dx = [ref[t["text"]][0] - t["box"][0] for t in s["texts"] if t["text"] in ref]
            dy = [ref[t["text"]][1] - t["box"][1] for t in s["texts"] if t["text"] in ref]
            if not dx: continue
            ox, oy = statistics.median(dx), statistics.median(dy); P = fit([t["box"] for t in s["texts"]], r["texts"])
            bodies = [p["box"] for p in s["parts"]]; labels = [t["box"] for t in s["texts"]]; keep = []
            for kind, b, txt in cand:
                b = [b[0]-ox, b[1]-oy, b[2]-ox, b[3]-oy]
                if b[2]-b[0] <= 0 or b[3]-b[1] <= 0: continue
                if max((inside_frac(b, q) for q in bodies), default=0) >= 0.5: continue
                if any(inside_frac(b, q) > 0.2 or inside_frac(q, b) > 0.2 for q in labels): continue
                px = P(b)
                if px[2] <= 0 or px[3] <= 0 or px[0] >= r["W"] or px[1] >= r["H"]: continue
                keep.append((kind, [round(v, 3) for v in b], px, txt))
            rng = random.Random(s["id"]); cap = max(20, 2*len(s["texts"]))
            pr = [k for k in keep if k[0] == "print"]; ot = [k for k in keep if k[0] != "print"]; rng.shuffle(pr); rng.shuffle(ot)
            sel = (pr + ot)[:cap]
            json.dump(dict(id=s["id"], boxes=[k[1] for k in sel], px=[k[2] for k in sel], kind=[k[0] for k in sel], text=[k[3] for k in sel],
                           n_cand=dict(print=len(pr), other=len(ot))), open(OUT/f"{s['id']}.json", "w"))
        print(f"{key} ok {len(todo)} sides {time.time()-t0:.1f}s", flush=True)
    except Exception as ex:
        print(f"{key} FAILED {repr(ex)[:160]}", flush=True)
