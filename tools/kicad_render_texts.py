#!/usr/bin/env python3
"""Render the KiCad board sides of CrossClass_data_v2 (2026-10-15) so text (and later part) image patches can be cut for training.
One `kicad-cli pcb render` per side (orthographic, --quality basic, transparent background, the side's own view; bottom sides come out
mirrored, as kicad.json stores them), WITHOUT 3D models: board, solder mask, pads, silkscreen; no component bodies. Scale: a label height
drawn from WACV's designator heights (assoc/wacv_profile.json, px) / the side's median text height in kicad.json (mm); long side <= 4096 px.
Mapping kicad.json (mm, relative to the union of the side's boxes) -> pixels: the offset of that union from the absolute pcbnew reference-text
boxes (median over the side's texts), and mm -> px from the opaque board area in the render (= the board edge bbox).
Writes linkdata/kicad_render/<side id>.jpg (q 92) and <side id>.json {W, H, ppm, offset_spread_mm, texts: [[x0, y0, x1, y1] px per
kicad.json text], parts: [...]}; log lines "<id> ok <W>x<H> <seconds>" or "<id> FAILED ...". A side whose offset_spread_mm is above ~0.05
did not align (e.g. duplicate reference names) and should be left out. Test 2026-10-15: 49 sides in ~15 s with 8 workers.
Usage: kicad_render_texts.py --worker I --nworkers N [--limit K]   (system python3 with pcbnew; CROSSCLASS project root)"""
import argparse, json, os, random, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import pcbnew
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); OUT = ROOT/"linkdata/kicad_render"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--maxside", type=int, default=4096); a = ap.parse_args()
WLAB = [r["label_px"] for r in json.load(open(Path.home()/"projects/gdino_fewshot/assoc/wacv_profile.json")) if r["label_px"] == r["label_px"]]
K = json.load(open(ROOT/"linkdata/CrossClass_data_v2/kicad.json"))
if a.limit: K = sorted(K, key=lambda b: b["id"])[:a.limit]
by_key = {}
for b in K: by_key.setdefault(b["id"].rsplit("_", 1)[0], []).append(b)
def strip_models(txt):
    out, i = [], 0
    while True:
        j = txt.find("(model ", i)
        if j < 0: out.append(txt[i:]); break
        out.append(txt[i:j]); d = 0; k = j
        while True:
            if txt[k] == "(": d += 1
            elif txt[k] == ")":
                d -= 1
                if d == 0: break
            k += 1
        i = k+1
    return "".join(out)
mm = lambda bb: [bb.GetX()/1e6, bb.GetY()/1e6, (bb.GetX()+bb.GetWidth())/1e6, (bb.GetY()+bb.GetHeight())/1e6]
for n, (key, sides) in enumerate(sorted(by_key.items())):
    if n % a.nworkers != a.worker or all((OUT/f"{s['id']}.json").exists() for s in sides): continue
    t0 = time.time(); f = ROOT/f"kicad_scan/boards/{key}.kicad_pcb"; tmp = Path(f"/tmp/xinrui_kr_{a.worker}")
    try:
        brd = pcbnew.LoadBoard(str(f)); e = mm(brd.GetBoardEdgesBoundingBox()); bw, bh = e[2]-e[0], e[3]-e[1]
        if bw <= 0 or bh <= 0: print(f"{key} SKIPPED no board outline", flush=True); continue
        tmp.mkdir(exist_ok=True); fm = tmp/"nomodel.kicad_pcb"; fm.write_text(strip_models(f.read_text()))
        for s in sides:
            if (OUT/f"{s['id']}.json").exists(): continue
            bottom = s["side"] == "bottom"; silk = pcbnew.B_SilkS if bottom else pcbnew.F_SilkS
            hs = [min(t["box"][2]-t["box"][0], t["box"][3]-t["box"][1]) for t in s["texts"]]
            ppm = random.Random(s["id"]).choice(WLAB)/max(statistics.median(hs), 0.3); ppm = min(ppm, a.maxside/max(bw, bh))
            W, H = max(64, int(round(bw*ppm*1.04))), max(64, int(round(bh*ppm*1.04))); png = tmp/"r.png"
            subprocess.run(["kicad-cli", "pcb", "render", "-o", str(png), "-w", str(W), "-h", str(H), "--side", "bottom" if bottom else "top",
                            "--background", "transparent", "--quality", "basic", str(fm)], capture_output=True, text=True, timeout=1800)
            A = np.asarray(Image.open(png)); ys, xs = np.where(A[:, :, 3] > 0)
            mx0, mx1, my0, my1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1; sx, sy = (mx1-mx0)/bw, (my1-my0)/bh
            # absolute (bottom: mirrored) mm of the side's reference texts -> offset of kicad.json's relative frame
            ref = {}
            for fp in brd.GetFootprints():
                r = fp.Reference()
                if r.GetLayer() == silk and r.IsVisible(): ref.setdefault(fp.GetReference(), mm(r.GetBoundingBox()))
            dx, dy = [], []
            for t in s["texts"]:
                if t["text"] not in ref: continue
                x0, y0, x1, y1 = ref[t["text"]]; ax0 = -x1 if bottom else x0
                dx.append(ax0 - t["box"][0]); dy.append(y0 - t["box"][1])
            ox, oy = statistics.median(dx), statistics.median(dy)
            ex0 = -e[2] if bottom else e[0]                                     # left edge of the (mirrored) board in the same frame
            P = lambda b: [round(mx0 + (b[0]+ox-ex0)*sx, 1), round(my0 + (b[1]+oy-e[1])*sy, 1), round(mx0 + (b[2]+ox-ex0)*sx, 1), round(my0 + (b[3]+oy-e[1])*sy, 1)]
            rgb = Image.alpha_composite(Image.new("RGBA", (A.shape[1], A.shape[0]), (0, 0, 0, 255)), Image.fromarray(A)).convert("RGB")
            rgb.save(OUT/f"{s['id']}.jpg", quality=92)
            json.dump(dict(id=s["id"], W=A.shape[1], H=A.shape[0], ppm=round((sx+sy)/2, 3), offset_spread_mm=round(float(np.std(dx) + np.std(dy)), 4),
                           texts=[P(t["box"]) for t in s["texts"]], parts=[P(p["box"]) for p in s["parts"]]), open(OUT/f"{s['id']}.json", "w"))
            print(f"{s['id']} ok {A.shape[1]}x{A.shape[0]} {time.time()-t0:.1f}s", flush=True); t0 = time.time()
    except Exception as ex:
        print(f"{key} FAILED {repr(ex)[:160]}", flush=True)
