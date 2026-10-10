#!/usr/bin/env python3
"""Stage 2 of the KiCad link set: .kicad_pcb -> boards in the linking format of linkdata/wacv.json, one record per board side.
- Parts: footprints on the side whose designator prefix maps to one of the 9 classes (table of kicad_render/render_all.py). Box = the
  physical body: bounding box of the pads and the Fab-layer shapes (fallback: footprint box without text). Footprints with no class
  (test points, holes, fiducials, logos) are not parts; their visible labels stay as texts that name no part, like WACV's test points.
- Texts: visible Reference fields on the side's silkscreen whose string looks like a designator, minus labels a photo would not show:
  >= 50 % of the label box inside any footprint body (printed under a component).
- Excluded: OLIMEX/*, Duet3D/*, BehrensG/ARTY_Z7 (possible overlap with WACV OlimexRed / DuetWIFI / Arty).
- Bottom sides are mirrored in x, as a photo of the back would be. A side is kept with >= 10 linkable parts. Units: mm.
Output linkdata/kicad_all.json + linkdata/kicad_all_stats.json.
Usage: /usr/bin/python3 kicad_links.py [workers]   (system python with KiCad 9 pcbnew)"""
import json, re, sys, os
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); IN = ROOT/"kicad_boards"; OUT = ROOT/"linkdata"
DES = re.compile(r"^[A-Z]{1,4}\d{1,4}[A-Z]?$")
# boards that might share layout or design style with a WACV board (WACV OlimexRed, Arty, DuetWIFI): kept out of training
EXCLUDE = re.compile(r"^(OLIMEX/|Duet3D/|BehrensG/ARTY_Z7)", re.I)
PFX = [("RN", 0), ("R", 0), ("FB", 2), ("L", 2), ("C", 1), ("LED", 4), ("D", 4), ("SW", 5), ("S", 5), ("Q", 6), ("IC", 7), ("U", 7), ("Y", 8), ("X", 8), ("J", 3), ("P", 3)]
def pcls(ref):
    m = re.match(r"([A-Z]+)\d", ref); return next((c for p, c in PFX if m and m.group(1) == p), -1)

def box_mm(bb): return [bb.GetX()/1e6, bb.GetY()/1e6, (bb.GetX()+bb.GetWidth())/1e6, (bb.GetY()+bb.GetHeight())/1e6]
def union(bs): return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
def inside_frac(t, b):
    iw = max(0.0, min(t[2], b[2])-max(t[0], b[0])); ih = max(0.0, min(t[3], b[3])-max(t[1], b[1]))
    return iw*ih/max((t[2]-t[0])*(t[3]-t[1]), 1e-9)

def extract(entry):
    import pcbnew
    key, name = entry["key"], entry["name"]; out = []; st = dict(sides_kept=0, hidden_labels=0)
    if EXCLUDE.search(name): return out, dict(excluded=1)
    try:
        brd = pcbnew.LoadBoard(str(IN/f"{key}.kicad_pcb"))
    except Exception:
        brd = None
    if brd is None: return out, dict(failed=1)
    for side, cu, silk, fab in (("top", pcbnew.F_Cu, pcbnew.F_SilkS, pcbnew.F_Fab), ("bottom", pcbnew.B_Cu, pcbnew.B_SilkS, pcbnew.B_Fab)):
        parts, texts, bodies = [], [], []
        for fp in brd.GetFootprints():
            if fp.GetLayer() != cu: continue
            bs = [box_mm(p.GetBoundingBox()) for p in fp.Pads()]
            bs += [box_mm(g.GetBoundingBox()) for g in fp.GraphicalItems() if g.GetLayer() == fab and g.Type() == pcbnew.PCB_SHAPE_T]
            body = union(bs) if bs else box_mm(fp.GetBoundingBox(False, False)); bodies.append(body)
            ref = fp.GetReference(); r = fp.Reference()
            if DES.match(ref) and pcls(ref) >= 0: parts.append(dict(box=body, des=ref, cls=pcls(ref)))
            if DES.match(ref) and r.IsVisible() and r.GetLayer() == silk:
                texts.append(dict(box=box_mm(r.GetBoundingBox()), text=ref, angle=round(r.GetTextAngleDegrees(), 1)))
        keep = [t for t in texts if max((inside_frac(t["box"], b) for b in bodies), default=0) < 0.5]
        st["hidden_labels"] += len(texts)-len(keep); texts = keep
        if side == "bottom":
            for o in parts + texts: x0, y0, x1, y1 = o["box"]; o["box"] = [-x1, y0, -x0, y1]
        tset = {t["text"] for t in texts}; n_link = sum(1 for p in parts if p["des"] in tset)
        if n_link < 10: continue
        allb = [o["box"] for o in parts+texts]; u = union(allb)
        for o in parts+texts: o["box"] = [round(o["box"][0]-u[0], 3), round(o["box"][1]-u[1], 3), round(o["box"][2]-u[0], 3), round(o["box"][3]-u[1], 3)]
        out.append(dict(id=f"{key}_{side}", source="kicad", board=name, side=side, W=round(u[2]-u[0], 2), H=round(u[3]-u[1], 2), parts=parts, texts=texts))
        st["sides_kept"] += 1
    return out, st

if __name__ == "__main__":
    sel = json.load(open(IN/"selected.json")); B = []; tot = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        for k, (o, st) in enumerate(pool.imap_unordered(extract, sel, chunksize=4)):
            B += o
            for a, v in st.items(): tot[a] = tot.get(a, 0)+v
            if k % 200 == 0: print(f"boards {k+1}/{len(sel)}, sides kept {len(B)}", flush=True)
    B.sort(key=lambda b: b["id"]); json.dump(B, open(OUT/"kicad_all.json", "w"))
    n_p = sum(len(b["parts"]) for b in B); n_t = sum(len(b["texts"]) for b in B)
    n_l = sum(sum(1 for p in b["parts"] if p["des"] in {t["text"] for t in b["texts"]}) for b in B)
    stats = dict(boards_in=len(sel), projects=len({b["board"] for b in B}), sides=len(B), top=sum(b["side"] == "top" for b in B),
                 parts=n_p, texts=n_t, linkable=n_l, **tot)
    json.dump(stats, open(OUT/"kicad_all_stats.json", "w"), indent=1); print(json.dumps(stats))
