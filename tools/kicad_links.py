#!/usr/bin/env python3
"""Stage 2 of the KiCad link set: .kicad_pcb -> boards in the linking format of linkdata/wacv.json, one record per board side.
- Parts: footprints on the side whose designator prefix maps to one of the 9 classes (table of kicad_render/render_all.py). Box = the
  physical body: bounding box of the pads and the Fab-layer shapes (fallback: footprint box without text). Footprints with no class
  (test points, holes, fiducials, logos) are not parts; their visible labels stay as texts that name no part, like WACV's test points.
- Texts: visible Reference fields on the side's silkscreen whose string looks like a designator, minus labels a photo would not show:
  >= 50 % of the label box inside any footprint body (printed under a component).
- Excluded projects: possible recreations of / close relatives to WACV boards (EXCLUDE below).
- Bottom sides are mirrored in x, as a photo of the back would be. A side is kept with >= 10 linkable parts. Units: mm.
- Metadata per side: group (lower-case repo name, for fork-safe train/val splits), density (parts / cm² of the side's part extent),
  smd_share, KiCad file version, part count.
Robust to KiCad loader crashes, hangs (5 min per board) and memory blow-ups (24 GB per worker): each worker writes the board it is on to w<i>.cur; the driver relaunches a crashed worker and skips
that board (w<i>.failed).
Usage: /usr/bin/python3 kicad_links.py --run 24 [--list kicad_boards/selected.json --in kicad_boards --out linkdata/kicad_all.json]
       full set: --list kicad_scan/selected.json --in kicad_scan/boards --out linkdata/kicad_full.json"""
import json, re, sys, os, argparse, subprocess, time
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))
DES = re.compile(r"^[A-Z]{1,4}\d{1,4}[A-Z]?$")
EXCLUDE = re.compile(r"^(OLIMEX/|Duet3D/|BehrensG/ARTY_Z7|boobalakrishnan08/Arty100T|antmicro/arty-expansion-board|muhardianab/ArduinoMega2560|"
                     r"ISLOWIsmaeli/arduino-mega-design|AlanFord/ArduinoMega6axis|Moddler/CCD-Arduino-Mega|Villabaal/ArduinoMega-Autoclave|"
                     r"gggmmm/pcb-spartan-3|Fabeltranm/SPARTAN6|jonthomasson/SpartanMini|tcfh2003/Virtex-6|viktor-nikolov/MicroZed)", re.I)
PFX = [("RN", 0), ("R", 0), ("FB", 2), ("L", 2), ("C", 1), ("LED", 4), ("D", 4), ("SW", 5), ("S", 5), ("Q", 6), ("IC", 7), ("U", 7), ("Y", 8), ("X", 8), ("J", 3), ("P", 3)]
def pcls(ref):
    m = re.match(r"([A-Z]+)\d", ref); return next((c for p, c in PFX if m and m.group(1) == p), -1)
def box_mm(bb): return [bb.GetX()/1e6, bb.GetY()/1e6, (bb.GetX()+bb.GetWidth())/1e6, (bb.GetY()+bb.GetHeight())/1e6]
def union(bs): return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
def inside_frac(t, b):
    iw = max(0.0, min(t[2], b[2])-max(t[0], b[0])); ih = max(0.0, min(t[3], b[3])-max(t[1], b[1]))
    return iw*ih/max((t[2]-t[0])*(t[3]-t[1]), 1e-9)

def extract(entry, indir):
    import pcbnew
    key, name = entry["key"], entry["name"]; out = []
    if EXCLUDE.search(name): return out, "excluded"
    try: brd = pcbnew.LoadBoard(str(Path(indir)/f"{key}.kicad_pcb"))
    except Exception: brd = None
    if brd is None: return out, "load_failed"
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
        texts = [t for t in texts if max((inside_frac(t["box"], b) for b in bodies), default=0) < 0.5]
        if side == "bottom":
            for o in parts + texts: x0, y0, x1, y1 = o["box"]; o["box"] = [-x1, y0, -x0, y1]
        tset = {t["text"] for t in texts}; n_link = sum(1 for p in parts if p["des"] in tset)
        if n_link < 10: continue
        u = union([o["box"] for o in parts+texts]); pu = union([p["box"] for p in parts])
        for o in parts+texts: o["box"] = [round(o["box"][0]-u[0], 3), round(o["box"][1]-u[1], 3), round(o["box"][2]-u[0], 3), round(o["box"][3]-u[1], 3)]
        area_cm2 = max((pu[2]-pu[0])*(pu[3]-pu[1])/100, 0.01)
        out.append(dict(id=f"{key}_{side}", source="kicad", board=name, group=name.split("/")[-1].lower(), side=side,
                        W=round(u[2]-u[0], 2), H=round(u[3]-u[1], 2), density=round(len(parts)/area_cm2, 3), n_parts=len(parts), n_links=n_link,
                        smd_share=entry.get("smd_share"), version=entry.get("version"), parts=parts, texts=texts))
    return out, "ok" if out else "no_side"

def worker(a):
    import resource, signal
    resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))   # a runaway board fails this worker instead of eating the shared machine
    sel = json.load(open(a.list)); od = Path(a.outdir); i, n = a.worker, a.nworkers
    done = set(open(od/f"w{i}.done").read().split()) if (od/f"w{i}.done").exists() else set()
    failed = set(open(od/f"w{i}.failed").read().split()) if (od/f"w{i}.failed").exists() else set()
    with open(od/f"w{i}.jsonl", "a") as fo, open(od/f"w{i}.done", "a") as fd, open(od/f"w{i}.status", "a") as fs:
        for k, e in enumerate(sel):
            if k % n != i or e["key"] in done or e["key"] in failed: continue
            (od/f"w{i}.cur").write_text(e["key"])
            signal.alarm(300)                     # default SIGALRM action kills the worker, even inside KiCad's C++; driver skips the board
            o, st = extract(e, a.indir); signal.alarm(0)
            for b in o: fo.write(json.dumps(b)+"\n")
            fs.write(f"{e['key']} {st}\n"); fd.write(e["key"]+"\n"); fo.flush(); fd.flush(); fs.flush()
    (od/f"w{i}.cur").unlink(missing_ok=True)

def driver(a):
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True); n = a.run
    cmd = lambda i: [sys.executable, __file__, "--worker", str(i), "--nworkers", str(n), "--list", a.list, "--in", a.indir, "--outdir", str(od)]
    procs = {i: subprocess.Popen(cmd(i), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(n)}; crashes = 0; t0 = time.time()
    while procs:
        time.sleep(5)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None: continue
            if rc == 0: del procs[i]; continue
            cur = od/f"w{i}.cur"; bad = cur.read_text().strip() if cur.exists() else ""
            with open(od/f"w{i}.failed", "a") as f: f.write(bad+"\n")
            crashes += 1; procs[i] = subprocess.Popen(cmd(i), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if int(time.time()-t0) % 120 < 5:
            nd = sum(len(open(f).read().split()) for f in od.glob("w*.done")); print(f"{int(time.time()-t0)}s boards done {nd}, crashes {crashes}", flush=True)
    B = [json.loads(l) for f in sorted(od.glob("w*.jsonl")) for l in open(f)]
    B.sort(key=lambda b: b["id"]); json.dump(B, open(a.out, "w"))
    st = {}
    for f in od.glob("w*.status"):
        for l in open(f): s = l.split()[1]; st[s] = st.get(s, 0)+1
    st["crashed"] = sum(len([x for x in open(f).read().split() if x]) for f in od.glob("w*.failed"))
    links = sum(b["n_links"] for b in B)
    stats = dict(boards_in=len(json.load(open(a.list))), status=st, sides=len(B), top=sum(b["side"] == "top" for b in B), layouts=len({b["id"].rsplit("_", 1)[0] for b in B}),
                 groups=len({b["group"] for b in B}), parts=sum(b["n_parts"] for b in B), texts=sum(len(b["texts"]) for b in B), links=links)
    json.dump(stats, open(Path(a.out).with_name(Path(a.out).stem+"_stats.json"), "w"), indent=1); print(json.dumps(stats))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=0); ap.add_argument("--worker", type=int, default=-1); ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--list", default=str(ROOT/"kicad_boards/selected.json")); ap.add_argument("--in", dest="indir", default=str(ROOT/"kicad_boards"))
    ap.add_argument("--out", default=str(ROOT/"linkdata/kicad_all.json")); ap.add_argument("--outdir", default="")
    a = ap.parse_args(); a.outdir = a.outdir or str(Path(a.out).with_suffix(".parts"))
    worker(a) if a.worker >= 0 else driver(a) if a.run else print(__doc__)
