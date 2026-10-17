#!/usr/bin/env python3
"""Part type from the KiCad footprint name and Value field, for every part of CrossClass_data_v2 (2026-10-17, proposal B: a class target
that does not come from the designator prefix, so a model can learn board naming conventions such as LEDs named L1). Keyword rules as in
tools/kicad_world_probe.py; -1 when no rule fires. Parts are matched to footprints by designator (first footprint with that reference on
the side's copper layer). Writes linkdata/kicad_fptype/<side>.json {id, fptype: [class or -1 per kicad.json part], fp: [lib:name], value: [...]}.
Usage: /usr/bin/python3 kicad_fptype.py --worker I --nworkers N   (system python with pcbnew; CROSSCLASS project root)"""
import argparse, json, os, sys, time
from pathlib import Path
import pcbnew
sys.path.insert(0, str(Path(__file__).parent))
from kicad_world_probe import fp_type
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); OUT = ROOT/"linkdata/kicad_fptype"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1); a = ap.parse_args()
K = json.load(open(ROOT/"linkdata/CrossClass_data_v2/kicad.json")); by_key = {}
for b in K: by_key.setdefault(b["id"].rsplit("_", 1)[0], []).append(b)
for n, (key, sides) in enumerate(sorted(by_key.items())):
    if n % a.nworkers != a.worker or all((OUT/f"{s['id']}.json").exists() for s in sides): continue
    t0 = time.time()
    try:
        brd = pcbnew.LoadBoard(str(ROOT/f"kicad_scan/boards/{key}.kicad_pcb"))
        for s in sides:
            cu = pcbnew.B_Cu if s["side"] == "bottom" else pcbnew.F_Cu; fps = {}
            for fp in brd.GetFootprints():
                if fp.GetLayer() == cu: fps.setdefault(fp.GetReference(), fp)
            ft, fpn, val = [], [], []
            for p in s["parts"]:
                fp = fps.get(p["des"])
                if fp is None: ft.append(-1); fpn.append(""); val.append(""); continue
                fid = fp.GetFPID(); lib, name = str(fid.GetLibNickname()), str(fid.GetLibItemName()); v = fp.GetValue(); t = fp_type(lib, name, v)
                ft.append(-1 if t is None else t); fpn.append(f"{lib}:{name}"); val.append(v)
            json.dump(dict(id=s["id"], fptype=ft, fp=fpn, value=val), open(OUT/f"{s['id']}.json", "w"))
        print(f"{key} ok {len(sides)} sides {time.time()-t0:.1f}s", flush=True)
    except Exception as ex:
        print(f"{key} FAILED {repr(ex)[:160]}", flush=True)
