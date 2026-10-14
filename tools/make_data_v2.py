#!/usr/bin/env python3
"""Assemble linkdata/CrossClass_data_v2/ (2026-10-14): KiCad sides re-extracted by tools/kicad_links.py with the v2 part rule (any real
component counts, whatever its prefix; extended prefix -> class table, class 9 = other), plus the WACV files of CrossClass_data_v1
unchanged (wacv.json, folds.json, wacv_clean_v1.json). Validation split: the same rule as v1 (repo groups with md5(group) % 20 == 0).
Writes kicad.json, split.json, stats.json, the WACV copies and README.md, and prints the change against v1 per label prefix.
Usage: make_data_v2.py [--src linkdata/kicad_full_v2.json]   (python: any python3)"""
import argparse, json, hashlib, re, shutil, collections, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); V1 = ROOT/"linkdata/CrossClass_data_v1"; V2 = ROOT/"linkdata/CrossClass_data_v2"
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(ROOT/"linkdata/kicad_full_v2.json")); a = ap.parse_args()
K = json.load(open(a.src)); V2.mkdir(parents=True, exist_ok=True)
json.dump(K, open(V2/"kicad.json", "w"))
groups = sorted({b["group"] for b in K}); val = [g for g in groups if int(hashlib.md5(g.encode()).hexdigest(), 16) % 20 == 0]
json.dump(dict(rule="validation = repo groups (lower-case repo name) with md5(group) % 20 == 0", val_groups=val), open(V2/"split.json", "w"), indent=0)
for f in ("wacv.json", "folds.json", "wacv_clean_v1.json"): shutil.copy(V1/f, V2/f)
vs = set(val); pfx = lambda s: (re.match(r"[A-Z]+", s) or [""])[0]
def count(bs):
    return dict(sides=len(bs), groups=len({b["group"] for b in bs}), parts=sum(b["n_parts"] for b in bs), links=sum(b["n_links"] for b in bs))
tr = [b for b in K if b["group"] not in vs]; va = [b for b in K if b["group"] in vs]
cls = collections.Counter(p["cls"] for b in K for p in b["parts"])
stats = dict(name="CrossClass_data_v2", train=count(tr), val=count(va), train_wacv_density=count([b for b in tr if 2.1 <= b["density"] <= 11.2]),
             classes={str(k): v for k, v in sorted(cls.items())})
json.dump(stats, open(V2/"stats.json", "w"), indent=1)
# what changed against v1: label prefixes of linked parts (train split)
old = json.load(open(V1/"kicad.json")); ov = set(json.load(open(V1/"split.json"))["val_groups"])
def linked(bs, vg):
    c = collections.Counter()
    for b in bs:
        if b["group"] in vg: continue
        ts = {t["text"] for t in b["texts"]}
        for p in b["parts"]:
            if p["des"] in ts: c[pfx(p["des"])] += 1
    return c
c1, c2 = linked(old, ov), linked(K, vs)
diff = sorted(((p, c1[p], c2[p]) for p in set(c1) | set(c2) if c2[p] != c1[p]), key=lambda x: -(x[2]-x[1]))
oth = collections.Counter(pfx(p["des"]) for b in tr for p in b["parts"] if p["cls"] == 9)
README = f"""# CrossClass_data_v2 — part ↔ designator link data (built 2026-10-14, `tools/make_data_v2.py`)

Same format and WACV files as CrossClass_data_v1; only `kicad.json` changed. KiCad sides re-extracted by `tools/kicad_links.py` with the
v2 part rule: every real, populated component is a part whatever its designator prefix (not parts: footprints without pads,
board-only or do-not-populate footprints, mounting holes / fiducials / test points / net ties / solder jumpers / logos by footprint name,
prefixes TP H MH FID NT SJ GND LOGO G REF). v1 kept only footprints with prefixes R RN FB L C LED D SW S Q IC U Y X J P, so jumpers (JP),
CN connectors, LD / DS LEDs, buttons, pots ... were texts naming no part, and string-reading linkers learned that those prefixes never
name a part (v2_S, 2026-10-14). Class from an extended prefix table (WACV conventions); unknown prefixes = class 9 "other".

| split | sides | groups | parts | links |
|---|---|---|---|---|
| train | {stats['train']['sides']:,} | {stats['train']['groups']:,} | {stats['train']['parts']:,} | {stats['train']['links']:,} |
| val | {stats['val']['sides']:,} | {stats['val']['groups']:,} | {stats['val']['parts']:,} | {stats['val']['links']:,} |

Class ids 0 resistor, 1 capacitor, 2 inductor, 3 connector, 4 diode, 5 switch, 6 transistor, 7 IC, 8 oscillator, 9 other: {stats['classes']}.
Biggest changes in linked parts (train, v1 → v2): {', '.join(f'{p} {o:,} → {n:,}' for p, o, n in diff[:20])}.
Most common class-9 prefixes: {', '.join(f'{p} {n}' for p, n in oth.most_common(15))}.
"""
open(V2/"README.md", "w").write(README); print(README)
