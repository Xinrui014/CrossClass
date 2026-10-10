#!/usr/bin/env python3
"""Score the full linkformer runs on the 42 WACV test images (oracle track, exact pairs) from the per-image counts in
results/linkformer/<tag>.json, next to B7 / B8 (results/linking/oracle.json). Writes results/linkformer/full_report.md.
  P_<data>_<tok>         pre-trained on KiCad only, evaluated once on all 45 WACV images (zero-shot) and on KiCad validation
  K0_<tok>_fold<k>       trained on WACV fold k only (no KiCad)
  FT_full_<tok>_fold<k>  P_full_<tok> fine-tuned on WACV fold k
Usage: linkformer_report.py"""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); R = ROOT/"results/linkformer"
DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}

def summ(per):
    per = {k: v for k, v in per.items() if k not in DEV}
    c = {k: sum(v[k] for v in per.values()) for k in ("linkable", "right", "links")}
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    return dict(n_img=len(per), acc=acc, prec=prec, f1=2*acc*prec/max(acc+prec, 1e-9), **c)
def load(tag): p = R/f"{tag}.json"; return json.load(open(p)) if p.exists() else None
def folds(prefix):
    per = {}
    for k in range(3):
        r = load(f"{prefix}_fold{k}")
        if not r: return None
        per.update(r["final"][f"fold{k}test"]["per_image"])
    return per

rows = []
o = json.load(open(ROOT/"results/linking/oracle.json"))
for name, key in (("B7 trees, geometry", "B7 learned pair scorer, geometry only + one-to-one"), ("B8 trees + detector class (③)", "B8 learned pair scorer + detector class + one-to-one (current ③ linker)")):
    s = o[key]; rows.append((name, "WACV folds", s["acc"], s["prec"], s["f1"], None))
for tok in ("E2", "E1"):
    for data, label in (("full", "all 1,749 KiCad sides"), ("1000", "1,000 KiCad projects"), ("300", "300 KiCad projects")):
        r = load(f"P_{data}_{tok}")
        if r: s = summ(r["final"]["all45"]["per_image"]); rows.append((f"transformer {tok}, KiCad only (zero-shot)", label, s["acc"], s["prec"], s["f1"], r["final"]["kicadval"]["summary"]["acc"]))
    per = folds(f"K0_{tok}")
    if per: s = summ(per); rows.append((f"transformer {tok}, WACV only", "WACV folds", s["acc"], s["prec"], s["f1"], None))
    per = folds(f"FT_full_{tok}")
    if per: s = summ(per); rows.append((f"transformer {tok}, KiCad pre-train + WACV fine-tune", "KiCad + WACV folds", s["acc"], s["prec"], s["f1"], None))
md = ["# Transformer linker, full runs: 42 WACV test images, oracle track, exact pairs", "",
      "| method | training data | right label | link precision | link F1 | KiCad val right label |", "|---|---|---|---|---|---|"]
for n, d, a, p, f, kv in rows: md.append(f"| {n} | {d} | {100*a:.1f} % | {100*p:.1f} % | {f:.3f} | {'' if kv is None else f'{100*kv:.1f} %'} |")
open(R/"full_report.md", "w").write("\n".join(md)+"\n"); print("\n".join(md))
