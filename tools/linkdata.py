#!/usr/bin/env python3
"""Boards in the fixed linking format (LINKING.md §10): per board, parts = boxes, texts = boxes + designator-like strings,
links = exact string matches (a part's designator == a text's string). Part classes are kept for analysis only, never as model input.
  wacv     - the 45 annotated WACV images (assoc_lib.load_gt)
  kicad199 - the 199 rendered KiCad boards (gdino_fewshot/kicad_render/renders/*.json; pixel boxes, exact Reference texts)
Writes linkdata/<name>.json = [{id, source, board, W, H, parts: [{box, des, cls}], texts: [{box, text, angle}]}].
Usage: linkdata.py   (python: ~/venvs/gdino_fewshot, PYTHONPATH=~/projects/gdino_fewshot/assoc)"""
import json, re, os
from pathlib import Path
from PIL import Image
from assoc_lib import load_gt, RAW, DES, E
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); OUT = ROOT/"linkdata"; OUT.mkdir(exist_ok=True)
board = lambda st: re.sub(r"_(Top|Bottom)\d*$", "", st)

wacv = []; IMGS = sorted(json.load(open(E/"preds_wacv_1280.json")))   # the 45 images used everywhere in CrossClass
for st in IMGS:
    d = RAW/st; parts, texts = load_gt(st)
    W, H = Image.open(next(f for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))).size
    wacv.append(dict(id=d.name, source="wacv", board=board(d.name), W=W, H=H,
                     parts=[dict(box=p["box"], des=p["des"], cls=p["cls"]) for p in parts],
                     texts=[dict(box=t["box"], text=t["text"], angle=None) for t in texts]))
json.dump(wacv, open(OUT/"wacv.json", "w"))

kic = []
for jf in sorted((Path.home()/"projects/gdino_fewshot/kicad_render/renders").glob("b[0-9][0-9][0-9].json")):
    k = json.load(open(jf))
    parts = [dict(box=p["box"], des=p["ref"] if DES.match(p["ref"]) else "", cls=p["cls"]) for p in k["parts"] if p["cls"] >= 0]
    texts = [dict(box=t["box"], text=t["text"], angle=t.get("angle")) for t in k["texts"] if DES.match(t["text"])]
    kic.append(dict(id=k["id"], source="kicad", board=k["name"], W=k["W"], H=k["H"], parts=parts, texts=texts))
json.dump(kic, open(OUT/"kicad199.json", "w"))
import shutil; shutil.copy(Path.home()/"projects/gdino_fewshot/wacv_in/folds.json", OUT/"folds.json")   # board folds travel with the data

for name, B in (("wacv", wacv), ("kicad199", kic)):
    n_p = sum(len(b["parts"]) for b in B); n_t = sum(len(b["texts"]) for b in B)
    n_l = sum(sum(1 for p in b["parts"] if p["des"] and p["des"] in {t["text"] for t in b["texts"]}) for b in B)
    print(f"{name}: {len(B)} boards, {n_p} parts, {n_t} designator texts, {n_l} linkable parts")
