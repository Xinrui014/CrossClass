#!/usr/bin/env python3
"""Candidates for cleaning the WACV link annotation (WACV-clean v1), with one crop each, for the owner's review page (page/review/).
  A  split / shared designator: more annotated parts carry a designator than it is printed on the board -> are these boxes one part?
  B  typo: a part's designator is not printed, but a label 1 character away with the same prefix, naming no part, sits within 3 part sizes
  C  missing designator: a part without designator; the nearest label naming no part, whose prefix fits the part's class, within 2 part sizes
     (each free label offered to its nearest such part only)
Writes page/review/cands.json + page/review/img/<id>.jpg crops and results/linking/wacv_clean_candidates.json (same list, no images).
Usage: wacv_review_cands.py   (python: ~/venvs/gdino_fewshot, PYTHONPATH=~/projects/gdino_fewshot/assoc)"""
import json, re, io, base64, collections, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from assoc_lib import RAW, pfx_class
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))
W = json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv.json")); DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
CYAN, ORANGE, YELLOW, MAG = (34, 211, 238), (251, 146, 60), (250, 204, 21), (232, 121, 249)
pfx = lambda s: (re.match(r"[A-Z]+", s or "") or [""])[0]
def ed1(a, b):
    if a == b or abs(len(a)-len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i]+l[i+1:] == s for i in range(len(l)))
ctr = lambda bx: ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
size = lambda bx: max(bx[2]-bx[0], bx[3]-bx[1], 1)
dist = lambda a, b: float(np.hypot(ctr(a)[0]-ctr(b)[0], ctr(a)[1]-ctr(b)[1]))

def crop(img, b, focus_parts, labels, out=380):
    """Square crop around the focus boxes; focus parts magenta (A) or cyan, labels yellow, everything else thin."""
    boxes = [b["parts"][i]["box"] for i in focus_parts] + [b["texts"][j]["box"] for j in labels]; B = np.array(boxes)
    pad = max(1.3*max(size(x) for x in boxes), 40); x0, y0 = B[:, 0].min()-pad, B[:, 1].min()-pad; x1, y1 = B[:, 2].max()+pad, B[:, 3].max()+pad
    side = max(x1-x0, y1-y0); cx, cy = (x0+x1)/2, (y0+y1)/2; x0, y0 = cx-side/2, cy-side/2; s = out/side
    im = img.crop((int(x0), int(y0), int(x0+side), int(y0+side))).resize((out, out), Image.LANCZOS); d = ImageDraw.Draw(im)
    T = lambda bx: [(bx[0]-x0)*s, (bx[1]-y0)*s, (bx[2]-x0)*s, (bx[3]-y0)*s]
    vis = lambda bx: bx[2] > x0 and bx[0] < x0+side and bx[3] > y0 and bx[1] < y0+side
    for t in b["texts"]:
        if vis(t["box"]): d.rectangle(T(t["box"]), outline=ORANGE, width=1)
    for p in b["parts"]:
        if vis(p["box"]): d.rectangle(T(p["box"]), outline=CYAN, width=1)
    for j in labels: d.rectangle(T(b["texts"][j]["box"]), outline=YELLOW, width=3)
    col = MAG if len(focus_parts) > 1 else CYAN
    for i in focus_parts: d.rectangle(T(b["parts"][i]["box"]), outline=col, width=4)
    return im

cands = []
for b in W:
    P, Tx = b["parts"], b["texts"]; ts = collections.Counter(t["text"] for t in Tx); pd = collections.Counter(p["des"] for p in P if p["des"])
    img = Image.open(next(f for f in sorted((RAW/b["id"]).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))).convert("RGB")
    free = [j for j, t in enumerate(Tx) if t["text"] not in pd]
    for d, n in sorted(pd.items()):                                                       # A
        if ts[d] and n > ts[d]:
            idx = [i for i, p in enumerate(P) if p["des"] == d]; lab = [j for j, t in enumerate(Tx) if t["text"] == d]
            cands.append(dict(id=f"A_{b['id']}_{d}", type="A", image=b["id"], dev=b["id"] in DEV, parts=idx, labels=lab, designator=d,
                              classes=[CLS[P[i]["cls"]] for i in idx], n_parts=n, n_labels=ts[d], crop=crop(img, b, idx, lab)))
    for i, p in enumerate(P):                                                             # B
        if p["des"] and p["des"] not in ts:
            near = sorted(free, key=lambda j: dist(p["box"], Tx[j]["box"]))[:6]
            ok = [j for j in near if ed1(Tx[j]["text"], p["des"]) and pfx(Tx[j]["text"]) == pfx(p["des"]) and dist(p["box"], Tx[j]["box"])/size(p["box"]) <= 3]
            if ok:
                j = ok[0]; cands.append(dict(id=f"B_{b['id']}_{i}", type="B", image=b["id"], dev=b["id"] in DEV, parts=[i], labels=[j], designator=p["des"],
                                             printed=Tx[j]["text"], classes=[CLS[p["cls"]]], crop=crop(img, b, [i], [j])))
    offers = {}                                                                           # C: nearest des-less part of a fitting class per free label
    for j in free:
        c = pfx_class(Tx[j]["text"])
        if c is None: continue
        best = None
        for i, p in enumerate(P):
            if p["des"] or p["cls"] != c: continue
            r = dist(p["box"], Tx[j]["box"])/size(p["box"])
            if r <= 2 and (best is None or r < best[1]): best = (i, r)
        if best and (best[0] not in offers or best[1] < offers[best[0]][1]): offers[best[0]] = (j, best[1])
    for i, (j, r) in sorted(offers.items()):
        cands.append(dict(id=f"C_{b['id']}_{i}", type="C", image=b["id"], dev=b["id"] in DEV, parts=[i], labels=[j], designator="",
                          printed=Tx[j]["text"], classes=[CLS[P[i]["cls"]]], dist=round(r, 2), crop=crop(img, b, [i], [j])))
cands.sort(key=lambda c: (c["image"], c["type"], c["id"]))
IMG = ROOT/"page/review/img"; IMG.mkdir(parents=True, exist_ok=True)
for c in cands: c["crop"].save(IMG/f"{c['id']}.jpg", quality=82); c["img"] = f"img/{c['id']}.jpg"; del c["crop"]
json.dump(cands, open(ROOT/"page/review/cands.json", "w"))
json.dump(cands, open(ROOT/"results/linking/wacv_clean_candidates.json", "w"), indent=0)
cnt = collections.Counter(c["type"] for c in cands); print("candidates", len(cands), dict(cnt), "| images", len({c["image"] for c in cands}),
      "| size MB %.1f" % ((ROOT/"page/review/cands.json").stat().st_size/1e6))
