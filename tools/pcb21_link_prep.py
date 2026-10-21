#!/usr/bin/env python3
"""Pre-links for the owner's offline link-labelling tool on 21PCB (owner 2026-10-20: label the part -> text links directly; no
LinkFormer, so the GT does not favour our method). 28 photos, GT parts of data/21pcb/final_v1/gt9_21pcb.json.
Texts: batch 1 (12 design-aligned photos) = the owner's GT text boxes, string = the owner's answer on the text review page if any,
else its suggestion (tools/pcb21_text_review.py; boxes answered 'not text' are left out); batch 2 (16 photos) = all PP-OCRv6 boxes of
data/21pcb/prelabel.json (junk included; texts nobody links to are the useless patches).
Links (one-to-one, "how" tells the owner how much to trust it):
  match : batch 1, the part's designator (design file) equals the text string or its OCR reading
  rule  : batch 2, the BoM-constrained rule link of tools/pcb21_prelabel.py (GT part = pre-label part with IoU >= 0.5)
  near  : the rest, Hungarian on edge gap / part size (+0.5 if the string is not within edit distance 1 of the designator;
          batch 2: only designator-like texts whose prefix class equals the part class), gap < GAP part sizes
  none  : no candidate
Output: data/21pcb/linktool/data/<stem>.json in the coordinates of the downscaled photo img/<stem>.jpg (copied from data/21pcb/labelme/):
  {stem, batch, img, iw, ih, scale (img px per native px), parts: [{id, box, cls, des, link (text id or null), how, checked}],
   texts: [{id, box, str, src}]}
  pcb21_link_prep.py   (env ~/venvs/gdino_fewshot)"""
import json, os, re, shutil, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; OUT = D/"linktool"
sys.path.insert(0, str(Path.home()/"projects/gdino_fewshot/assoc")); from assoc_lib import pfx_class
GAP = 2.0
norm = lambda s: re.sub(r"[^A-Z0-9]", "", (s or "").upper())
DES = re.compile(r"^[A-Z]{1,5}[0-9]{1,4}[A-Z]?$")

def lev(a, b):
    p = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        q = [i]
        for j, cb in enumerate(b, 1): q.append(min(p[j]+1, q[j-1]+1, p[j-1]+(ca != cb)))
        p = q
    return p[-1]

def gap(a, b):
    return float(np.hypot(max(0, max(a[0], b[0]) - min(a[2], b[2])), max(0, max(a[1], b[1]) - min(a[3], b[3]))))

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = w*h
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

G = json.load(open(D/"final_v1/gt9_21pcb.json")); PL = json.load(open(D/"prelabel.json"))["photos"]
items = {it["id"]: it for it in json.load(open(ROOT/"page/text_review/items.json"))}
ANS = json.load(open(D/"text_review_answers.json")) if (D/"text_review_answers.json").exists() else {}
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
(OUT/"data").mkdir(parents=True, exist_ok=True); (OUT/"img").mkdir(exist_ok=True); tot = {}
for st, g in sorted(G.items()):
    batch = 1 if g["texts"] else 2
    iw, ih = Image.open(D/"labelme"/f"{st}.jpg").size; sc = iw / g["W"]
    parts = [dict(box=b[:4], cls=b[5], des=(b[6] or "").strip()) for b in g["boxes"]]
    texts = []
    if batch == 1:
        for i, t in enumerate(g["texts"]):
            iid = f"{st}#{i}"; a = ANS.get(iid); it = items.get(iid)
            if a and a["status"] == "not_text": continue
            s = a["value"] if a and a["status"] in ("ok", "fixed") else (it["sugg"] if it else t[4])
            texts.append(dict(box=t[:4], str=s, read=(it or {}).get("read", ""), src="gt"))
    else:
        texts = [dict(box=t["box"], str=t["text"], read=t["text"], src="ocr") for t in PL[st]["texts"]]
    link = {}; how = {}
    if batch == 1:                                                   # exact designator match, nearest text if repeated
        for k, p in enumerate(parts):
            d = norm(p["des"])
            if not d: continue
            c = [j for j, t in enumerate(texts) if j not in link.values() and d in (norm(t["str"]), norm(t["read"]))]
            if c: j = min(c, key=lambda j: gap(p["box"], texts[j]["box"])); link[k] = j; how[k] = "match"
    else:                                                            # the pre-label rule link
        PP = PL[st]["parts"]
        for k, p in enumerate(parts):
            best = max(range(len(PP)), key=lambda q: iou(p["box"], PP[q]["box"]), default=None)
            if best is None or iou(p["box"], PP[best]["box"]) < 0.5: continue
            j = PP[best].get("text")
            if j is not None and j >= 0 and j not in link.values(): link[k] = j; how[k] = "rule"
    rest = [k for k in range(len(parts)) if k not in link]; free = [j for j in range(len(texts)) if j not in link.values()]
    if rest and free:
        C = np.full((len(rest), len(free)), 1e6)
        for a, k in enumerate(rest):
            p = parts[k]; sz = max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1], 1.0); d = norm(p["des"])
            for b, j in enumerate(free):
                t = texts[j]; s = norm(t["str"]); r = gap(p["box"], t["box"]) / sz
                if r >= GAP: continue
                if batch == 2 and not (DES.match(s) and p["cls"] in CLS and pfx_class(s) == CLS.index(p["cls"])): continue
                if batch == 1 and not s: continue
                C[a, b] = r + (0.0 if d and lev(d, s) <= 1 else 0.5)
        for a, b in zip(*linear_sum_assignment(C)):
            if C[a, b] < 1e6: link[rest[a]] = free[b]; how[rest[a]] = "near"
    S = lambda b: [round(v*sc, 1) for v in b]
    doc = dict(stem=st, batch=batch, img=f"img/{st}.jpg", iw=iw, ih=ih, scale=sc,
               parts=[dict(id=f"p{k}", box=S(p["box"]), cls=p["cls"], des=p["des"], link=(f"t{link[k]}" if k in link else None),
                           how=how.get(k, "none"), checked=False) for k, p in enumerate(parts)],
               texts=[dict(id=f"t{j}", box=S(t["box"]), str=t["str"], src=t["src"]) for j, t in enumerate(texts)])
    json.dump(doc, open(OUT/"data"/f"{st}.json", "w")); shutil.copy2(D/"labelme"/f"{st}.jpg", OUT/"img"/f"{st}.jpg")
    c = {h: sum(1 for p in doc["parts"] if p["how"] == h) for h in ("match", "rule", "near", "none")}
    tot[st] = c; print(f"{st} (batch {batch}): {len(parts)} parts, {len(texts)} texts, {c}", flush=True)
s = {h: sum(v[h] for v in tot.values()) for h in ("match", "rule", "near", "none")}; print("TOTAL", s, sum(s.values()), "parts")
