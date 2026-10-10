#!/usr/bin/env python3
"""Per-part comparison of the class each method gives to every GT part on the 42 WACV test images:
  det      PCB-Dense v22 detector alone (reference)                      preds_wacv42_1280.json
  qwen     BASELINE: Qwen3.8 decides the class of EVERY box (v2 prompt)  --qwen wacvqc2ALL42
  ocr      OURS: PP-OCRv6 text + learned linking (p >= 0.5)               --ocr wacvocrp05_test
  (combo   OCR + linking first, else Qwen reads the label — only with --combo <set>; the owner's three-way report leaves it out)
A part's answer under a method = class of its best-IoU detection (IoU >= 0.5, conf >= 0.25) or "missed".
Writes <out>/compare.json (counts per true class x method, confusion pairs with counts, samples) and <out>/img/*.jpg
(the two-panel image the VLM saw for that detection — qwen_label/crops_both/, downsized to 768 px wide — or, when the part has
no base detection, a 256 px crop around the GT box, 3x its size, GT box in magenta). Project root = $CROSSCLASS. Usage: err_compare.py [--out DIR, default page/] [--qwen wacvqc3ALL42] [--ocr wacvocrp05_test]"""
import argparse, json, os, re, random, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); E = ROOT/"data"; Q = ROOT/"answers"; IMG = ROOT/"data/wacv_images"
sys_path_tools = str(ROOT/"tools"); import sys; sys.path.insert(0, sys_path_tools); os.chdir(E)   # analyze9 reads from the cwd
import analyze9 as A
Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT/"page")); ap.add_argument("--combo", default="")
ap.add_argument("--qwen", default="wacvqc2ALL42"); ap.add_argument("--ocr", default="wacvocrp05_test"); ap.add_argument("--per_pair", type=int, default=4); ap.add_argument("--max_per_class", type=int, default=16)
a = ap.parse_args(); OUT = Path(a.out); (OUT/"img").mkdir(parents=True, exist_ok=True); NM = A.NM
PFX = [("RN", 0), ("RA", 0), ("RP", 0), ("RM", 0), ("PR", 0), ("R", 0), ("FB", 2), ("TR", 2), ("L", 2), ("T", 2), ("CE", 1), ("FC", 1), ("C", 1),
       ("LED", 4), ("LD", 4), ("DS", 4), ("ZD", 4), ("CR", 4), ("D", 4), ("Z", 4), ("SW", 5), ("BTN", 5), ("S", 5), ("Q", 6), ("IC", 7), ("U", 7),
       ("Y", 8), ("X", 8), ("JP", 3), ("CN", 3), ("CON", 3), ("J", 3), ("P", 3)]
def pfx_class(d):
    m = re.match(r"([A-Z]+)\d", d or "")
    if not m: return None
    for p, c in PFX:
        if m.group(1) == p: return c
    return None
def nms_classwise(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = A.iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
base = json.load(open(E/"preds_wacv42_1280.json")); imgs = sorted(base)
# reader preds: primary rule from results/qwen_label (high confidence AND detector score < 0.7), prefix table, class-wise NMS
ans = {}
for l in open(Q/"answers_q38.jsonl"):
    r = json.loads(l); ans[r["key"]] = r
qread = {}
for st in imgs:
    nb = []
    for k, p in enumerate(base[st]):
        p = list(p); r = ans.get(f"{st}|{k}")
        if r and r["confidence"] == "high" and p[4] < 0.7:
            c = pfx_class(r["designator"])
            if c is not None: p[5] = c
        nb.append(p)
    qread[st] = nms_classwise(nb)
M = {"det": base, "qwen": json.load(open(E/f"preds_{a.qwen}_1280.json")), "ocr": json.load(open(E/f"preds_{a.ocr}_1280.json"))}
if a.combo: M["combo"] = json.load(open(E/f"preds_{a.combo}_1280.json"))
qc_ans = {}
for l in open(Q/"answers_qc2.jsonl"):
    r = json.loads(l); qc_ans[r["key"]] = r
def answer(preds, st, gb):
    P = [p for p in preds.get(st, []) if p[4] >= 0.25]
    if not P: return "missed", None
    I = A.iou(np.array([gb]), np.array([p[:4] for p in P]))[0]; j = int(np.argmax(I))
    return (int(P[j][5]), float(P[j][4])) if I[j] >= 0.5 else ("missed", None)
parts = []   # one record per GT part
for d in A.load("wacv42"):
    st = d["st"]
    for i, (gb, gc) in enumerate(zip(d["gm"], d["gc"])):
        rec = dict(st=st, i=i, box=[float(v) for v in gb], gt=int(gc))
        for m, P in M.items(): rec[m], rec[m+"_conf"] = answer(P, st, gb)
        # what the two Qwen passes said for the base detection on this part (for captions)
        bI = A.iou(np.array([gb]), np.array([p[:4] for p in base[st]]))[0] if base[st] else np.zeros(0)
        if len(bI) and bI.max() >= 0.5:
            k = int(np.argmax(bI)); r1, r2 = ans.get(f"{st}|{k}"), qc_ans.get(f"{st}|{k}")
            rec["read"] = (r1["designator"], r1["confidence"]) if r1 else None; rec["qc"] = (NM[r2["qwen_cls"]] if r2 and r2["qwen_cls"] is not None else "unsure", r2["confidence"]) if r2 else None
            rec["det_k"] = k
        parts.append(rec)
methods = list(M)
counts = {c: {m: collections.Counter() for m in methods} for c in NM}
pairs = {m: collections.Counter() for m in methods}
for r in parts:
    for m in methods:
        v = r[m]; t = NM[r["gt"]]
        if v == "missed": counts[t][m]["missed"] += 1; pairs[m][(t, "missed")] += 1
        elif v == r["gt"]: counts[t][m]["right"] += 1
        else: counts[t][m]["wrong"] += 1; pairs[m][(t, NM[v])] += 1
# samples: per true class, for each method's most frequent wrong pairs, a few parts (seeded), deduplicated; crops around the GT box
rng = random.Random(0); cache = {}; samples = {}
def crop(rec):
    st = rec["st"]; fn = OUT/"img"/f"{st}__{rec['i']}.jpg"
    if fn.exists(): return fn.name
    both = ROOT/"crops_both"/f"wacv__{st}__{rec.get('det_k', -1)}.jpg"
    if "det_k" in rec and both.exists():
        im = Image.open(both).convert("RGB"); im.resize((768, round(768*im.height/im.width)), Image.LANCZOS).save(fn, quality=85); return fn.name
    if st not in cache: cache[st] = Image.open(next(IMG.glob(st+".*"))).convert("RGB")
    im = cache[st]; b = rec["box"]; W, Hh = im.size; L = max(b[2]-b[0], b[3]-b[1]); side = int(min(max(3*L, 120), 1600))
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; x0 = int(min(max(cx-side/2, 0), max(W-side, 0))); y0 = int(min(max(cy-side/2, 0), max(Hh-side, 0)))
    c = im.crop((x0, y0, x0+side, y0+side)); s = 256/side; c = c.resize((256, 256), Image.LANCZOS if s < 1 else Image.BICUBIC)
    dr = ImageDraw.Draw(c); dr.rectangle([(b[0]-x0)*s-2, (b[1]-y0)*s-2, (b[2]-x0)*s+2, (b[3]-y0)*s+2], outline=(255, 0, 255), width=2)
    c.save(fn, quality=85); return fn.name
for t in NM:
    chosen = []; seen = set()
    for m in methods:
        top = [pr for pr, n in pairs[m].most_common() if pr[0] == t][:3]
        for pr in top:
            cands = [r for r in parts if NM[r["gt"]] == t and ((r[m] == "missed") if pr[1] == "missed" else (r[m] != "missed" and r[m] != r["gt"] and NM[r[m]] == pr[1]))]
            rng.shuffle(cands)
            for r in cands:
                key = (r["st"], r["i"])
                if key in seen: continue
                seen.add(key); chosen.append(dict(r, img=crop(r), why=f"{m}: {t} → {pr[1]}"))
                if sum(1 for c in chosen if c["why"] == f"{m}: {t} → {pr[1]}") >= a.per_pair: break
            if len(chosen) >= a.max_per_class: break
        if len(chosen) >= a.max_per_class: break
    samples[t] = chosen
json.dump(dict(n_parts=len(parts), methods=methods, counts={t: {m: dict(c) for m, c in cm.items()} for t, cm in counts.items()},
               pairs={m: [[list(k), v] for k, v in pairs[m].most_common()] for m in methods}, samples=samples, names=NM), open(OUT/"compare.json", "w"), indent=1)
print("parts", len(parts))
for t in NM: print(t, {m: dict(counts[t][m]) for m in methods})
for m in methods: print(m, pairs[m].most_common(8))
