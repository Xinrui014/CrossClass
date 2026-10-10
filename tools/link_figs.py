#!/usr/bin/env python3
"""Real-board figures for the linking page (page/linking.html): a WACV test crop with the true links, the same crop linked by
B1 / B4 / B8, image-token crops, and a KiCad render crop with its exact links. Output: page/img/linking/*.jpg
Usage: link_figs.py   (python: ~/venvs/gdino_fewshot)"""
import sys, json, glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import link_baselines as LB
from assoc_lib import load_gt, attach_detector, pair_features, FEAT, RAW
OUT = LB.ROOT/"page/img/linking"; OUT.mkdir(parents=True, exist_ok=True)
CYAN, ORANGE, YELLOW, RED, GREY = (34, 211, 238), (251, 146, 60), (250, 204, 21), (239, 68, 68), (200, 200, 200)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError: FONT = ImageFont.load_default()

# oracle data + links for B1 / B4 / B8
data = {}
for st in LB.IMGS:
    parts, texts = load_gt(st); parts = attach_detector(parts, LB.preds[st])
    feats = pair_features(parts, texts); y = [int(parts[i]["des"] != "" and texts[j]["text"] == parts[i]["des"]) for i, j, _ in feats]
    data[st] = dict(pt=(parts, texts), g=LB.geometry(parts, texts) if parts and texts else None, feats=feats, y=y)
b8, _ = LB.learned_links(data, list(range(len(FEAT))))
links = {}
for st in LB.TEST:
    if data[st]["g"] is None: continue
    p, t = data[st]["pt"]; g = data[st]["g"]
    links[st] = dict(B1=LB.nearest_centre(p, t, g, 2), B4=LB.onetoone_centre(p, t, g, 3), B8=b8.get(st, {}))

def inside(b, w): return b[0] >= w[0] and b[1] >= w[1] and b[2] <= w[2] and b[3] <= w[3]
def window_score(st, w):
    parts, texts = data[st]["pt"]; tin = {j for j, t in enumerate(texts) if inside(t["box"], w)}; strs = {texts[j]["text"]: j for j in tin}
    pin = [i for i, p in enumerate(parts) if inside(p["box"], w)]
    if not 9 <= len(pin) <= 22: return None
    wrong = {"B1": 0, "B4": 0, "B8": 0}; L = 0; none = 0
    for i in pin:
        for m in wrong:
            j = links[st][m].get(i)
            if j is not None and j not in tin: return None
        d = parts[i]["des"]
        if d and d in {texts[j]["text"] for j in range(len(texts))}:
            if d not in strs: return None
            L += 1
            for m in wrong:
                j = links[st][m].get(i); wrong[m] += j is None or texts[j]["text"] != d
        else: none += 1
    if L < 7 or none < 1 or none > 4: return None
    return 3*wrong["B1"]+2*(wrong["B1"]-wrong["B4"] > 0)-4*wrong["B8"]-1.5*wrong["B4"], wrong, L, none

best = None
for st in links:
    parts = data[st]["pt"][0]; W, H = Image.open(sorted((RAW/st).glob("*.jpg"))[0]).size
    if W < 2000: continue
    med = float(np.median([max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]) for p in parts]))
    ww, wh, sd = 16*med, 11*med, 2*med
    for x in np.arange(0, W-ww, sd):
        for y in np.arange(0, H-wh, sd):
            r = window_score(st, (x, y, x+ww, y+wh))
            if r and (best is None or r[0] > best[0]): best = (r[0], st, (x, y, x+ww, y+wh), r[1:])
_, ST, WIN, info = best
print("window", ST, [round(v) for v in WIN], "wrong", info[0], "linkable", info[1], "no label", info[2])
img = Image.open(sorted((RAW/ST).glob("*.jpg"))[0]).convert("RGB")
crop = img.crop(tuple(int(v) for v in WIN)); S = 960/crop.width; crop = crop.resize((960, int(crop.height*S)), Image.LANCZOS)
parts, texts = data[ST]["pt"]
def tr(b): return [(b[0]-WIN[0])*S, (b[1]-WIN[1])*S, (b[2]-WIN[0])*S, (b[3]-WIN[1])*S]
def ctr(b): b = tr(b); return ((b[0]+b[2])/2, (b[1]+b[3])/2)
pin = [i for i, p in enumerate(parts) if inside(p["box"], WIN)]; tin = [j for j, t in enumerate(texts) if inside(t["box"], WIN)]
strs = {t["text"] for t in texts}

def line(d, a, b, col):
    d.line([a, b], fill=(0, 0, 0), width=7); d.line([a, b], fill=col, width=3)
def dashed_box(d, b, col):
    x0, y0, x1, y1 = b
    for (a, c) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(int(np.hypot(c[0]-a[0], c[1]-a[1])/8), 1)
        for k in range(0, n, 2):
            t0, t1 = k/n, min((k+1)/n, 1); d.line([(a[0]+(c[0]-a[0])*t0, a[1]+(c[1]-a[1])*t0), (a[0]+(c[0]-a[0])*t1, a[1]+(c[1]-a[1])*t1)], fill=col, width=3)
def base(draw_links=None):
    im = crop.copy(); d = ImageDraw.Draw(im)
    for j in tin: d.rectangle(tr(texts[j]["box"]), outline=ORANGE, width=3)
    for i in pin:
        lab = parts[i]["des"] and parts[i]["des"] in strs
        (d.rectangle(tr(parts[i]["box"]), outline=CYAN, width=3) if lab else dashed_box(d, tr(parts[i]["box"]), GREY))
    if draw_links is None:   # true links
        for i in pin:
            if parts[i]["des"] in strs:
                j = next(j for j in tin if texts[j]["text"] == parts[i]["des"]); line(d, ctr(parts[i]["box"]), ctr(texts[j]["box"]), YELLOW)
    else:
        for i in pin:
            j = draw_links.get(i); des = parts[i]["des"]; lab = des and des in strs
            if j is not None: line(d, ctr(parts[i]["box"]), ctr(texts[j]["box"]), YELLOW if lab and texts[j]["text"] == des else RED)
            elif lab:
                c = ctr(parts[i]["box"]); d.line([(c[0]-9, c[1]-9), (c[0]+9, c[1]+9)], fill=RED, width=4); d.line([(c[0]-9, c[1]+9), (c[0]+9, c[1]-9)], fill=RED, width=4)
    return im
base().save(OUT/"task.jpg", quality=88)
for m in ("B1", "B4", "B8"):
    im = base(links[ST][m]); d = ImageDraw.Draw(im)
    nw = sum(1 for i in pin if parts[i]["des"] in strs and (links[ST][m].get(i) is None or texts[links[ST][m][i]]["text"] != parts[i]["des"]))
    d.rectangle([0, 0, 330, 38], fill=(0, 0, 0)); d.text((10, 7), f"{m}: {nw} wrong of {info[1]}", fill=(255, 255, 255), font=FONT)
    im.save(OUT/f"links_{m}.jpg", quality=88)

# image-token crops: one labelled part and its text, each with context (box grown 2x), native pixels upscaled
lab = [i for i in pin if parts[i]["des"] in strs]
i0 = max([i for i in lab if parts[i]["des"].startswith("C")] or lab, key=lambda i: (parts[i]["box"][2]-parts[i]["box"][0])*(parts[i]["box"][3]-parts[i]["box"][1]))
j0 = next(j for j in tin if texts[j]["text"] == parts[i0]["des"])
def ctx_crop(b, k=2.0, out=300):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; s = k*max(b[2]-b[0], b[3]-b[1])/2
    c = img.crop((int(cx-s), int(cy-s), int(cx+s), int(cy+s))); f = out/c.width; c = c.resize((out, out), Image.LANCZOS)
    d = ImageDraw.Draw(c); d.rectangle([(b[0]-cx+s)*f, (b[1]-cy+s)*f, (b[2]-cx+s)*f, (b[3]-cy+s)*f], outline=(255, 0, 255), width=3); return c
ctx_crop(parts[i0]["box"], 2.5).save(OUT/"tok_part.jpg", quality=90); ctx_crop(texts[j0]["box"], 2.5).save(OUT/"tok_text.jpg", quality=90)
print("token example:", parts[i0]["des"], "class", parts[i0]["cls"])

# KiCad render crop at a similar density: parts with their exact Reference text
K = H_ = None
KR = Path.home()/"projects/gdino_fewshot/kicad_render/renders"; bestk = None
for jf in sorted(KR.glob("b0[0-4]?.json")):
    k = json.load(open(jf)); P = [p for p in k["parts"] if p["cls"] >= 0]; T = {t["text"]: t for t in k["texts"]}
    if len(P) < 30: continue
    med = float(np.median([max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]) for p in P])); ww, wh = 16*med, 11*med
    for x in np.arange(0, k["W"]-ww, 2*med):
        for y in np.arange(0, k["H"]-wh, 2*med):
            w = (x, y, x+ww, y+wh); pi = [p for p in P if inside(p["box"], w)]
            ok = [p for p in pi if p["ref"] in T and inside(T[p["ref"]]["box"], w)]
            if 9 <= len(pi) <= 22 and len(ok) >= len(pi)-1 and (bestk is None or len(ok) > bestk[0]): bestk = (len(ok), jf, w)
_, jf, w = bestk; k = json.load(open(jf)); T = {t["text"]: t for t in k["texts"]}
kim = Image.open(jf.with_suffix(".png")).convert("RGB").crop(tuple(int(v) for v in w)); S2 = 960/kim.width; kim = kim.resize((960, int(kim.height*S2)), Image.LANCZOS)
d = ImageDraw.Draw(kim); trk = lambda b: [(b[0]-w[0])*S2, (b[1]-w[1])*S2, (b[2]-w[0])*S2, (b[3]-w[1])*S2]
for t in k["texts"]:
    if inside(t["box"], w): d.rectangle(trk(t["box"]), outline=ORANGE, width=3)
for p in k["parts"]:
    if p["cls"] >= 0 and inside(p["box"], w):
        d.rectangle(trk(p["box"]), outline=CYAN, width=3)
        if p["ref"] in T and inside(T[p["ref"]]["box"], w):
            a = trk(p["box"]); b = trk(T[p["ref"]]["box"]); line(d, ((a[0]+a[2])/2, (a[1]+a[3])/2), ((b[0]+b[2])/2, (b[1]+b[3])/2), YELLOW)
kim.save(OUT/"kicad.jpg", quality=88)
print("kicad", k["id"], k["name"], [round(v) for v in w])
json.dump(dict(wacv=ST, window=[round(v) for v in WIN], wrong=info[0], linkable=info[1], no_label=info[2], token_des=parts[i0]["des"],
               kicad=k["id"], kicad_name=k["name"]), open(OUT/"figs.json", "w"), indent=1)
