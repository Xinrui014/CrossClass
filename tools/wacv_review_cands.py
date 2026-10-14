#!/usr/bin/env python3
"""Candidates for cleaning the WACV link annotation (WACV-clean v1), with one crop each, for the owner's review page (page/review/).
  A  split / shared designator: more annotated parts carry a designator than it is printed on the board -> are these boxes one part?
  B  typo: a part's designator is not printed, but a label 1 character away with the same prefix, naming no part, sits within 3 part sizes
  C  missing designator: a part without designator; the nearest label naming no part, whose prefix fits the part's class, within 2 part sizes
     (each free label offered to its nearest such part only)
Writes page/review/cands.json + page/review/img/<id>.jpg crops and results/linking/wacv_clean_candidates.json (same list, no images),
plus the full photo of every board with a candidate (page/review/boards/<image>.jpg) and its boxes (page/review/boards.json) for the
page's zoomable viewer (boxes, part designators pd / classes pc, text strings tt); cands.json "parts" / "labels" index into those lists.
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
photo = lambda b: next(f for f in sorted((RAW/b["id"]).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
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
    img = Image.open(photo(b)).convert("RGB")
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
# M = manual candidates the owner found while reviewing (part, the label text to suggest or None, what to do)
EXTRA = [("Spartan6Redux_Top", 68, None, "这个大 IC（FPGA）在标注里写成 U5；板上印的是 FPGA，但没有文字框。按 T 框住照片上的 FPGA 字样，按 Enter。"),
         ("Spartan6Redux_Top", 52, "U5", "这个小三脚 IC 在标注里写成 Q1；丝印图上它是 U5（U5 竖着印在左边）。")]
for img, i, lab, note in EXTRA:
    b = next(x for x in W if x["id"] == img); P, Tx = b["parts"], b["texts"]
    js = [j for j, t in enumerate(Tx) if lab and t["text"] == lab]
    cands.append(dict(id=f"M_{img}_{i}", type="M", image=img, dev=img in DEV, parts=[i], labels=js[:1], designator=P[i]["des"] or "",
                      printed=lab or "", classes=[CLS[P[i]["cls"]]], note=note, crop=crop(Image.open(photo(b)).convert("RGB"), b, [i], js[:1])))
cands.sort(key=lambda c: (c["image"], c["type"], c["id"]))
DROPPED = {"ML365_Bottom"}                    # owner 2026-10-13: no design file, too dense to clean by eye -> left out of wacv_clean_v1
SURE = {"agree", "wrong", "missed", "unprinted"}                                        # tools/design_check.py statuses it is sure of
for c in cands:                                         # settled when the board's design file answers every part of the candidate
    f = ROOT/"results/linking/design_check"/f"{c['image']}.json"
    dp = json.load(open(f))["parts"] if f.exists() else None
    c["by_design"] = bool(dp) and all(dp[i]["status"] in SURE for i in c["parts"])
    c["dropped"] = c["image"] in DROPPED
    if dp: c["design"] = [dp[i]["design"] for i in c["parts"]]
IMG = ROOT/"page/review/img"; IMG.mkdir(parents=True, exist_ok=True)
for c in cands: c["crop"].save(IMG/f"{c['id']}.jpg", quality=82); c["img"] = f"img/{c['id']}.jpg"; del c["crop"]
json.dump(cands, open(ROOT/"page/review/cands.json", "w"))
json.dump(cands, open(ROOT/"results/linking/wacv_clean_candidates.json", "w"), indent=0)
DC = {f.stem: json.load(open(f)) for f in sorted((ROOT/"results/linking/design_check").glob("*.json"))}   # tools/design_check.py results
BD = ROOT/"page/review/boards"; BD.mkdir(parents=True, exist_ok=True); boards = {}                # full photos for the viewer
for b in W:
    if not any(c["image"] == b["id"] for c in cands) and b["id"] not in DC: continue
    im = Image.open(photo(b)).convert("RGB"); im.save(BD/f"{b['id']}.jpg", quality=88)
    R = lambda bx: [round(v) for v in bx]
    boards[b["id"]] = dict(w=im.width, h=im.height, src=f"boards/{b['id']}.jpg", med=round(float(np.median([size(p["box"]) for p in b["parts"]]))),
                           parts=[R(p["box"]) for p in b["parts"]], texts=[R(t["box"]) for t in b["texts"]],
                           pd=[p["des"] or "" for p in b["parts"]], pc=[CLS[p["cls"]] for p in b["parts"]], tt=[t["text"] for t in b["texts"]])
json.dump(boards, open(ROOT/"page/review/boards.json", "w"))

# design-file check tab: every part the check calls wrong / missed / uncertain, plus every candidate part on a checked board,
# with a crop showing the design's designators (yellow), the annotated label (red) and the design's label (green)
from PIL import ImageFont
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
STATUS = {"wrong", "missed", "uncertain"}
def near_text(b, i, s):
    js = [j for j, t in enumerate(b["texts"]) if s and t["text"] == s]
    return min(js, key=lambda j: dist(b["parts"][i]["box"], b["texts"][j]["box"])) if js else -1
def dcrop(img, b, i, pts, aj, dj, out=380):
    bx = b["parts"][i]["box"]; med = float(np.median([size(p["box"]) for p in b["parts"]]))
    half = max(size(bx), med) * 2.4; cx, cy = ctr(bx); x0, y0 = cx - half, cy - half; s = out / (2 * half)
    im = img.crop((int(x0), int(y0), int(x0 + 2 * half), int(y0 + 2 * half))).resize((out, out), Image.LANCZOS); d = ImageDraw.Draw(im)
    T = lambda q: [(q[0] - x0) * s, (q[1] - y0) * s, (q[2] - x0) * s, (q[3] - y0) * s]
    for t in b["texts"]: d.rectangle(T(t["box"]), outline=ORANGE, width=1)
    if aj >= 0: d.rectangle(T(b["texts"][aj]["box"]), outline=(239, 68, 68), width=3)
    if dj >= 0: d.rectangle(T(b["texts"][dj]["box"]), outline=(74, 222, 128), width=3)
    d.rectangle(T(bx), outline=MAG, width=3)
    for r, x, y in pts:
        if x0 < x < x0 + 2 * half and y0 < y < y0 + 2 * half:
            X, Y = (x - x0) * s, (y - y0) * s; d.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=YELLOW)
            d.text((X + 5, Y - 7), r, fill=YELLOW, font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    return im
design, ncrop = {}, 0
for b in W:
    if b["id"] not in DC: continue
    r = DC[b["id"]]; cp = {i: c["id"] for c in cands if c["image"] == b["id"] for i in c["parts"]}
    img = Image.open(photo(b)).convert("RGB"); items = []
    for o in r["parts"]:
        i = o["part"]
        if o["status"] not in STATUS and i not in cp: continue
        aj, dj = near_text(b, i, o["annotated"]), near_text(b, i, o["design"])
        iid = f"D_{b['id']}_{i}"; dcrop(img, b, i, r["design_pts"], aj, dj).save(IMG/f"{iid}.jpg", quality=82); ncrop += 1
        items.append(dict(id=iid, image=b["id"], part=i, cls=CLS[b["parts"][i]["cls"]], annotated=o["annotated"], design=o["design"] or "",
                          status=o["status"], dist=o["dist"], cand=cp.get(i, ""), ann_text=aj, des_text=dj, img=f"img/{iid}.jpg"))
    design[b["id"]] = dict(source=r["centroid"].split("/")[1], side=r["side"], residual=r["median_residual_parts"], med=r["med"],
                           tally={k: v for k, v in r["tally"].items()}, pts=r["design_pts"], items=items, dev=b["id"] in DEV)
design = dict(sorted(design.items(), key=lambda kv: (-len(kv[1]["items"]), kv[0])))       # boards with most items first
json.dump(design, open(ROOT/"page/review/design.json", "w"))
print("design tab:", len(design), "images,", ncrop, "items")
print("boards", len(boards), "| photos MB %.0f" % (sum(f.stat().st_size for f in BD.iterdir())/1e6))
cnt = collections.Counter(c["type"] for c in cands); print("candidates", len(cands), dict(cnt), "| images", len({c["image"] for c in cands}),
      "| size MB %.1f" % ((ROOT/"page/review/cands.json").stat().st_size/1e6))
