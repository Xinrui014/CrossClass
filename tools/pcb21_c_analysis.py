#!/usr/bin/env python3
"""Analysis of the soft cascade C on 21PCB (the method that shows the LinkFormer + OCR part most directly). C = soft links of
v2d2_I_noise05_100k on the rule boxes x PARSeq prefix probabilities, detector scores only as the fallback (no link / no prefix).
Per GT part found by the detector (IoU >= 0.5, score >= 0.25), top-1 class of the detector and of C (one class per box).
Error causes for C, checked against the owner's link GT (data/21pcb/link_gt_v1.json, 2026-10-21; before that a prefix comparison
with the GT designator was used): the part's GT label box vs the OCR box of its strongest link (same box = IoU >= 0.3 or a centre
inside the other box); link = strongest link with link mass sum_j a_ij >= 0.5.
  no printed label            : the part has no label on the board (GT none); C keeps the detector class, or links a wrong text
  label missed by OCR         : the part has a label, but no OCR box covers it, so it cannot be linked
  label found, linker said none : an OCR box covers the label, but the link mass is < 0.5 -> detector fallback
  wrong text linked           : the strongest link goes to another box than the part's label
  right label, OCR misread    : the right box is linked, but PARSeq's reading has no known prefix (digits first, garbage)
  right label, prefix not in table : the right box is linked and read as a designator whose prefix the table does not know (B, F, TVS ...)
  right label, table class differs : the right box is linked and read, but the prefix table maps it to another class (T, JP ...)
Outputs results/pcb21/c_analysis.json and page/report21/img/c_err.jpg, c_fix.jpg (part box magenta, linked text box cyan).
  pcb21_c_analysis.py   (env ~/venvs/gdino_fewshot, CPU)"""
import json, os, re, sys, collections, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = E/"21pcb/e2e"; OUT = ROOT/"page/report21/img"
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class, PFX
os.chdir(E); from analyze9 import iou
NM = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
P9 = json.load(open(LD/"preds_cls9.json")); G = json.load(open(E/"gt9_21pcb.json")); GD = json.load(open(E/"21pcb/final_v1/gt9_21pcb.json"))
LG = json.load(open(E/"21pcb/link_gt_v1.json"))       # parts in the same order as the class GT boxes (tools/pcb21_sync_class_gt.py)
def same_box(a, b):
    ca, cb = ((a[0]+a[2])/2, (a[1]+a[3])/2), ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return float(iou(np.array([a]), np.array([b]))[0][0]) >= 0.3 or (b[0] <= ca[0] <= b[2] and b[1] <= ca[1] <= b[3]) or (a[0] <= cb[0] <= a[2] and a[1] <= cb[1] <= a[3])
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Z = np.load(LD/"e2e_tf_all.npz"); SOFT = json.load(open(LD/"soft_noise05_infil.json"))
CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
read = lambda st, j: "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
def q_soft(st, j):
    Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
    for p, c in PFX:
        if len(p) < 8: out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
    out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); return out / out.sum()
pre = lambda s: (re.match(r"[A-Z]+", s or "") or [""])[0]
conf = {m: np.zeros((9, 9), int) for m in ("det", "C")}; cause = collections.Counter(); errs, fixes = [], []
cause_by_cls = collections.defaultdict(collections.Counter)
for st in sorted(G):
    dets = P9[st]; db = np.array([d[:4] for d in dets]); ok = np.array([d[4] >= 0.25 for d in dets]); dpos = {k: i for i, k in enumerate(B[st]["det_index"])}
    for gi, g in enumerate(G[st]["boxes"]):
        if g[4] < 0: continue
        I = iou(np.array([g[:4]]), db)[0]; I[~ok] = 0; k = int(I.argmax())
        if I[k] < 0.5: continue
        d = np.array(dets[k][6:15]); d = d / d.sum(); P = d; r = SOFT[st].get(str(dpos.get(k))); link = None; mass = 0.0
        if r:
            mass = sum(p for _, p in r["texts"]); P = (1 - mass) * d
            for j, p in r["texts"]: q = q_soft(st, j); P = P + p * (q[:9] + q[9] * d)
            if r["texts"]: j, p = max(r["texts"], key=lambda x: x[1]); link = (j, read(st, j), p)
        pd, pc = int(d.argmax()), int(P.argmax()); conf["det"][g[4], pd] += 1; conf["C"][g[4], pc] += 1
        des = GD[st]["boxes"][gi][6]
        rec = dict(st=st, gt=NM[g[4]], des=des, det=NM[pd], C=NM[pc], box=dets[k][:4], link=link, mass=round(mass, 2))
        if pc != g[4]:
            lp = LG[st]["parts"][gi]; gtb = LG[st]["texts"][lp["text"]]["box"] if lp["text"] >= 0 else None
            linked = link is not None and mass >= 0.5
            if gtb is None: c = "no printed label"
            elif linked and same_box(B[st]["texts"][link[0]]["box"], gtb):
                if pfx_class(link[1]) is None: c = "right label, prefix not in table" if re.match(r"[A-Z]{1,4}\d", link[1]) else "right label, OCR misread"
                else: c = "right label, table class differs"
            elif linked: c = "wrong text linked"
            elif any(same_box(t["box"], gtb) for t in B[st]["texts"]): c = "label found, linker said none"
            else: c = "label missed by OCR"
            rec["cause"] = c; cause[c] += 1; cause_by_cls[NM[g[4]]][c] += 1; errs.append(rec)
        elif pd != g[4]: fixes.append(rec)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17); small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
def sheet(items, path, f):
    T = 300; cols = 4; S = Image.new("RGB", (cols*T, T*((len(items)+cols-1)//cols)), (20, 22, 26)); cache = {}
    for n, e in enumerate(items):
        st = e["st"]
        if st not in cache: cache.clear(); cache[st] = Image.open(E/"21pcb/images"/f"{st}.jpg").convert("RGB")
        im = cache[st]; b = e["box"]; bs = [b] + ([B[st]["texts"][e["link"][0]]["box"]] if e["link"] and e["mass"] >= 0.5 else [])
        x0 = min(x[0] for x in bs); y0 = min(x[1] for x in bs); x1 = max(x[2] for x in bs); y1 = max(x[3] for x in bs)
        s = max(x1-x0, y1-y0) * 1.6 + 150; cx, cy = (x0+x1)/2, (y0+y1)/2; X0, Y0 = cx-s/2, cy-s/2
        c = im.crop((int(X0), int(Y0), int(X0+s), int(Y0+s))); z = (T-62)/s; c = c.resize((T-62, T-62)); dr = ImageDraw.Draw(c)
        R = lambda q: [(q[0]-X0)*z, (q[1]-Y0)*z, (q[2]-X0)*z, (q[3]-Y0)*z]; dr.rectangle(R(b), outline=(255, 0, 255), width=3)
        if len(bs) > 1: dr.rectangle(R(bs[1]), outline=(0, 230, 255), width=2)
        t = Image.new("RGB", (T, T), (20, 22, 26)); t.paste(c, (31, 52)); dt = ImageDraw.Draw(t); l1, l2 = f(e)
        dt.text((4, 2), l1, fill=(255, 255, 255), font=font); dt.text((4, 26), l2, fill=(255, 210, 90), font=small); S.paste(t, ((n % cols)*T, (n//cols)*T))
    S.save(path, quality=84)
random.seed(1); pick = []
for c in [c for c, _ in cause.most_common()]:
    pool = [e for e in errs if e["cause"] == c]; pick += random.sample(pool, min(4, len(pool)))
tag = lambda e: f"{e['st'].split('_')[0]}{e['st'][-2:]} {e['des'] or '-'}"
sheet(pick, OUT/"c_err.jpg", lambda e: (f"GT {e['gt']} -> C {e['C']}", f"{tag(e)}  " + (e["cause"].split(" (")[0] + (f": read '{e['link'][1]}'" if e["link"] and e["mass"] >= 0.5 else ""))))
fx = random.sample(fixes, min(12, len(fixes)))
sheet(fx, OUT/"c_fix.jpg", lambda e: (f"detector {e['det']} -> C {e['C']}", f"{tag(e)}  read '{e['link'][1]}'" if e["link"] else tag(e)))
acc = {m: float(np.trace(v)/v.sum()) for m, v in conf.items()}
per = {m: {NM[i]: round(100*v[i, i]/max(v[i].sum(), 1), 1) for i in range(9)} for m, v in conf.items()}
res = dict(n_found=int(conf["det"].sum()), acc=acc, per_class=per, n_err=len(errs), n_fix=len(fixes), causes=dict(cause.most_common()),
           causes_by_class={k: dict(v) for k, v in cause_by_cls.items()}, sheet_causes=[e["cause"] for e in pick],
           broken=sum(1 for e in errs if e["det"] == e["gt"]))
json.dump(res, open(ROOT/"results/pcb21/c_analysis.json", "w"), indent=1); print(json.dumps(res, indent=1))
