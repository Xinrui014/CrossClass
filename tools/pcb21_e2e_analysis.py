#!/usr/bin/env python3
"""Where the 21PCB gain comes from, and the remaining errors (for the results report). Per GT part (class >= 0) of the 28 photos, the
found detection = highest IoU detector box with score >= 0.25 and IoU >= 0.5. Its top-1 class under: the detector (argmax of the 9
scores), hard pipeline, soft cascade C, B class head, B mix ((head + C) / 2) — the same per-box P as tools/pcb21_e2e_score.py.
Outputs results/pcb21/e2e_analysis.json: per method accuracy on found parts and confusion matrix; detector -> B mix transitions
(fixed / broken / both wrong) by GT class; the B mix errors with what drove them (strongest link and its PARSeq reading, head
class, C class, detector class) and crop sheets page/report21/img/err_<k>.jpg / fix_<k>.jpg (part box magenta, linked text box cyan).
  pcb21_e2e_analysis.py   (env ~/venvs/gdino_fewshot, CPU)"""
import json, os, sys, collections, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = E/"21pcb/e2e"; OUT = ROOT/"page/report21/img"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import pfx_class, PFX
sys.path.insert(0, str(E)); from analyze9 import iou
NM = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
P9 = json.load(open(LD/"preds_cls9.json")); G = json.load(open(E/"gt9_21pcb.json")); GD = json.load(open(E/"21pcb/final_v1/gt9_21pcb.json"))
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; BI = {b["id"]: b for b in json.load(open(LD/"e2e_boards_infil.json"))}
Z = np.load(LD/"e2e_tf_all.npz"); HARD = json.load(open(LD/"links_noise05_infil_raw.json")); SOFT = json.load(open(LD/"soft_noise05_infil.json"))
BO = json.load(open(LD/"b_drop03_infil.json")); CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
read = lambda st, j: "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
def q_soft(st, j):
    Q = Z[st][j][:304].astype(np.float64).reshape(8, 38); Q = Q / Q.sum(1, keepdims=True); dig = Q[:, 1:11].sum(1); out = np.zeros(10)
    for p, c in PFX:
        if len(p) < 8: out[c] += np.prod([Q[k, 11 + ord(ch) - 65] for k, ch in enumerate(p)]) * dig[len(p)]
    out[:9] = np.minimum(out[:9], 1.0); out[9] = max(0.0, 1.0 - out[:9].sum()); return out / out.sum()
def probs(st, k):
    dd = P9[st][k]; d = np.array(dd[6:15]); d = d / max(d.sum(), 1e-9); i = {kk: ii for ii, kk in enumerate(B[st]["det_index"])}.get(k)
    P = dict(det=d, hard=d.copy(), C=d.copy(), head=d.copy(), mix=d.copy()); info = {}
    if i is None: return P, info
    h = HARD[st].get(str(i))
    if h and h[0] is not None and h[1] >= 0.5:
        jo = BI[st]["texts"][h[0]]["orig"]; r = read(st, jo); c = pfx_class(r); info["hard"] = (jo, r, h[1])
        if c is not None: P["hard"] = (1 - h[1]) * d; P["hard"][c] += h[1]
    s = SOFT[st].get(str(i))
    if s:
        C = (1.0 - sum(p for _, p in s["texts"])) * d
        for j, p in s["texts"]: q = q_soft(st, j); C = C + p * (q[:9] + q[9] * d)
        P["C"] = C
    r = BO[st].get(str(i))
    if r:
        H = np.array(r["cls"]); Cd = (1.0 - sum(p for _, p in r["texts"])) * d
        for j, p in r["texts"]: q = q_soft(st, j); Cd = Cd + p * (q[:9] + q[9] * d)
        P["head"] = H; P["mix"] = (H + Cd) / 2; P["BC"] = Cd
        if r["texts"]: j, p = max(r["texts"], key=lambda x: x[1]); info["blink"] = (j, read(st, j), p)
        info["bnone"] = r["none"]
    return P, info

M = ["det", "hard", "C", "head", "mix"]; conf = {m: np.zeros((9, 9), int) for m in M}; trans = collections.defaultdict(collections.Counter)
errs, fixes = [], []
for st in sorted(G):
    dets = [d for d in P9[st]]; db = np.array([d[:4] for d in dets]).reshape(-1, 4); ok = np.array([d[4] >= 0.25 for d in dets])
    for gi, g in enumerate(G[st]["boxes"]):
        if g[4] < 0 or not len(db): continue
        I = iou(np.array([g[:4]]), db)[0]; I[~ok] = 0; k = int(I.argmax())
        if I[k] < 0.5: continue
        P, info = probs(st, k); pred = {m: int(np.argmax(P[m])) for m in M}
        for m in M: conf[m][g[4], pred[m]] += 1
        a, b = pred["det"] == g[4], pred["mix"] == g[4]
        trans[NM[g[4]]]["fixed" if (not a and b) else "broken" if (a and not b) else "both wrong" if not b else "both right"] += 1
        rec = dict(st=st, gt=NM[g[4]], des=GD[st]["boxes"][gi][6], det=NM[pred["det"]], mix=NM[pred["mix"]], head=NM[int(np.argmax(P["head"]))],
                   C=NM[int(np.argmax(P.get("BC", P["C"])))], box=dets[k][:4], link=info.get("blink"), none=round(info.get("bnone", 1.0), 2))
        if not b: errs.append(rec)
        elif not a: fixes.append(rec)

def why(e):
    if e["link"] is None or e["none"] >= 0.5: return "no label linked (head/detector decided)"
    return "linked text read '%s'" % e["link"][1]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17); small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
def sheet(items, path, title_fn):
    T = 300; cols = 4; S = Image.new("RGB", (cols*T, T*((len(items)+cols-1)//cols)), (20, 22, 26)); cache = {}
    for n, e in enumerate(items):
        st = e["st"]
        if st not in cache: cache.clear(); cache[st] = Image.open(E/"21pcb/images"/f"{st}.jpg").convert("RGB")
        im = cache[st]; b = e["box"]; bs = [b]
        if e["link"]: bs.append(B[st]["texts"][e["link"][0]]["box"])
        x0 = min(x[0] for x in bs); y0 = min(x[1] for x in bs); x1 = max(x[2] for x in bs); y1 = max(x[3] for x in bs)
        s = max(x1-x0, y1-y0) * 1.6 + 150; cx, cy = (x0+x1)/2, (y0+y1)/2; X0, Y0 = cx-s/2, cy-s/2
        c = im.crop((int(X0), int(Y0), int(X0+s), int(Y0+s))); z = (T-62)/s; c = c.resize((T-62, T-62)); d = ImageDraw.Draw(c)
        R = lambda q: [(q[0]-X0)*z, (q[1]-Y0)*z, (q[2]-X0)*z, (q[3]-Y0)*z]
        d.rectangle(R(b), outline=(255, 0, 255), width=3)
        if e["link"]: d.rectangle(R(bs[1]), outline=(0, 230, 255), width=2)
        t = Image.new("RGB", (T, T), (20, 22, 26)); t.paste(c, (31, 52)); dt = ImageDraw.Draw(t); l1, l2 = title_fn(e)
        dt.text((4, 2), l1, fill=(255, 255, 255), font=font); dt.text((4, 26), l2, fill=(255, 210, 90), font=small)
        S.paste(t, ((n % cols)*T, (n//cols)*T))
    S.save(path, quality=84)
random.seed(0)
# errors: up to 2 per (gt, mix) confusion pair, most frequent pairs first
pairs = collections.Counter((e["gt"], e["mix"]) for e in errs); pick = []
for (g, p), _ in pairs.most_common():
    pick += random.sample([e for e in errs if (e["gt"], e["mix"]) == (g, p)], min(2, pairs[(g, p)]))
pick = pick[:24]
sheet(pick, OUT/"errors.jpg", lambda e: (f"GT {e['gt']}  ->  pred {e['mix']}", f"{e['st'].split('_')[0]}{e['st'][-2:]} {e['des'] or ''}  {why(e)}"))
fx = random.sample(fixes, min(16, len(fixes)))
sheet(fx, OUT/"fixes.jpg", lambda e: (f"detector {e['det']} -> {e['mix']}", f"{e['st'].split('_')[0]}{e['st'][-2:]} {e['des'] or ''}  {why(e)}"))
res = dict(acc={m: float(np.trace(conf[m]) / conf[m].sum()) for m in M}, n_found=int(conf["det"].sum()), confusion={m: conf[m].tolist() for m in M},
           transitions={k: dict(v) for k, v in trans.items()}, error_pairs=[[g, p, n] for (g, p), n in pairs.most_common()],
           error_causes=dict(collections.Counter("no link" if (e["link"] is None or e["none"] >= 0.5) else "linked" for e in errs)),
           n_err=len(errs), n_fix=len(fixes))
json.dump(res, open(ROOT/"results/pcb21/e2e_analysis.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("acc", "n_found", "n_err", "n_fix", "error_causes")}, indent=0))
print("transitions", json.dumps(res["transitions"])); print("top error pairs", res["error_pairs"][:12])
