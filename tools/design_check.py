#!/usr/bin/env python3
"""Check one WACV board's link labels against the board's real design (centroid / pick-and-place file).
  1. read the design: a centroid file (Allegro "Placed Component Report": REFDES, SYM_X, SYM_Y, SYM_MIRROR), an Eagle XML .brd
     (element name / x / y / rot "M…" = bottom; through-hole packages on both sides) or an IPC-D-356 netlist (part centre = mean of its pads; side from the SMD pads'
     access layer, through-hole-only parts on both sides); keep the side shown in the photo
  2. fit a homography design -> photo with RANSAC on parts whose annotated designator equals a centroid REFDES
     (annotation errors are outliers), refine on inliers
  3. keep every part whose annotated designator is in the design within --sure part sizes of its box; then match the rest
     one-to-one (Hungarian on centre distance), in two passes: (a) all parts,
     distance <= gate x median part size; (b) parts still unmatched, distance <= gate x their own size (big connectors have their
     design origin at pin 1, not the centre), only to unmatched design parts whose designator prefix fits the part's class -> true designator
  4. compare with the annotation: agree / wrong (annotated designator is not the design's) / missed (no designator, but a text box
     with the design designator exists) / unprinted (no designator and no such text box: consistent) / no_design_part;
     a disagreement whose match is farther than --sure part sizes is "uncertain" (often a different board revision), not an error -
     except for big connectors / switches (>= 2 median sizes) whose design origin (pin 1) lies on the box
Writes results/linking/design_check/<image>.json (per part) and prints a summary + the review candidates on that board.
Usage: design_check.py --image ATTIOT_Bottom --centroid designfiles/FRDM-K64F/raw/CEN-28163_E.smt [--side top|bottom]
       design_check.py --image ArduinoMega_Top --centroid designfiles/ArduinoMega2560/raw/Arduino_MEGA_2560-Rev3.brd
(python: ~/venvs/gdino_fewshot)"""
import argparse, csv, json, os, re, collections, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
FITS = {"resistor": ("R", "RN", "RA", "RP"), "capacitor": ("C",), "inductor": ("L", "FB"), "connector": ("J", "JP", "P", "CN", "CON", "USB", "X"),
        "diode": ("D", "LED", "CR"), "switch": ("SW", "S"), "transistor": ("Q",), "IC": ("U", "IC"), "oscillator": ("Y", "X", "OSC")}
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))
ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--centroid", required=True)
ap.add_argument("--side", default="top"); ap.add_argument("--gate", type=float, default=1.2, help="max match distance in part sizes")
ap.add_argument("--sure", type=float, default=0.3, help="a disagreement counts as an error only within this distance (part sizes)")
a = ap.parse_args()

def load_design(path):
    """-> [(refdes, x, y, bottom)] in design units, y up"""
    if path.suffix.lower() == ".brd" and open(path, "rb").read(5) == b"<?xml":                       # Eagle 6+ board
        root = ET.parse(path).getroot(); th = set()
        for lib in root.iter("library"):
            for pk in lib.iter("package"):
                if pk.find("pad") is not None: th.add((lib.get("name"), pk.get("name")))
        out = []
        for e in root.iter("element"):
            bot = e.get("rot", "").startswith("M"); r = (e.get("name"), float(e.get("x")), float(e.get("y")))
            out.append((*r, bot))
            if (e.get("library"), e.get("package")) in th: out.append((*r, not bot))
        return out
    rows = open(path, encoding="latin-1").read().splitlines()
    if any(l[:3] in ("317", "327") for l in rows):                                                         # IPC-D-356
        pads = collections.defaultdict(list)
        for l in rows:
            m = l[:3] in ("317", "327") and re.search(r"A(\d\d)X([+-]\d+)Y([+-]\d+)", l)
            if m and l[20:26].strip() and l[20:26].strip() != "VIA": pads[l[20:26].strip()].append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        top_layer, bot_layer = 1, max(a for v in pads.values() for a, _, _ in v)
        out = []
        for r, v in pads.items():
            x, y = np.mean([p[1] for p in v]), np.mean([p[2] for p in v]); acc = {p[0] for p in v}
            if top_layer in acc or acc == {0}: out.append((r, x, y, False))
            if bot_layer in acc or acc == {0}: out.append((r, x, y, True))
        return out
    hdr = next(i for i, l in enumerate(rows) if l.startswith("REFDES,"))
    return [(r["REFDES"].strip(), float(r["SYM_X"]), float(r["SYM_Y"]), r["SYM_MIRROR"].strip() == "YES") for r in csv.DictReader(rows[hdr:])]
cen = [d for d in load_design(ROOT/a.centroid) if d[3] == (a.side == "bottom")]
ref = [d[0] for d in cen]
D = np.array([[d[1], -d[2]] for d in cen])                                       # design y up -> image-like y down
if a.side == "bottom": D[:, 0] *= -1                                             # bottom is seen mirrored

W = {b["id"]: b for b in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv.json"))}; b = W[a.image]
P, T = b["parts"], b["texts"]
ctr = lambda bx: ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
C = np.array([ctr(p["box"]) for p in P]); S = np.array([max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]) for p in P], float); med = float(np.median(S))
ri = {r: i for i, r in enumerate(ref)}
pairs = [(i, ri[p["des"]]) for i, p in enumerate(P) if p["des"] in ri]
if len(pairs) < 6: raise SystemExit(f"{a.image}: only {len(pairs)} annotated designators match the design ({a.side}) - cannot align")
src = D[[j for _, j in pairs]].astype(np.float32); dst = C[[i for i, _ in pairs]].astype(np.float32)
H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=1.0*med, maxIters=20000, confidence=0.999)
inl = inl.ravel().astype(bool)
H, _ = cv2.findHomography(src[inl], dst[inl], 0)                                  # refine on inliers
Dp = cv2.perspectiveTransform(D[None].astype(np.float32), H)[0]
res = np.linalg.norm(Dp[[j for _, j in pairs]][inl] - C[[i for i, _ in pairs]][inl], axis=1)
print(f"{a.image}: {len(P)} annotated parts, {len(ref)} design parts ({a.side}); {len(pairs)} name pairs, {inl.sum()} RANSAC inliers, "
      f"median residual {np.median(res):.1f} px = {np.median(res)/med:.2f} part sizes (median part {med:.0f} px)")

dist = np.linalg.norm(C[:, None] - Dp[None], axis=2)
c1 = dist / med; cost = c1.copy()
# name-consistent parts first: the annotated designator exists in the design and sits on this box -> keep it (otherwise two
# design parts at one spot, e.g. top and bottom of the board, could be swapped by the assignment)
match = {i: j for i, j in pairs if c1[i, j] <= a.sure}
fi, fj = [i for i in range(len(P)) if i not in match], [j for j in range(len(ref)) if j not in set(match.values())]
if fi and fj:
    sub = c1[np.ix_(fi, fj)]; r_, c_ = linear_sum_assignment(np.where(sub <= a.gate, sub, 1e6))
    match.update({fi[x]: fj[y] for x, y in zip(r_, c_) if sub[x, y] <= a.gate})
li, lj = [i for i in range(len(P)) if i not in match], [j for j in range(len(ref)) if j not in set(match.values())]
if li and lj:                                                                     # pass (b): big parts, own size, class must fit
    fits = np.array([[re.match(r"[A-Z]*", ref[j]).group() in FITS[CLS[P[i]["cls"]]] for j in lj] for i in li])
    c2 = dist[np.ix_(li, lj)] / S[li][:, None]; c2 = np.where(fits & (c2 <= a.gate), c2, 1e6)
    for x, y in zip(*linear_sum_assignment(c2)):
        if c2[x, y] < 1e6: match[li[x]] = lj[y]; cost[li[x], lj[y]] = c2[x, y]
texts = collections.Counter(t["text"] for t in T)
out, tally = [], collections.Counter()
for i, p in enumerate(P):
    true = ref[match[i]] if i in match else None; ann = p["des"] or ""
    st = "no_design_part" if true is None else ("agree" if ann == true else ("wrong" if ann else ("missed" if texts[true] else "unprinted")))
    if st in ("wrong", "missed") and cost[i, match[i]] > a.sure:          # far from the centre: sure only if the design origin
        bx = p["box"]; mx, my = 0.1 * (bx[2] - bx[0]), 0.1 * (bx[3] - bx[1]); qx, qy = Dp[match[i]]   # (e.g. pin 1) lies on the box
        big = S[i] >= 2 * med and CLS[p["cls"]] in ("connector", "switch")   # headers / connectors: origin at pin 1
        if not (big and bx[0] - mx <= qx <= bx[2] + mx and bx[1] - my <= qy <= bx[3] + my): st = "uncertain"
    tally[st] += 1
    out.append(dict(part=i, annotated=ann, design=true, status=st, dist=round(float(cost[i, match[i]]), 2) if i in match else None,
                    text_exists=bool(true and texts[true]), box=p["box"]))
tally_txt = collections.Counter(o["text_exists"] for o in out if o["design"])
print("parts:", dict(tally), "| true designator printed as a text box:", tally_txt[True], "yes /", tally_txt[False], "no")
unused = sorted(set(ref) - {ref[j] for j in match.values()})
print(f"design parts with no annotated box nearby: {len(unused)}", " ".join(unused[:40]), "…" if len(unused) > 40 else "")
for o in out:
    if o["status"] in ("wrong", "missed"): print(f"  part {o['part']:3d}: annotated {o['annotated'] or '-':6s} design {o['design']:6s} (d={o['dist']}) text box '{o['design']}' {'exists' if o['text_exists'] else 'MISSING'}")
cands = [c for c in json.load(open(ROOT/"page/review/cands.json")) if c["image"] == a.image]
if cands:
    print("review candidates on this board:")
    for c in cands:
        print(f"  {c['id']:28s} " + " | ".join(f"part {i}: annotated {out[i]['annotated'] or '-'} -> design {out[i]['design']}" for i in c["parts"])
              + (f"   (label offered: {c.get('printed')})" if c["type"] != "A" else f"   (shared label {c['designator']})"))
od = ROOT/"results/linking/design_check"; od.mkdir(parents=True, exist_ok=True)
json.dump(dict(image=a.image, centroid=a.centroid, side=a.side, H=H.tolist(), inliers=int(inl.sum()), median_residual_parts=round(float(np.median(res)/med), 3),
               med=round(med, 1), design_pts=[[r, round(float(x), 1), round(float(y), 1)] for r, (x, y) in zip(ref, Dp)],
               tally=dict(tally), parts=out, design_unmatched=unused), open(od/f"{a.image}.json", "w"), indent=0)
