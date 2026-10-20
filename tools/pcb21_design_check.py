#!/usr/bin/env python3
"""Check and correct the 21PCB pre-labels of every board that has design placement data (no learned model involved).
Input: designfiles/21pcb/placement/<Card>.json (tools/pcb21_placement.py: parts [{ref, x, y, side, box, ...}], design frame, y up)
and data/21pcb/prelabel.json. Populated parts only (BoM rows without DNI).
Alignment per photo (bottom side: design x mirrored, as the photo looks at the bottom):
  1. name pairs: pre-label parts that carry a BoM designator (box centre), plus OCR texts that read a BoM designator (a label sits
     within about one part size of its part, so these pairs are weaker); RANSAC homography, threshold 1.5 median part sizes;
  2. two refinement rounds: project design boxes, match them one-to-one to pre-label boxes (Hungarian on centre distance, gate
     0.6 x part size), re-fit the homography on the matched centres (RANSAC, 0.3 part sizes).
Outcome: right / wrong / no designator on a matched box, design parts with no box (missed), pre-label boxes with no design part
(extra), class disagreements. A photo with fewer than 6 usable pairs is not aligned (its pre-labels stay as they are).
Writes results/pcb21/design_check_<stem>.json, results/pcb21/design_check_summary.json, data/21pcb/design_fixed.json
  pcb21_design_check.py [Card6,Card19]   (env ~/venvs/gdino_fewshot: cv2, scipy)"""
import json, os, sys, collections
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; R = ROOT/"results/pcb21"
PLC = ROOT/"designfiles/21pcb/placement"; R.mkdir(parents=True, exist_ok=True)

def to_img_like(pts, bottom):
    a = np.array(pts, float).reshape(-1, 2).copy(); a[:, 1] *= -1
    if bottom: a[:, 0] *= -1
    return a

def project(H, boxes, bottom):
    out = []
    for b in boxes:
        q = to_img_like([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], bottom)
        q = cv2.perspectiveTransform(q[None].astype(np.float32), H)[0]; out.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
    return np.array(out).reshape(-1, 4)

def match(Dbox, P, S):
    if not len(P) or not len(Dbox): return {}
    C = np.array([[(p["box"][0]+p["box"][2])/2, (p["box"][1]+p["box"][3])/2] for p in P])
    Dc = np.c_[(Dbox[:, 0]+Dbox[:, 2])/2, (Dbox[:, 1]+Dbox[:, 3])/2]; Dsz = np.maximum(Dbox[:, 2]-Dbox[:, 0], Dbox[:, 3]-Dbox[:, 1])
    dist = np.linalg.norm(Dc[:, None] - C[None], axis=2); gate = 0.6*np.maximum(Dsz[:, None], S[None]*0.5)
    cost = np.where(dist <= gate, dist/np.maximum(Dsz[:, None], 1), 1e6); m = {}
    for j, i in zip(*linear_sum_assignment(cost)):
        if cost[j, i] < 1e6: m[j] = i
    return m

PL = json.load(open(D/"prelabel.json")); only = [s for s in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if s]
fixed, summary = {}, []
for pf in sorted(PLC.glob("Card*.json")):
    plc = json.load(open(pf)); card = plc["card"]
    if only and card not in only: continue
    bom = PL["bom"].get(card, {})
    for st, v in PL["photos"].items():
        if st.split("_")[0] != card: continue
        P = v["parts"]; T = v["texts"]
        read = {p["ref"] for p in P if p["ref"]} | {t["ref"] for t in T if t["ref"]}
        side_of = lambda b: {p["ref"].upper() for p in plc["parts"] if (p["side"] == "bottom") == b}
        # the photo's side is the design side whose designators the photo shows (Card8: the file names _T / _B are swapped)
        own, other = st.endswith("_B"), not st.endswith("_B")
        n_own, n_other = len(read & side_of(own)), len(read & side_of(other))
        bottom = other if (n_other >= 10 and n_other >= 3*max(n_own, 1)) else own   # swap only on strong evidence
        swapped = bottom != st.endswith("_B")
        dp = [p for p in plc["parts"] if (p["side"] == "bottom") == bottom and p["ref"].upper() in bom and not bom[p["ref"].upper()]["dni"]]
        refs = [p["ref"].upper() for p in dp]; ri = {r: k for k, r in enumerate(refs)}
        S = np.array([max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]) for p in P]) if P else np.array([])
        med = float(np.median(S)) if len(S) else 150.0
        # design boxes: pads / body when known, else a square of the median design part size around the centre
        sizes = [max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]) for p in dp if p.get("box")]
        dmed = float(np.median(sizes)) if sizes else None
        Dbox_d = [p["box"] if p.get("box") else [p["x"]-(dmed or 1)/2, p["y"]-(dmed or 1)/2, p["x"]+(dmed or 1)/2, p["y"]+(dmed or 1)/2] for p in dp]
        Dctr = to_img_like([[p["x"], p["y"]] for p in dp], bottom) if dp else np.zeros((0, 2))
        src, dst = [], []
        for p in P:
            if p["ref"] in ri: src.append(Dctr[ri[p["ref"]]]); dst.append([(p["box"][0]+p["box"][2])/2, (p["box"][1]+p["box"][3])/2])
        npart = len(src)
        for t in T:
            if t["ref"] in ri: src.append(Dctr[ri[t["ref"]]]); dst.append([(t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2])
        info = dict(photo=st, design_side="bottom" if bottom else "top", side_swapped=swapped, design_parts=len(dp), prelabel_parts=len(P), part_pairs=npart, text_pairs=len(src)-npart)
        if len(src) < 6 or len(dp) < 4:
            info["aligned"] = False; summary.append(info); print(json.dumps(info)); continue
        src, dst = np.float32(src), np.float32(dst)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=1.5*med, maxIters=20000, confidence=0.999)
        if H is None: info["aligned"] = False; summary.append(info); print(json.dumps(info)); continue
        for _ in range(2):
            Dbox = project(H, Dbox_d, bottom); m = match(Dbox, P, S)
            if len(m) < 6: break
            a = np.float32([Dctr[j] for j in m]); b = np.float32([[(P[i]["box"][0]+P[i]["box"][2])/2, (P[i]["box"][1]+P[i]["box"][3])/2] for i in m.values()])
            H2, inl = cv2.findHomography(a, b, cv2.RANSAC, ransacReprojThreshold=0.3*med, maxIters=20000, confidence=0.999)
            if H2 is not None: H = H2
        Dbox = project(H, Dbox_d, bottom); m = match(Dbox, P, S)
        Dc = np.c_[(Dbox[:, 0]+Dbox[:, 2])/2, (Dbox[:, 1]+Dbox[:, 3])/2]
        res = [np.hypot(Dc[j][0]-(P[i]["box"][0]+P[i]["box"][2])/2, Dc[j][1]-(P[i]["box"][1]+P[i]["box"][3])/2) for j, i in m.items()]
        cnt = collections.Counter(); out_parts = []; ex = collections.defaultdict(list)
        for j, r in enumerate(refs):
            b = bom[r]
            if j not in m:
                cnt["missed"] += 1; ex["missed"].append(r); bx = Dbox[j]
                if not dp[j].get("box"):        # no pad data: a square of the photo's median part size around the design centre
                    cx, cy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2; bx = [cx-med/2, cy-med/2, cx+med/2, cy+med/2]
                out_parts.append(dict(box=[round(float(x), 1) for x in bx], cls=b["cls"], ref=r, src="design_missed",
                                      box_known=bool(dp[j].get("box")))); continue
            p = P[m[j]]; k = "right" if p["ref"] == r else ("none" if p["ref"] is None else "wrong"); cnt[k] += 1
            if k == "wrong": ex["wrong"].append((p["ref"], r))
            if p["cls"] != b["cls"]: cnt["class_wrong"] += 1; ex["class"].append(dict(ref=r, pre=p["cls"], bom=b["raw"]))
            out_parts.append(dict(box=p["box"], cls=b["cls"], ref=r, src="design_matched", pre_ref=p["ref"], pre_cls=p["cls"], pre_text=p["text"]))
        used = set(m.values())
        for i, p in enumerate(P):
            if i not in used:
                cnt["extra"] += 1; ex["extra"].append(dict(box=p["box"], ref=p["ref"], cls=p["cls"], score=p["score"]))
                out_parts.append(dict(box=p["box"], cls=p["cls"], ref=None, src="extra", pre_ref=p["ref"], score=p["score"], pre_text=p["text"]))
        nm = cnt["right"]+cnt["wrong"]+cnt["none"]
        if not res or np.median(res) > 0.3*med or nm < 0.5*len(refs):          # alignment failed: keep the pre-labels
            info.update(aligned=False, matched=nm, residual_px=round(float(np.median(res)), 1) if res else None)
            summary.append(info); print(json.dumps(info)); continue
        info.update(aligned=True, residual_px=round(float(np.median(res)), 1) if res else None, median_part_px=round(med, 1),
                    residual_part_sizes=round(float(np.median(res))/med, 3) if res else None, counts=dict(cnt),
                    box_recall=round(nm/max(len(refs), 1), 3), designator_right_of_matched=round(cnt["right"]/max(nm, 1), 3),
                    class_right_of_matched=round(1-cnt["class_wrong"]/max(nm, 1), 3))
        json.dump(dict(summary=info, **ex), open(R/f"design_check_{st}.json", "w"), indent=1)
        fixed[st] = dict(parts=out_parts, H=H.tolist()); summary.append(info); print(json.dumps(info))
if only:                                   # a subset run updates the earlier output instead of replacing it
    old = json.load(open(D/"design_fixed.json")) if (D/"design_fixed.json").exists() else {}
    old = {k: v for k, v in old.items() if k.split("_")[0] not in only}; old.update(fixed); fixed = old
    olds = json.load(open(R/"design_check_summary.json")) if (R/"design_check_summary.json").exists() else []
    summary = [s for s in olds if s["photo"].split("_")[0] not in only] + summary
json.dump(fixed, open(D/"design_fixed.json", "w")); json.dump(summary, open(R/"design_check_summary.json", "w"), indent=1)
