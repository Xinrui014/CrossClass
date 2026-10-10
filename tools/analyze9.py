#!/usr/bin/env python3
"""Cross-domain error analysis of the fixed PCB-Dense v22 detector (9 classes).
Operating point conf 0.25, IoU 0.5 (same as all our tables). For every GT component one outcome:
  correct | wrong_class (other-class box IoU>=.5) | bad_box (same-class box IoU .1-.5) | weak (same-class box IoU>=.5, score<.25)
  | missed (nothing of the above; sub-split: only a weak other-class box / nothing at IoU>=.5 at any score = no_box)
For every prediction >= .25 that is not correct one type:
  duplicate | wrong_class | bad_box | on_ignore (IoU>=.3 with an ignore-type annotation) | background (IoU<.1 with every annotation) | near_other
TIDE-style dAP50: mAP50 gain if one error type alone were fixed (Bolya et al. ECCV 2020, simplified)."""
import json, sys, collections
import numpy as np
NM = ["Resistor","Capacitor","Inductor","Connector","Diode","Switch","Transistor","IC","Oscillator"]
THR, FG, BG, IGN = 0.25, 0.5, 0.1, 0.3
SIZE_BINS = [0, 8, 16, 32, 64, 128, 1e9]; SIZE_LAB = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]

def iou(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None, :]-inter+1e-9)

def ap101(tp, conf, npos):
    """101-point interpolated AP, identical to tools/score_sets.py ap_at"""
    if npos == 0: return None
    if len(tp) == 0: return 0.0
    o = np.argsort(-conf, kind="stable"); hit = tp[o] > 0
    ctp = np.cumsum(hit); cfp = np.cumsum(~hit); rec = ctp/npos; prec = ctp/(ctp+cfp)
    env = np.maximum.accumulate(prec[::-1])[::-1]; ts = np.arange(101)/100; idx = np.searchsorted(rec, ts, side="left")
    return float(np.where(idx < len(rec), env[np.minimum(idx, len(rec)-1)], 0).sum()/101)

def load(setname):
    gt = json.load(open(f"gt9_{setname}.json")); pr = json.load(open(f"preds_{setname}_1280.json"))
    data = []
    for st, g in gt.items():
        b = g["boxes"]; s = 1280/max(g["W"], g["H"])
        gm = np.array([x[:4] for x in b if x[4] >= 0]).reshape(-1, 4); gc = np.array([x[4] for x in b if x[4] >= 0], int)
        graw = [x[5] for x in b if x[4] >= 0]
        gi = np.array([x[:4] for x in b if x[4] < 0]).reshape(-1, 4); giraw = [x[5] for x in b if x[4] < 0]
        p = np.array(pr.get(st, [])).reshape(-1, 6); p = p[np.argsort(-p[:, 4])]
        data.append(dict(st=st, scale=s, gm=gm, gc=gc, graw=graw, gi=gi, giraw=giraw, pb=p[:, :4], ps=p[:, 4], pc=p[:, 5].astype(int)))
    return data

def match(d, keep):
    """greedy per class in score order over predictions where keep is True -> tp flag per pred, matched-gt index"""
    n = len(d["ps"]); tpi = -np.ones(n, int); gmatched = -np.ones(len(d["gc"]), int)
    I = iou(d["pb"], d["gm"])
    for k in np.where(keep)[0]:
        c = d["pc"][k]; cand = np.where((d["gc"] == c) & (gmatched < 0) & (I[k] >= FG))[0]
        if len(cand): j = cand[np.argmax(I[k, cand])]; tpi[k] = j; gmatched[j] = k
    return tpi, gmatched, I

def pred_types(d, keep, tpi, gmatched, I):
    Ii = iou(d["pb"], d["gi"]); t = np.array([""]*len(d["ps"]), dtype=object); tgt = -np.ones(len(d["ps"]), int)
    for k in np.where(keep)[0]:
        if tpi[k] >= 0: t[k] = "correct"; continue
        c = d["pc"][k]; row = I[k]; same = d["gc"] == c
        if np.any(same & (row >= FG)): t[k] = "duplicate"; continue
        oth = (~same) & (row >= FG)
        if np.any(oth): t[k] = "wrong_class"; tgt[k] = np.where(oth)[0][np.argmax(row[oth])]; continue
        loc = same & (row >= BG)
        if np.any(loc): t[k] = "bad_box"; tgt[k] = np.where(loc)[0][np.argmax(row[loc])]; continue
        mi = Ii[k].max() if Ii.shape[1] else 0.0
        if mi >= IGN: t[k] = "on_ignore"; continue
        mm = row.max() if len(row) else 0.0
        t[k] = "background" if max(mm, mi) < BG else "near_other"
    return t, tgt, Ii

def analyse(setname):
    data = load(setname); R = {"set": setname}
    gt_out = collections.Counter(); gt_cls = collections.defaultdict(collections.Counter); gt_size = collections.defaultdict(collections.Counter)
    pr_out = collections.Counter(); pr_cls = collections.defaultdict(collections.Counter); conf = np.zeros((9, 9), int)
    ign_raw = collections.Counter(); board = []; miss_raw = collections.defaultdict(collections.Counter)
    for d in data:
        keep = d["ps"] >= THR; tpi, gm, I = match(d, keep); t, tgt, Ii = pred_types(d, keep, tpi, gm, I)
        for k in np.where(keep)[0]:
            pr_out[t[k]] += 1; pr_cls[t[k]][NM[d["pc"][k]]] += 1
            if t[k] == "on_ignore": ign_raw[d["giraw"][int(np.argmax(Ii[k]))]] += 1
        bc = collections.Counter()
        for j in range(len(d["gc"])):
            c = d["gc"][j]; col = I[:, j]
            sz = max(d["gm"][j, 2]-d["gm"][j, 0], d["gm"][j, 3]-d["gm"][j, 1])*d["scale"]
            if gm[j] >= 0: o = "correct"
            else:
                hi = keep & (col >= FG) & (d["pc"] != c)
                if np.any(hi): o = "wrong_class"; conf[c, d["pc"][np.where(hi)[0][0]]] += 1
                elif np.any(keep & (d["pc"] == c) & (col >= BG)): o = "bad_box"
                elif np.any((~keep) & (d["pc"] == c) & (col >= FG)): o = "weak"
                elif np.any((~keep) & (col >= FG)): o = "missed_weak_other"
                else: o = "missed_no_box"
            gt_out[o] += 1; gt_cls[NM[c]][o] += 1; bc[o] += 1
            gt_size[SIZE_LAB[np.searchsorted(SIZE_BINS, sz, side="right")-1]][o] += 1
            if o.startswith("missed"): miss_raw[NM[c]][d["graw"][j]] += 1
        nfp = sum(1 for k in np.where(keep)[0] if t[k] not in ("correct", "on_ignore"))
        board.append(dict(st=d["st"], n=len(d["gc"]), correct=bc["correct"], wrong=bc["wrong_class"], badbox=bc["bad_box"],
                          weak=bc["weak"], missed=bc["missed_weak_other"]+bc["missed_no_box"], fp=nfp))
    R.update(gt_out=dict(gt_out), gt_cls={k: dict(v) for k, v in gt_cls.items()}, gt_size={k: dict(v) for k, v in gt_size.items()},
             pr_out=dict(pr_out), pr_cls={k: dict(v) for k, v in pr_cls.items()}, confusion=conf.tolist(), on_ignore_raw=dict(ign_raw.most_common(15)),
             missed_raw={k: dict(v.most_common(6)) for k, v in miss_raw.items()}, boards=board)
    R["dAP"] = tide(data); return R

def evaluate(data, fixed):
    """mAP50 over classes with GT; fixed[i] = (pb, ps, pc, gmask) per image; ignore-region detections are skipped."""
    per = collections.defaultdict(lambda: [[], [], 0])
    for d, (pb, ps, pc, gmask) in zip(data, fixed):
        gm, gc = d["gm"][gmask], d["gc"][gmask]; o = np.argsort(-ps); pb, ps, pc = pb[o], ps[o], pc[o]
        I = iou(pb, gm); Ii = iou(pb, d["gi"]); used = np.zeros(len(gc), bool)
        for c in range(9): per[c][2] += int((gc == c).sum())
        for k in range(len(ps)):
            c = pc[k]; cand = np.where((gc == c) & (~used) & (I[k] >= FG))[0] if len(gc) else []
            if len(cand): used[cand[np.argmax(I[k, cand])]] = True; per[c][0].append(1); per[c][1].append(ps[k])
            else:
                if Ii.shape[1] and Ii[k].max() >= IGN and not np.any((gc == c) & (I[k] >= FG)): continue
                per[c][0].append(0); per[c][1].append(ps[k])
    aps = {NM[c]: ap101(np.array(v[0], float), np.array(v[1], float), v[2]) for c, v in per.items()}
    vals = [a for a in aps.values() if a is not None]; return float(np.mean(vals)), aps

def tide(data):
    """types over ALL predictions (score >= .001), each fix applied alone"""
    info = []
    for d in data:
        keep = np.ones(len(d["ps"]), bool); tpi, gm, I = match(d, keep); t, tgt, _ = pred_types(d, keep, tpi, gm, I)
        info.append((t, tgt, gm))
    def base(d): return d["pb"].copy(), d["ps"].copy(), d["pc"].copy(), np.ones(len(d["gc"]), bool)
    res = {}; res["base"] = evaluate(data, [base(d) for d in data])[0]
    for fix in ("wrong_class", "bad_box", "duplicate", "background", "near_other", "missed"):
        fx = []
        for d, (t, tgt, gm) in zip(data, info):
            pb, ps, pc, gmask = base(d)
            if fix in ("wrong_class", "bad_box"):
                drop = np.zeros(len(ps), bool); seen = set()
                for k in np.where(t == fix)[0]:
                    j = tgt[k]
                    if gm[j] >= 0 or j in seen: drop[k] = True; continue
                    seen.add(j); pc[k] = d["gc"][j]; pb[k] = d["gm"][j] if fix == "bad_box" else pb[k]
                    if fix == "wrong_class": pb[k] = pb[k]
                pb, ps, pc = pb[~drop], ps[~drop], pc[~drop]
            elif fix in ("duplicate", "background", "near_other"):
                k = t != fix; pb, ps, pc = pb[k], ps[k], pc[k]
            elif fix == "missed":
                covered = set(int(j) for j in tgt[(t == "wrong_class") | (t == "bad_box")])
                gmask = np.array([(gm[j] >= 0) or (j in covered) for j in range(len(d["gc"]))], bool)
            fx.append((pb, ps, pc, gmask))
        res[fix] = evaluate(data, fx)[0] - res["base"]
    res["base_per_class"] = evaluate(data, [base(d) for d in data])[1]
    return res

if __name__ == "__main__":
    out = {s: analyse(s) for s in sys.argv[1:]}
    json.dump(out, open("err9_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
    for s, R in out.items():
        n = sum(R["gt_out"].values()); print("\n==", s, "GT", n)
        for k, v in sorted(R["gt_out"].items(), key=lambda x: -x[1]): print(f"  GT {k:18s} {v:6d} {v/n:6.1%}")
        m = sum(R["pr_out"].values())
        for k, v in sorted(R["pr_out"].items(), key=lambda x: -x[1]): print(f"  PR {k:18s} {v:6d} {v/m:6.1%}")
        print("  dAP", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["dAP"].items() if k != "base_per_class"})
