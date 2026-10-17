#!/usr/bin/env python3
"""Error attribution of the end-to-end class pipeline on WACV-clean v1 test photos (clean41 GT: part boxes, classes, links).
Pipeline = detector boxes (conf >= 0.25) + I_100k links on the rule-filtered OCR boxes (p >= 0.5) + reading (Claude where read, else
PARSeq) -> prefix class, else detector class; class-wise NMS 0.6. Each labelled / unlabelled GT part found by the detector is put in one
cause group; then, one group at a time, the wrong classes of that group are set to the GT class ("fixed") and the metrics are recomputed:
mAP50, macro-F1 (det_metrics), APr (mean AP of SW / L / Q / Y), balanced class accuracy on found parts. Usage: E2EDIR TODO READS RUN"""
import json, os, re, sys, subprocess, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); E = ROOT/"data"; LD = Path(sys.argv[1]); RUN = sys.argv[4]
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))}
sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc"))
from assoc_lib import pfx_class
os.chdir(E); import analyze9 as A; from analyze9 import iou
DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}; RARE = ["Switch", "Inductor", "Transistor", "Oscillator"]
W = {w["id"]: w for w in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json")) if w["id"] not in DEV}
G0 = json.load(open("gt9_wacv.json")); P = json.load(open("preds_wacv_1280.json"))
B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}; Lk = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json")); RD = json.load(open(LD/"e2e_read_all.json"))
gt = {}                                                    # clean41 GT in gt9 form; old non-component regions kept as ignore regions
for st, w in W.items():
    bx = [[*p["box"], (p["cls"] if 0 <= p.get("cls", -1) <= 8 and not p.get("ign") else -1), "clean"] for p in w["parts"]]
    gt[st] = dict(W=G0[st]["W"], H=G0[st]["H"], boxes=bx + [b for b in G0[st]["boxes"] if b[4] < 0])
def cover(a, b):
    iw = max(0, min(a[2], b[2])-max(a[0], b[0])); ih = max(0, min(a[3], b[3])-max(a[1], b[1])); return iw*ih/max((b[2]-b[0])*(b[3]-b[1]), 1e-9)
def nms(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]
# pipeline classes per detection, and a cause group per found GT part
cls = {st: [int(d[5]) for d in P[st]] for st in W}; rows = []
for st, w in W.items():
    b = B[st]; db = np.array([p["box"] for p in b["parts"]]).reshape(-1, 4); ob = [t["box"] for t in b["texts"]]
    for i, (j, p) in Lk.get(st, {}).items():
        if j is None or p < 0.5: continue
        c = pfx_class(CL.get((st, j), RD[st][j]))
        if c is not None: cls[st][b["det_index"][int(i)]] = c
    for pi, q in enumerate(w["parts"]):
        if q.get("ign") or not 0 <= q.get("cls", -1) <= 8: continue
        r = dict(st=st, gcls=q["cls"])
        I = iou(np.array([q["box"]]), db)[0] if len(db) else np.zeros(0)
        if not len(I) or I.max() < 0.5: r["group"] = "detector missed the part"; rows.append(r); continue
        i = int(np.argmax(I)); k = b["det_index"][i]; r["k"] = k; j, p = Lk.get(st, {}).get(str(i), (None, 0)); j = j if (j is not None and p >= 0.5) else None
        if q.get("link") is not None:
            tb = w["texts"][q["link"]]["box"]; ok = [jj for jj, o in enumerate(ob) if cover(o, tb) >= 0.5 or cover(tb, o) >= 0.5]
            other = lambda jj: any(cover(ob[jj], t["box"]) >= 0.5 or cover(t["box"], ob[jj]) >= 0.5 for t in w["texts"])
            r["group"] = ("label: OCR has no box on it" if not ok else "label: linked to it" if j in ok else "label: linked to none" if j is None
                          else "label: linked to a neighbour's label" if other(j) else "label: linked to a junk box")
        else: r["group"] = "no label: linked to none" if j is None else "no label: linked to a box"
        rows.append(r)
NM = A.NM
def metrics(tag, fixgroups=(), allfound=False):
    c2 = {st: list(v) for st, v in cls.items()}
    for r in rows:
        if "k" in r and (allfound or r["group"] in fixgroups): c2[r["st"]][r["k"]] = r["gcls"]
    out = {st: nms([[*P[st][k][:5], c2[st][k]] for k in range(len(P[st])) if P[st][k][4] >= 0.001]) for st in W}
    name = f"c41{tag}"; json.dump(out, open(f"preds_{name}_1280.json", "w")); json.dump(gt, open(f"gt9_{name}.json", "w"))
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), f"x={name}"], check=True, capture_output=True)
    dm = json.load(open("det_metrics_q38.json"))[0]; data = A.load(name)
    _, aps = A.evaluate(data, [(d["pb"], d["ps"], d["pc"], np.ones(len(d["gc"]), bool)) for d in data])
    pc = collections.defaultdict(lambda: [0, 0])          # found parts / class right, per GT class (conf >= 0.25, IoU >= 0.5)
    for st in W:
        g = [x for x in gt[st]["boxes"] if x[4] >= 0]; gb = np.array([x[:4] for x in g]).reshape(-1, 4); d = np.array([x for x in out[st] if x[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for m, x in enumerate(g):
            jj = int(np.argmax(I[m]))
            if I[m, jj] >= 0.5: pc[x[4]][0] += 1; pc[x[4]][1] += int(d[jj, 5]) == x[4]
    bal = np.mean([pc[c][1]/pc[c][0] for c in range(9) if pc[c][0]]); micro = sum(v[1] for v in pc.values())/sum(v[0] for v in pc.values())
    apr = np.mean([aps[n] for n in RARE if aps.get(n) is not None])
    return dict(tag=tag, map50=dm["map50"], macro_f1=dm["macro_f1"], apr=apr, bal=bal, micro=micro, aps=aps)
grp = collections.Counter(r["group"] for r in rows); wrong = collections.Counter()
for r in rows:
    if "k" in r and cls[r["st"]][r["k"]] != r["gcls"]: wrong[r["group"]] += 1
det_only = dict(cls); cls_pipe = cls
print(f"{'group':40s} parts  wrong class")
for g, n in grp.most_common(): print(f"{g:40s} {n:5d}  {wrong[g]:5d}")
res = [metrics("pipe")]
for g in [g for g, _ in grp.most_common() if g != "detector missed the part"]: res.append(metrics(re.sub(r"\W+", "_", g)[:30], fixgroups=(g,)))
res.append(metrics("allfound", allfound=True))
cls = {st: [int(d[5]) for d in P[st]] for st in W}; res.insert(0, metrics("det"))
groups = ["detector only", "pipeline"] + [g for g, _ in grp.most_common() if g != "detector missed the part"] + ["all found parts right"]
print(f"\n{'':44s} mAP50  macroF1   APr   bal.acc  acc")
for g, r in zip(groups, res):
    pre = "fix: " if g not in ("detector only", "pipeline", "all found parts right") else ""
    print(f"{pre+g:44s} {r['map50']:.3f}  {r['macro_f1']:.3f}  {r['apr']:.3f}  {100*r['bal']:5.1f}  {100*r['micro']:5.1f}")
print("\nAP50 per class, detector / pipeline:", {n: f"{res[0]['aps'][n]:.2f}/{res[1]['aps'][n]:.2f}" for n in NM})
# breakdown of "label: linked to it" and "label: linked to none" wrong cases
cls = cls_pipe
for gname in ("label: linked to it", "label: linked to none"):
    sub = collections.Counter(); ex = collections.defaultdict(list)
    for st, w in W.items():
        b = B[st]; db = np.array([p["box"] for p in b["parts"]]).reshape(-1, 4)
        for pi, q in enumerate(w["parts"]):
            if q.get("ign") or not 0 <= q.get("cls", -1) <= 8 or q.get("link") is None: continue
            I = iou(np.array([q["box"]]), db)[0] if len(db) else np.zeros(0)
            if not len(I) or I.max() < 0.5: continue
            i = int(np.argmax(I)); k = b["det_index"][i]; j, p = Lk.get(st, {}).get(str(i), (None, 0))
            r = next((r for r in rows if r.get("k") == k and r["st"] == st and r["gcls"] == q["cls"]), None)
            if r is None or r["group"] != gname or cls[st][k] == q["cls"]: continue
            true = w["texts"][q["link"]]["text"]; tc = pfx_class(true); dc = int(P[st][k][5])
            if gname == "label: linked to it":
                rd = CL.get((st, j), RD[st][j]); rc = pfx_class(rd)
                why = ("true label names another class (naming)" if tc != q["cls"] else "misread: wrong class" if rc is not None else "misread: no class -> detector wrong")
                ex[why].append(f"{true}->{rd or '-'} GT {NM[q['cls']][:4]} det {NM[dc][:4]}")
            else:
                why = "true label names another class (naming)" if tc != q["cls"] else "detector wrong, label would fix it"
                ex[why].append(f"{true} GT {NM[q['cls']][:4]} det {NM[dc][:4]} p {p:.2f}")
            sub[(why, NM[q["cls"]])] += 1
    print(f"\n== {gname}: wrong cases by reason and GT class")
    for (why, c), n in sorted(sub.items(), key=lambda x: (x[0][0], -x[1])): print(f"  {why:42s} {c:11s} {n}")
    for why, e in ex.items(): print(f"  e.g. {why}: {e[:8]}")
# wrong cases per group: GT class, wrong predicted class, detector confidence
print("\n== wrong cases per group: GT class / predicted class / mean confidence")
for g in ("label: linked to it", "label: linked to none", "no label: linked to none", "label: OCR has no box on it"):
    W_ = [r for r in rows if r["group"] == g and "k" in r and cls[r["st"]][r["k"]] != r["gcls"]]
    gtc = collections.Counter(NM[r["gcls"]][:5] for r in W_); prc = collections.Counter(NM[cls[r["st"]][r["k"]]][:5] for r in W_)
    conf = np.mean([P[r["st"]][r["k"]][4] for r in W_]); rare = sum(1 for r in W_ if NM[r["gcls"]] in RARE + ["Diode"])
    print(f"  {g:32s} n {len(W_):3d} conf {conf:.2f} GT rare+diode {rare:3d} | GT {dict(gtc.most_common())} | pred {dict(prc.most_common())}")
print("\n== combined sources")
LINKG = ("label: linked to none", "label: OCR has no box on it", "label: linked to a neighbour's label", "label: linked to a junk box")
for name, gs in (("link problems", LINKG), ("reading / naming", ("label: linked to it",)), ("no label", ("no label: linked to none", "no label: linked to a box"))):
    r = metrics(re.sub(r"\W+", "_", name), fixgroups=gs); n = sum(1 for x in rows if x["group"] in gs and "k" in x and cls[x["st"]][x["k"]] != x["gcls"])
    print(f"  {name:18s} wrong {n:4d}  mAP50 {r['map50']:.3f}  macroF1 {r['macro_f1']:.3f}  APr {r['apr']:.3f}  bal {100*r['bal']:.1f}  acc {100*r['micro']:.1f}")
