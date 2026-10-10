#!/usr/bin/env python3
"""Where the learned linkers go wrong (oracle track, 42 WACV test images).
Error categories for a linkable part (its designator is printed and annotated):
  outside   - the true label is not among the part's candidates (12 nearest texts within 8 part sizes)
  conflict  - the true label was given to another part (one-to-one assignment)
  none      - the part got 'none' although its label was free
  other     - the part took a different label while its own label stayed unused
plus links given to parts with no annotated label, texts that name no annotated part, and parts that share a designator.
Writes results/linking/errors.json and gallery crops page/img/linking/err_<cat>_<k>.jpg.
Usage: link_errors.py   (python: ~/venvs/gdino_fewshot, PYTHONPATH=~/projects/gdino_fewshot/assoc)"""
import sys, json, re, random, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import link_baselines as LB
from assoc_lib import load_gt, attach_detector, pair_features, pfx_class, iou, FEAT, RAW
OUT = LB.ROOT/"page/img/linking"; OUT.mkdir(parents=True, exist_ok=True)
CYAN, ORANGE, YELLOW, RED, MAG = (34, 211, 238), (251, 146, 60), (250, 204, 21), (239, 68, 68), (232, 121, 249)
pfx = lambda s: (re.match(r"[A-Z]+", s or "") or [""])[0]

data = {}
for st in LB.IMGS:
    parts, texts = load_gt(st); parts = attach_detector(parts, LB.preds[st])
    feats = pair_features(parts, texts); y = [int(parts[i]["des"] != "" and texts[j]["text"] == parts[i]["des"]) for i, j, _ in feats]
    data[st] = dict(pt=(parts, texts), g=LB.geometry(parts, texts) if parts and texts else None, feats=feats, y=y)
runs = {"B7": LB.learned_links(data, LB.GEO), "B8": LB.learned_links(data, list(range(len(FEAT))))}

res = {}; examples = collections.defaultdict(list)
for m, (links, probs) in runs.items():
    c = collections.Counter(); byboard = collections.Counter(); linkb = collections.Counter(); rank_wrong = []
    for st in LB.TEST:
        if data[st]["g"] is None: continue
        parts, texts = data[st]["pt"]; L = links.get(st, {}); P = probs.get(st, {})
        strs = collections.Counter(t["text"] for t in texts); dcount = collections.Counter(p["des"] for p in parts if p["des"])
        owner = {j: i for i, j in L.items() if j is not None}
        cand = collections.defaultdict(set)
        for i, j, _ in data[st]["feats"]: cand[i].add(j)
        C = data[st]["g"]["C"]
        for i, p in enumerate(parts):
            d = p["des"]; j = L.get(i)
            if d and d in strs:
                c["linkable"] += 1; linkb[LB.board(st)] += 1
                if j is not None and texts[j]["text"] == d: c["right"] += 1; continue
                tj = [k for k, t in enumerate(texts) if t["text"] == d]
                if not any(k in cand[i] for k in tj): cat = "outside"
                elif any(k in owner and owner[k] != i for k in tj): cat = "conflict"
                elif j is None: cat = "none"
                else: cat = "other"
                c[cat] += 1; byboard[LB.board(st)] += 1
                same = j is not None and pfx(texts[j]["text"]) == pfx(d); c[cat+("_gotnone" if j is None else "_sameprefix" if same else "_diffprefix")] += 1
                c["shared_designator_errors"] += dcount[d] > 1
                order = list(np.argsort(C[i])); rank_wrong.append(min(order.index(k) for k in tj) + 1)
                if m == "B8":
                    k0 = min(tj, key=lambda k: C[i, k]); stealer = next((owner[k] for k in tj if k in owner and owner[k] != i), None)
                    examples[cat].append(dict(st=st, i=i, true=k0, pred=j, stealer=stealer, p=float(P.get((i, j), 0)) if j is not None else None))
            elif j is not None:
                t = texts[j]["text"]; named = t in dcount
                c["link_on_unlabelled"] += 1; c["link_on_unlabelled_text_names_no_part"] += not named
                if m == "B8" and not named: examples["unlabelled"].append(dict(st=st, i=i, true=None, pred=j, stealer=None, p=float(P.get((i, j), 0))))
    rw = np.array(rank_wrong)
    res[m] = dict(counts=dict(c), acc=c["right"]/c["linkable"],
                  wrong_true_label_rank={"nearest": int((rw == 1).sum()), "2nd-3rd": int(((rw >= 2) & (rw <= 3)).sum()), "4th-12th": int(((rw >= 4) & (rw <= 12)).sum()), ">12th": int((rw > 12).sum())},
                  by_board=sorted([(b, byboard[b], linkb[b], round(byboard[b]/linkb[b], 3)) for b in linkb], key=lambda r: -r[1]))
    print(m, json.dumps(res[m]["counts"]), res[m]["wrong_true_label_rank"])

# texts that name no annotated part; parts that share a designator (all 45 images)
nopart = collections.Counter(); shared = collections.Counter(); shared_ex = []
for st in LB.IMGS:
    parts, texts = data[st]["pt"]; dset = collections.Counter(p["des"] for p in parts if p["des"])
    for t in texts:
        if t["text"] not in dset: nopart[pfx(t["text"])] += 1
    for d, n in dset.items():
        if n < 2: continue
        idx = [i for i, p in enumerate(parts) if p["des"] == d]; B = np.array([parts[i]["box"] for i in idx]); I = iou(B, B); np.fill_diagonal(I, 0)
        kind = "overlapping boxes (duplicate annotation)" if I.max() > 0.3 else "separate parts"
        shared[kind] += n
        if kind == "separate parts" and len(shared_ex) < 60 and d in {t["text"] for t in texts}: shared_ex.append(dict(st=st, idx=idx, d=d))
res["texts_without_part_by_prefix"] = nopart.most_common(15); res["texts_without_part"] = sum(nopart.values())
res["shared_designator_parts"] = dict(shared)
print("texts naming no part:", sum(nopart.values()), nopart.most_common(12)); print("shared:", dict(shared))

# worked feature example: capacitor C105 on ML450_Top2 and its candidates
st = "ML450_Top2"; parts, texts = data[st]["pt"]; i0 = next(i for i, p in enumerate(parts) if p["des"] == "C105")
P8 = runs["B8"][1][st]; P7 = runs["B7"][1][st]
rows = [dict(text=texts[j]["text"], true=texts[j]["text"] == "C105", p_B7=round(float(P7[(i, j)]), 3), p_B8=round(float(P8[(i, j)]), 3),
             **{f: (round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v)) for f, v in zip(FEAT, fv)}) for i, j, fv in data[st]["feats"] if i == i0]
rows.sort(key=lambda r: r["dist"]); res["feature_example"] = dict(image=st, part="C105", part_box=parts[i0]["box"], candidates=rows)

# gallery crops
def crop_case(st, focus, true, pred, stealer, path, extra_parts=()):
    parts, texts = data[st]["pt"]; img = Image.open(sorted((RAW/st).glob("*.jpg"))[0]).convert("RGB")
    boxes = [parts[focus]["box"]] + [texts[k]["box"] for k in (true, pred) if k is not None] + ([parts[stealer]["box"]] if stealer is not None else []) + [parts[k]["box"] for k in extra_parts]
    B = np.array(boxes); sz = max(parts[focus]["box"][2]-parts[focus]["box"][0], parts[focus]["box"][3]-parts[focus]["box"][1])
    pad = max(1.2*sz, 40); x0, y0 = B[:, 0].min()-pad, B[:, 1].min()-pad; x1, y1 = B[:, 2].max()+pad, B[:, 3].max()+pad
    side = max(x1-x0, y1-y0); cx, cy = (x0+x1)/2, (y0+y1)/2; x0, y0, x1, y1 = cx-side/2, cy-side/2, cx+side/2, cy+side/2
    W = (x0, y0, x1, y1); s = 360/side; im = img.crop(tuple(int(v) for v in W)).resize((360, 360), Image.LANCZOS); d = ImageDraw.Draw(im)
    T = lambda b: [(b[0]-x0)*s, (b[1]-y0)*s, (b[2]-x0)*s, (b[3]-y0)*s]; ctr = lambda b: ((T(b)[0]+T(b)[2])/2, (T(b)[1]+T(b)[3])/2)
    vis = lambda b: b[2] > x0 and b[0] < x1 and b[3] > y0 and b[1] < y1
    for t in texts:
        if vis(t["box"]): d.rectangle(T(t["box"]), outline=ORANGE, width=1)
    for p in parts:
        if vis(p["box"]): d.rectangle(T(p["box"]), outline=CYAN, width=1)
    def ln(a, b, col, dash=False):
        if not dash: d.line([a, b], fill=(0, 0, 0), width=6); d.line([a, b], fill=col, width=3); return
        n = max(int(np.hypot(b[0]-a[0], b[1]-a[1])/7), 1)
        for k in range(0, n, 2): t0, t1 = k/n, min((k+1)/n, 1); d.line([(a[0]+(b[0]-a[0])*t0, a[1]+(b[1]-a[1])*t0), (a[0]+(b[0]-a[0])*t1, a[1]+(b[1]-a[1])*t1)], fill=col, width=3)
    if stealer is not None: d.rectangle(T(parts[stealer]["box"]), outline=MAG, width=3); ln(ctr(parts[stealer]["box"]), ctr(texts[true]["box"]), MAG)
    if true is not None: d.rectangle(T(texts[true]["box"]), outline=YELLOW, width=3); ln(ctr(parts[focus]["box"]), ctr(texts[true]["box"]), YELLOW, dash=True)
    if pred is not None: d.rectangle(T(texts[pred]["box"]), outline=RED, width=3); ln(ctr(parts[focus]["box"]), ctr(texts[pred]["box"]), RED)
    for k in extra_parts: d.rectangle(T(parts[k]["box"]), outline=MAG, width=3)
    d.rectangle(T(parts[focus]["box"]), outline=CYAN, width=4); im.save(path, quality=88)

random.seed(0); gal = {}
for cat in ("conflict", "other", "none", "outside", "unlabelled"):
    ex = examples[cat]; byb = collections.defaultdict(list)
    for e in ex: byb[LB.board(e["st"])].append(e)
    pick = []; boards = list(byb); random.shuffle(boards)
    while len(pick) < 6 and any(byb.values()):
        for b in boards:
            if byb[b] and len(pick) < 6: pick.append(byb[b].pop(random.randrange(len(byb[b]))))
    gal[cat] = []
    for k, e in enumerate(pick):
        parts, texts = data[e["st"]]["pt"]; fn = f"err_{cat}_{k}.jpg"
        crop_case(e["st"], e["i"], e["true"], e["pred"], e["stealer"], OUT/fn)
        gal[cat].append(dict(file=fn, board=e["st"], part=parts[e["i"]]["des"] or "(no label annotated)", cls=parts[e["i"]]["cls"],
                             pred=texts[e["pred"]]["text"] if e["pred"] is not None else "none", p=e["p"],
                             stealer=parts[e["stealer"]]["des"] if e["stealer"] is not None else None))
# one label, two separate parts
random.seed(1); gal["shared"] = []
for k, e in enumerate(random.sample(shared_ex, min(4, len(shared_ex)))):
    parts, texts = data[e["st"]]["pt"]; tj = next(j for j, t in enumerate(texts) if t["text"] == e["d"]); fn = f"err_shared_{k}.jpg"
    crop_case(e["st"], e["idx"][0], tj, None, None, OUT/fn, extra_parts=e["idx"][1:])
    gal["shared"].append(dict(file=fn, board=e["st"], part=e["d"], n=len(e["idx"]), cls=[parts[i]["cls"] for i in e["idx"]]))
res["gallery"] = gal
json.dump(res, open(LB.OUT/"errors.json", "w"), indent=1)
print(json.dumps(res["B8"]["by_board"][:10])); print(json.dumps(gal, indent=0)[:3000])
