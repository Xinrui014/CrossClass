#!/usr/bin/env python3
"""Text-part linking baselines on WACV (problem setting: LINKING.md).
Task: for every part on a board, pick the printed designator text that names it, or 'none'.
Ground-truth link: the part's annotated designator string equals the text's annotated string.

Two tracks:
  oracle - annotated part boxes + annotated designator texts (isolates linking from finding / reading)
  e2e    - PCB-Dense detections (score >= 0.25) + PP-OCRv6 designator-like strings; a linked prefix replaces the detector class
           -> class-aware detection metrics (det_metrics.py) on the 42 test images
Rule baselines fix their thresholds on the 3 dev images; learned linkers use the 3 board folds (wacv_in/folds.json).
Usage: link_baselines.py [oracle|e2e|all]   (python: ~/venvs/gdino_fewshot, CPU only)"""
import sys, os, json, re, itertools, subprocess, collections
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
H = Path.home(); sys.path.insert(0, str(H/"projects/gdino_fewshot/assoc"))
from assoc_lib import load_gt, attach_detector, pair_features, assign, pfx_class, norm_text, iou, DES, FEAT, CAT, E
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass")); OUT = ROOT/"results/linking"; OUT.mkdir(parents=True, exist_ok=True)
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
folds = json.load(open(H/"projects/gdino_fewshot/wacv_in/folds.json")); preds = json.load(open(E/"preds_wacv_1280.json"))
board = lambda st: re.sub(r"_(Top|Bottom)\d*$", "", st)
IMGS = sorted(preds); TEST = [st for st in IMGS if st not in DEV]
fold_of = {st: next(int(k) for k, v in folds.items() if board(st) in v["test_boards"]) for st in IMGS}
TAUS = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 1e9]
GEO = [k for k, f in enumerate(FEAT) if f not in ("prefix_cls", "det_cls", "fit", "det_score")]

def geometry(parts, texts):
    """Pixel centre distance, pixel edge gap, both also divided by the part size (longer side); orientation agreement."""
    P = np.array([p["box"] for p in parts], float).reshape(-1, 4); T = np.array([t["box"] for t in texts], float).reshape(-1, 4)
    pc = (P[:, :2]+P[:, 2:])/2; tc = (T[:, :2]+T[:, 2:])/2; sz = np.maximum(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]), 1)
    C = np.linalg.norm(pc[:, None]-tc[None], axis=2)
    gx = np.maximum(0, np.maximum(P[:, None, 0]-T[None, :, 2], T[None, :, 0]-P[:, None, 2])); gy = np.maximum(0, np.maximum(P[:, None, 1]-T[None, :, 3], T[None, :, 1]-P[:, None, 3]))
    G = np.hypot(gx, gy); ph = (P[:, 2]-P[:, 0]) >= (P[:, 3]-P[:, 1]); th = (T[:, 2]-T[:, 0]) >= (T[:, 3]-T[:, 1])
    return dict(C=C, G=G, Cn=C/sz[:, None], Gn=G/sz[:, None], orient_diff=(ph[:, None] != th[None]).astype(float))

def hungarian(cost, none_cost):
    """One-to-one: each part takes one text or 'none' (cost none_cost); each text names at most one part."""
    n, m = cost.shape
    if n == 0: return {}
    C = np.full((n, m+n), 1e6); C[:, :m] = cost; C[np.arange(n), m+np.arange(n)] = none_cost
    r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}

# ---------- rule baselines: method(parts, texts, g, params) -> {part index: text index or None}
def nearest_centre(parts, texts, g, tau):
    return {i: (int(np.argmin(g["C"][i])) if g["Cn"][i].min() <= tau else None) for i in range(len(parts))}
def nearest_edge(parts, texts, g, tau):
    out = {}
    for i in range(len(parts)):
        j = int(np.lexsort((g["C"][i], g["G"][i]))[0]); out[i] = j if g["Gn"][i, j] <= tau else None
    return out
def text_picks_part(parts, texts, g, tau):
    """Each text goes to the part it is closest to (edge gap, ties by centre); a part keeps the nearest of the texts that chose it."""
    owner = {}
    for j in range(len(texts)):
        i = int(np.lexsort((g["C"][:, j], g["G"][:, j]))[0])
        if g["Gn"][i, j] <= tau and (i not in owner or g["C"][i, j] < g["C"][i, owner[i]]): owner[i] = j
    return {i: owner.get(i) for i in range(len(parts))}
def onetoone_centre(parts, texts, g, tau): return hungarian(g["Cn"], tau)
def onetoone_edge(parts, texts, g, tau): return hungarian(g["Gn"]+0.01*g["Cn"], tau)
def onetoone_sta(parts, texts, g, prm):
    """P&ID-style global matching (Romaric et al., DocEng 2026): geometric + orientation + semantic (prefix vs detector class) cost."""
    a, b, tau = prm
    pre = np.array([pfx_class(t["text"]) if pfx_class(t["text"]) is not None else -1 for t in texts])
    det = np.array([p.get("det_cls", 9) for p in parts]); clash = (pre[None] >= 0) & (det[:, None] < 9) & (pre[None] != det[:, None])
    return hungarian(g["Gn"]+0.01*g["Cn"]+a*g["orient_diff"]+b*clash, tau)
RULES = [("B1 nearest text (centre)", nearest_centre, TAUS), ("B2 nearest text (edge gap)", nearest_edge, TAUS),
         ("B3 text picks its nearest part", text_picks_part, TAUS), ("B4 one-to-one, centre distance", onetoone_centre, TAUS),
         ("B5 one-to-one, edge gap", onetoone_edge, TAUS),
         ("B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style)", onetoone_sta,
          list(itertools.product([0, 0.5, 1, 2], [0, 0.5, 1, 2, 4], TAUS)))]

# ---------- scoring of a link map against ground truth
def score(items, links):
    """items: {st: (parts, texts)}; links: {st: {i: j|None}}. Parts carry 'des' (true designator or '')."""
    c = collections.Counter()
    for st, (parts, texts) in items.items():
        tset = set(t["text"] for t in texts); L = links[st]
        for i, p in enumerate(parts):
            j = L.get(i); linkable = bool(p["des"]) and p["des"] in tset; got = texts[j]["text"] if j is not None else None
            c["linkable"] += linkable; c["links"] += got is not None
            if linkable:
                c["right"] += got == p["des"]
                c["prefix_right"] += got is not None and pfx_class(got) is not None and pfx_class(got) == pfx_class(p["des"])
                c["none_on_linkable"] += got is None
            elif got is not None: c["link_on_unlinkable"] += 1
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    return dict(acc=acc, prefix_acc=c["prefix_right"]/max(c["linkable"], 1), prec=prec, f1=2*acc*prec/max(acc+prec, 1e-9), **c)

def learned_links(data, cols, none_p=0.005):
    """3 board folds: HistGB pair scorer on assoc_lib features, then one-to-one assignment. Returns links + pair probabilities."""
    links, probs = {}, {}
    cat = [cols.index(k) for k in CAT if k in cols]
    for k in range(3):
        tr = [st for st in IMGS if fold_of[st] != k and data[st]["feats"]]
        X = np.array([f for st in tr for _, _, f in data[st]["feats"]])[:, cols]; Y = np.array([y for st in tr for y in data[st]["y"]])
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, categorical_features=cat or None, random_state=0).fit(X, Y)
        for st in IMGS:
            if fold_of[st] != k: continue
            d = data[st]; links[st] = {}; probs[st] = {}
            if not d["feats"]: continue
            p = m.predict_proba(np.array([f for _, _, f in d["feats"]])[:, cols])[:, 1]
            pp = {(i, j): pr for (i, j, _), pr in zip(d["feats"], p)}; probs[st] = pp
            links[st] = assign(len(d["pt"][0]), len(d["pt"][1]), pp, none_p=none_p)
    return links, probs

def net_fix(items, links):
    """Detections on a GT part: class right after the link override minus class right before (fixed - broken), and both counts."""
    fixed = broken = 0
    for st, (parts, texts) in items.items():
        for i, p in enumerate(parts):
            if p.get("gt_cls", -1) < 0: continue
            j = links[st].get(i); c = p["det_cls"]
            if j is not None and pfx_class(texts[j]["text"]) is not None: c = pfx_class(texts[j]["text"])
            fixed += c == p["gt_cls"] and p["det_cls"] != p["gt_cls"]; broken += c != p["gt_cls"] and p["det_cls"] == p["gt_cls"]
    return fixed-broken, fixed, broken

def run_rules(data, items_dev, items_test, log, objective="f1"):
    """Each rule's parameter is the one that maximises the objective on the 3 dev images: link F1 (oracle) or net class fixes (e2e)."""
    rows = {}
    obj = (lambda L: score(items_dev, L)["f1"]) if objective == "f1" else (lambda L: net_fix(items_dev, L)[0])
    for name, fn, grid in RULES:
        best = max(grid, key=lambda prm: (obj({st: fn(*items_dev[st], data[st]["g"], prm) for st in items_dev}), -np.sum(prm)))
        rows[name] = dict(param=best, links={st: (fn(*data[st]["pt"], data[st]["g"], best) if data[st]["g"] is not None else {}) for st in IMGS})
        log(f"  {name}: dev-chosen parameter {best}")
    return rows

def table(rows, items_test, title):
    lines = [f"### {title}", "", "| method | right label (linkable parts) | right prefix class | link precision | link F1 | links on parts with no printed label |",
             "|---|---|---|---|---|---|"]
    res = {}
    for name, r in rows.items():
        s = score(items_test, {st: r["links"][st] for st in items_test}); res[name] = s
        lines.append(f"| {name} | {100*s['acc']:.1f} % | {100*s['prefix_acc']:.1f} % | {100*s['prec']:.1f} % | {s['f1']:.3f} | {s.get('link_on_unlinkable', 0)} |")
    return "\n".join(lines), res

# ---------- track 1: oracle
def oracle(log):
    data = {}
    for st in IMGS:
        parts, texts = load_gt(st); parts = attach_detector(parts, preds[st])
        feats = pair_features(parts, texts); y = [int(parts[i]["des"] != "" and texts[j]["text"] == parts[i]["des"]) for i, j, _ in feats]
        data[st] = dict(pt=(parts, texts), g=geometry(parts, texts) if parts and texts else None, feats=feats, y=y)
    items = {st: data[st]["pt"] for st in IMGS if data[st]["g"] is not None}
    dev = {st: items[st] for st in DEV}; test = {st: items[st] for st in TEST if st in items}
    st_ = score(test, {st: {} for st in test})
    log(f"oracle, 42 test images: parts {sum(len(p) for p, _ in test.values())}, designator texts {sum(len(t) for _, t in test.values())}, "
        f"linkable parts {st_['linkable']}")
    rows = run_rules(data, dev, test, log)
    for name, cols in (("B7 learned pair scorer, geometry only + one-to-one", GEO), ("B8 learned pair scorer + detector class + one-to-one (current ③ linker)", list(range(len(FEAT))))):
        lk, _ = learned_links(data, cols); rows[name] = dict(param="3 board folds, none_p 0.005", links={st: lk.get(st, {}) for st in IMGS})
    return table(rows, test, "Oracle track — annotated parts + annotated designator texts (42 test images)")

# ---------- track 2: end to end
def fix(s):
    s = norm_text(s); m = re.match(r"^([A-Z]{1,4})([0-9OIL]{1,4})([A-Z]?)$", s)
    return (m.group(1)+m.group(2).replace("O", "0").replace("I", "1").replace("L", "1")+m.group(3)) if m else s
def nms_classwise(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]

def e2e(log):
    OCR = json.load(open(H/"projects/gdino_fewshot/assoc/ocr_wacv.json")); data = {}
    for st in IMGS:
        gparts, _ = load_gt(st)
        texts = [dict(box=o["box"], text=fix(o["text"])) for o in OCR[st]]
        texts = [t for t in texts if DES.match(t["text"]) and (t["box"][2]-t["box"][0]) > 2 and (t["box"][3]-t["box"][1]) > 2]
        parts = [dict(box=p[:4], det_cls=int(p[5]), det_score=float(p[4]), k=k) for k, p in enumerate(preds[st]) if p[4] >= 0.25]
        G = np.array([g["box"] for g in gparts]).reshape(-1, 4); Iu = iou(np.array([p["box"] for p in parts]).reshape(-1, 4), G)
        for i, p in enumerate(parts):
            j = int(np.argmax(Iu[i])) if G.shape[0] else -1; on = j >= 0 and Iu[i, j] >= 0.5
            p["des"] = gparts[j]["des"] if on else ""; p["gt_cls"] = gparts[j]["cls"] if on else -1
        feats = pair_features(parts, texts); y = [int(parts[i]["des"] != "" and texts[j]["text"] == parts[i]["des"]) for i, j, _ in feats]
        data[st] = dict(pt=(parts, texts), g=geometry(parts, texts) if parts and texts else None, feats=feats, y=y)
    items = {st: data[st]["pt"] for st in IMGS if data[st]["g"] is not None}
    dev = {st: items[st] for st in DEV}; test = {st: items[st] for st in TEST if st in items}
    rows = run_rules(data, dev, test, log, objective="netfix")
    for name, cols in (("B7 learned pair scorer, geometry only + one-to-one", GEO), ("B8 learned pair scorer + detector class + one-to-one (current ③ linker)", list(range(len(FEAT))))):
        lk, pr = learned_links(data, cols)
        # frozen ③ rule: a link counts only when its pair probability >= 0.5
        rows[name] = dict(param="3 board folds, none_p 0.005, p >= 0.5",
                          links={st: {i: (j if j is not None and pr[st].get((i, j), 0) >= 0.5 else None) for i, j in lk.get(st, {}).items()} for st in IMGS})
    # upper bound: a perfect linker on the texts OCR actually read (each detection gets the OCR text equal to its true designator)
    rows["U1 perfect linker on the OCR texts (upper bound)"] = dict(param="oracle", links={st: {i: next((j for j, t in enumerate(data[st]["pt"][1]) if p["des"] and t["text"] == p["des"]), None)
                                                                   for i, p in enumerate(data[st]["pt"][0])} for st in IMGS})
    md, res = table(rows, test, "End-to-end track — detector boxes + PP-OCRv6 texts; link scores over detections on a GT part (42 test images)")
    md += "\n\nClass changes on the 42 test images (detections on a GT part): fixed = wrong -> right, broken = right -> wrong\n\n| method | fixed | broken | net |\n|---|---|---|---|\n"
    for name, r in rows.items():
        n, f, b = net_fix(test, {st: r["links"][st] for st in test}); res[name]["fixed"], res[name]["broken"] = f, b
        md += f"| {name} | {f} | {b} | {n:+d} |\n"
    # class from the linked prefix -> detection metrics
    ev = OUT/"eval"; ev.mkdir(exist_ok=True); args = ["detector=wacv42"]
    for s in ("gt9_wacv42.json", "preds_wacv42_1280.json"):
        (ev/s).is_symlink() or os.symlink(ROOT/"data"/s, ev/s)
    for n, (name, r) in enumerate(rows.items(), 1):
        out = {}
        for st in IMGS:
            parts = data[st]["pt"][0]; texts = data[st]["pt"][1]; L = r["links"].get(st, {}); byk = {p["k"]: i for i, p in enumerate(parts)}; nb = []
            for k, p in enumerate(preds[st]):
                p = list(p); i = byk.get(k); j = L.get(i) if i is not None else None
                if j is not None and pfx_class(texts[j]["text"]) is not None: p[5] = pfx_class(texts[j]["text"])
                nb.append(p)
            out[st] = nms_classwise(nb)
        key = f"link{n}"; json.dump(out, open(ev/f"preds_{key}_1280.json", "w"))
        (ev/f"gt9_{key}.json").is_symlink() or os.symlink(ROOT/"data/gt9_wacv42.json", ev/f"gt9_{key}.json"); args.append(f"{name.split()[0]}={key}")
    met = subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py"), *args], cwd=ev, capture_output=True, text=True)
    return md + "\n\nClass from the linked prefix (unlinked detections keep the detector class), 42 test images:\n\n" + met.stdout + met.stderr, res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"; logs = []
    def log(s): print(s, flush=True); logs.append(s)
    parts = []
    if mode in ("oracle", "all"): md, res = oracle(log); print(md); parts.append(md); json.dump(res, open(OUT/"oracle.json", "w"), indent=1)
    if mode in ("e2e", "all"): md, res = e2e(log); print(md); parts.append(md); json.dump(res, open(OUT/"e2e.json", "w"), indent=1)
    open(OUT/f"baselines_{mode}.md", "w").write("\n".join(logs) + "\n\n" + "\n\n".join(parts) + "\n")
