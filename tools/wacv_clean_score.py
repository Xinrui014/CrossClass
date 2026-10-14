#!/usr/bin/env python3
"""Rescore linkers (oracle track) on the old WACV labels and on WACV-clean v1 (tools/wacv_clean.py).
Test sets: old42 = wacv.json, 42 test images, truth = same string (the published numbers); old41 = old42 without ML365_Bottom;
clean41 = wacv_clean_v1.json, truth = explicit link, parts marked 'ign' left out. Score = right label / linkable parts (exact pair),
precision = right / links made, F1; ceiling = best possible one-to-one score with the given truth.
Linkers: B7 / B8 (HistGB pair scorer, 3 board folds, trained on the labels of the set being scored), transformer checkpoints
(KiCad only, zero-shot). Writes results/linking/wacv_clean_v1_scores.{json,md}.
Usage: wacv_clean_score.py [--ckpt ckpt/a.pt,ckpt/b.pt] [--device cuda]   (python: ~/venvs/gdino_fewshot)"""
import argparse, json, os, re, sys, collections
from pathlib import Path
import numpy as np, torch
H = Path.home(); sys.path.insert(0, str(H/"projects/gdino_fewshot/assoc")); sys.path.insert(0, str(Path(__file__).parent))
from assoc_lib import attach_detector, pair_features, assign, CAT, FEAT, E
from sklearn.ensemble import HistGradientBoostingClassifier
import linkformer as LF
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass")); D = ROOT/"linkdata/CrossClass_data_v1"; OUT = ROOT/"results/linking"
DEV = LF.DEV; folds = json.load(open(D/"folds.json")); preds = json.load(open(E/"preds_wacv_1280.json"))
fold_of = lambda b: next(int(k) for k, v in folds.items() if b["board"] in v["test_boards"])
GEO = [k for k, f in enumerate(FEAT) if f not in ("prefix_cls", "det_cls", "fit", "det_score")]

def gold(b):
    """Per part: target text indices ([] = none) and whether the part is left out."""
    strs = [t["text"] for t in b["texts"]]
    tg = [(([p["link"]] if p["link"] is not None else []) if "link" in p else [j for j, s in enumerate(strs) if p["des"] and s == p["des"]]) for p in b["parts"]]
    return tg, [bool(p.get("ign")) for p in b["parts"]]

def count(b, L):
    tg, ig = gold(b); c = collections.Counter()
    for i, t in enumerate(tg):
        if ig[i]: continue
        j = L.get(i); c["links"] += j is not None
        if t: c["linkable"] += 1; c["right"] += j in t
    return c

def ceiling(b):
    """Best one-to-one score: texts that are the target of several parts can serve only one of them."""
    tg, ig = gold(b); T = collections.defaultdict(list)
    for i, t in enumerate(tg):
        if not ig[i]: [T[j].append(i) for j in t]
    from scipy.optimize import linear_sum_assignment
    I = sorted({i for v in T.values() for i in v}); J = sorted(T)
    if not I: return 0
    C = np.zeros((len(I), len(J)))
    for c, j in enumerate(J):
        for i in T[j]: C[I.index(i), c] = -1
    r, c = linear_sum_assignment(C); return int(-C[r, c].sum())

def summ(boards, links):
    c = collections.Counter()
    for b in boards: c.update(count(b, links.get(b["id"], {})))
    acc, prec = c["right"]/max(c["linkable"], 1), c["right"]/max(c["links"], 1)
    return dict(acc=round(100*acc, 2), prec=round(100*prec, 2), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), wrong=c["linkable"]-c["right"], **c)

def learned(boards, cols, none_p=0.005):
    """B7 / B8 exactly as in link_baselines.py, with the pair label taken from the set's own truth (ign parts not trained on)."""
    data = {}
    for b in boards:
        parts = attach_detector([dict(p) for p in b["parts"]], preds[b["id"]]); texts = b["texts"]; tg, ig = gold(b)
        f = pair_features(parts, texts) if parts and texts else []
        data[b["id"]] = dict(f=[x for x in f if not ig[x[0]]], y=[int(x[1] in tg[x[0]]) for x in f if not ig[x[0]]], all=f, n=len(parts), m=len(texts))
    cat = [cols.index(k) for k in CAT if k in cols]; links = {}
    for k in range(3):
        tr = [b["id"] for b in boards if fold_of(b) != k and data[b["id"]]["f"]]
        X = np.array([x for s in tr for _, _, x in data[s]["f"]])[:, cols]; Y = np.array([y for s in tr for y in data[s]["y"]])
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, categorical_features=cat or None, random_state=0).fit(X, Y)
        for b in boards:
            d = data[b["id"]]
            if fold_of(b) != k or not d["all"]: continue
            p = m.predict_proba(np.array([x for _, _, x in d["all"]])[:, cols])[:, 1]
            links[b["id"]] = assign(d["n"], d["m"], {(i, j): pr for (i, j, _), pr in zip(d["all"], p)}, none_p=none_p)
    return links

@torch.no_grad()
def transformer(path, boards, dev):
    model, vocab, a = LF.load_model(path, dev)
    links = {}
    for b in boards:
        if not (b["parts"] and b["texts"]): continue
        e = LF.to_dev(LF.encode(b, vocab), dev); links[b["id"]] = LF.decode(model(e))
    return links

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=""); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    old = json.load(open(D/"wacv.json")); clean = json.load(open(D/"wacv_clean_v1.json"))
    sets = {"old42": [b for b in old if b["id"] not in DEV], "old41": [b for b in old if b["id"] not in DEV and b["id"] != "ML365_Bottom"],
            "clean41": [b for b in clean if b["id"] not in DEV]}
    full = {"old42": old, "old41": old, "clean41": clean}                     # B7 / B8 folds train on all 45 (44) images' labels
    res = {s: dict(ceiling=sum(ceiling(b) for b in bs), linkable=summ(bs, {})["linkable"], ignored=sum(sum(gold(b)[1]) for b in bs),
                   parts=sum(len(b["parts"]) for b in bs)) for s, bs in sets.items()}
    meth = {}
    for name, cols in (("B7", GEO), ("B8", list(range(len(FEAT))))):
        for s in sets:
            if s == "old41": continue
            lk = learned(full[s], cols); meth.setdefault(name, {})[s] = lk; print(name, s, summ(sets[s], lk), flush=True)
        meth[name]["old41"] = meth[name]["old42"]                             # same model, ML365_Bottom just not counted
    dev = torch.device(a.device)
    for p in [x for x in a.ckpt.split(",") if x]:
        name = Path(p).stem; meth[name] = {}
        for s in ("old42", "clean41"): meth[name][s] = transformer(p, sets[s], dev); print(name, s, summ(sets[s], meth[name][s]), flush=True)
        meth[name]["old41"] = meth[name]["old42"]
    for name, by in meth.items():
        for s in sets: res[s][name] = summ(sets[s], by[s])
    json.dump(res, open(OUT/"wacv_clean_v1_scores.json", "w"), indent=1)
    lines = ["| set | parts | linkable | left out | one-to-one ceiling | " + " | ".join(meth) + " |", "|---|---:|---:|---:|---:|" + "---:|"*len(meth)]
    for s in sets:
        r = res[s]; lines.append(f"| {s} | {r['parts']} | {r['linkable']} | {r['ignored']} | {100*r['ceiling']/max(r['linkable'], 1):.2f} % | "
                                 + " | ".join(f"{r[m]['acc']:.1f} % ({r[m]['wrong']} wrong)" for m in meth) + " |")
    open(OUT/"wacv_clean_v1_scores.md", "w").write("Right label (exact pair) on linkable parts; oracle track.\n\n" + "\n".join(lines) + "\n"); print("\n".join(lines))
