#!/usr/bin/env python3
"""How often is a designer's (or annotator's) label printed closer to another part than to its own, and how many linker errors are on
such labels? For each linked label t of part i: gap = edge gap between boxes in median part sizes; t is 'misplaced' when its nearest part
(by gap) is another part q and gap(t, i) - gap(t, q) >= MARGIN; 'crossed' when two parts' labels are each misplaced onto the other part.
On a misplaced label a reader of the board (and a geometry-only linker) is pointed at the wrong part, so the error is in the layout,
not in the linker. Runs one checkpoint on KiCad val and WACV-clean v1 test; also the share of misplaced labels in KiCad train by file
version. Writes results/linking/label_placement_<run>.json and page/errors/placement.json + sheets/place_*.jpg (crops, as link_error_gallery).
Usage: label_placement_check.py --ckpt ckpt/dv1_E1_XXL_150k_lr1e-4.pt [--device cuda]   (python: ~/venvs/gdino_fewshot)"""
import argparse, json, os, sys, random, collections
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import linkformer as LF, link_error_gallery as G
ROOT = G.ROOT; D = G.D; MARGIN = 0.5

def gaps(b):
    P = np.array([p["box"] for p in b["parts"]], float).reshape(-1, 4); T = np.array([t["box"] for t in b["texts"]], float).reshape(-1, 4)
    unit = float(np.median(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]))) if len(P) else 1.0
    gx = np.maximum(0, np.maximum(T[:, None, 0]-P[None, :, 2], P[None, :, 0]-T[:, None, 2]))
    gy = np.maximum(0, np.maximum(T[:, None, 1]-P[None, :, 3], P[None, :, 1]-T[:, None, 3]))
    return np.hypot(gx, gy)/max(unit, 1e-6)                                  # (texts, parts)

def placement(b):
    """Per part: None (no label / left out) or (misplaced?, nearest other part, crossed?)."""
    strs = [t["text"] for t in b["texts"]]
    lab = [((p["link"] if p["link"] is not None else None) if "link" in p else next((j for j, s in enumerate(strs) if p["des"] and s == p["des"]), None))
           if not p.get("ign") else None for p in b["parts"]]
    if not b["parts"] or not b["texts"]: return [None]*len(b["parts"])
    Gm = gaps(b); out = [None]*len(b["parts"]); near = {}
    for i, t in enumerate(lab):
        if t is None: continue
        g = Gm[t].copy(); own = g[i]; g[i] = np.inf; q = int(np.argmin(g))
        mis = bool(len(g) > 1 and own - g[q] >= MARGIN); near[i] = q if mis else None; out[i] = [mis, q if mis else None, False]
    for i, q in near.items():
        if q is not None and near.get(q) == i: out[i][2] = True
    return out

@torch.no_grad()
def links(model, vocab, b, dev):
    if not (b["parts"] and b["texts"]): return {}
    return LF.decode(model(LF.to_dev(LF.encode(b, vocab), dev)).float())

def stats(boards, model, vocab, dev):
    c = collections.Counter(); recs = []
    for b in boards:
        if not (b["parts"] and b["texts"]): continue
        pl = placement(b); L = links(model, vocab, b, dev); tg = LF.encode(b, vocab)["tgt"]
        owner = {j: k for k, t in enumerate(tg) for j in t}; nearest = gaps(b).argmin(1)     # the part each text is printed closest to
        for i, x in enumerate(pl):
            if x is None or not tg[i]: continue
            mis, q, crossed = x; j = L.get(i); ok = j in tg[i]; k = "mis" if mis else "ok"
            c[k] += 1; c[k+"_wrong"] += not ok; c["crossed"] += crossed; c["crossed_wrong"] += crossed and not ok
            followed = j is not None and int(nearest[j]) == i                               # the pick is printed closest to this part
            if not ok: c[k+"_wrong_followed_print"] += followed
            if mis: recs.append(dict(board=b["id"], part=i, true=tg[i][0], pred=j, owner=owner.get(j) if j is not None else None, crossed=crossed,
                                     right=ok, followed=followed, true_str=b["texts"][tg[i][0]]["text"], pred_str=b["texts"][j]["text"] if j is not None else "",
                                     owner_des=b["parts"][owner[j]]["des"] if j is not None and j in owner else ""))
    return c, recs

def pct(a, b): return round(100*a/max(b, 1), 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu"); A = ck["args"]; vocab = LF.Vocab(ck["vocab"]); d, Lr = LF.SIZES[A["size"]]; dev = torch.device(a.device)
    model = LF.LinkFormer(d, Lr, len(vocab.idx)+2, A["tokens"] == "E2").to(dev); model.load_state_dict(ck["model"]); model.eval(); run = Path(a.ckpt).stem
    kic = json.load(open(D/"kicad.json")); val = set(json.load(open(D/"split.json"))["val_groups"])
    kv = [b for b in kic if b["group"] in val]; wc = [b for b in json.load(open(D/"wacv_clean_v1.json")) if b["id"] not in LF.DEV]
    res = {}
    for name, bs in (("kicad_val", kv), ("wacv_clean41", wc)):
        c, recs = stats(bs, model, vocab, dev); n = c["mis"]+c["ok"]; wrong = c["mis_wrong"]+c["ok_wrong"]
        res[name] = dict(linkable=n, misplaced=c["mis"], misplaced_pct=pct(c["mis"], n), crossed=c["crossed"],
                         acc_all=pct(n-wrong, n), acc_on_misplaced=pct(c["mis"]-c["mis_wrong"], c["mis"]), acc_on_rest=pct(c["ok"]-c["ok_wrong"], c["ok"]),
                         wrong=wrong, wrong_on_misplaced=c["mis_wrong"], wrong_on_misplaced_pct=pct(c["mis_wrong"], wrong),
                         wrong_on_misplaced_followed_print=c["mis_wrong_followed_print"], wrong_on_rest_followed_print=c["ok_wrong_followed_print"],
                         crossed_wrong=c["crossed_wrong"])
        res[name+"_recs"] = recs; print(name, res[name], flush=True)
    by_v = collections.defaultdict(collections.Counter)                      # KiCad train: share of misplaced labels by file format version
    for b in kic:
        if b["group"] in val: continue
        v = str(b.get("version") or "?")[:4]; v = "≤2017" if v.isdigit() and int(v) <= 2017 else v
        for x in placement(b):
            if x is not None: by_v[v]["n"] += 1; by_v[v]["mis"] += x[0]
    res["kicad_train_by_version"] = {v: dict(labels=c["n"], misplaced_pct=pct(c["mis"], c["n"])) for v, c in sorted(by_v.items())}
    print(json.dumps(res["kicad_train_by_version"], indent=0))
    ktot = sum(c["n"] for c in by_v.values()); kmis = sum(c["mis"] for c in by_v.values()); res["kicad_train"] = dict(labels=ktot, misplaced=kmis, misplaced_pct=pct(kmis, ktot))
    print("kicad train", res["kicad_train"])
    json.dump({k: v for k, v in res.items() if not k.endswith("_recs")}, open(ROOT/f"results/linking/label_placement_{run}.json", "w"), indent=1)
    # crops: misplaced labels the model got wrong (the model followed the print) and some it still got right
    rng = random.Random(0); show = {}
    for name, bs, tag in (("kicad_val", kv, "placek"), ("wacv_clean41", wc, "placew")):
        R = res[name+"_recs"]; wr = [r for r in R if not r["right"]]; rt = [r for r in R if r["right"]]
        pick = rng.sample(wr, min(48, len(wr))) + rng.sample(rt, min(16, len(rt)))
        for r in pick: r["type"] = "neighbour" if r["pred"] is not None and r["owner"] is not None else ("none" if r["pred"] is None else "free")
        G.render(bs, pick, name == "kicad_val", tag)
        for r in pick:
            if name == "kicad_val": r["board_name"] = next(b["board"] for b in bs if b["id"] == r["board"])
        show[name] = pick
    json.dump(dict(summary={k: v for k, v in res.items() if not k.endswith("_recs")}, kicad=show["kicad_val"], wacv=show["wacv_clean41"]), open(G.OUT/"placement.json", "w"))
