#!/usr/bin/env python3
"""Why does the linker leave a labelled part without a label ('none' errors)? One checkpoint on KiCad val and WACV-clean v1 test.
For every none-error the model's own first choice (argmax over texts + none, before the one-to-one assignment) says which case it is:
  lost_true   first choice was the true label, but the assignment gave that text to another part
  lost_other  first choice was some other text, taken by another part, so the part fell back to none
  wants_none  first choice was none; split by how close the true label came (log p_none - log p_true < 1 = near miss)
For wants_none the true label's edge gap (median part sizes), the part's size, class and label placement are compared with correctly
linked parts. Then a sweep of a bias added to the none score before the assignment (negative = link more): right label / precision;
and the split vote: none errors where the texts together beat none (p_none < 0.5) but no single text does, with a decoder that links
every part whose p_none < 0.5.
Writes results/linking/none_<run>.json. Usage: none_analysis.py --ckpt ckpt/dv1_E1_XXL_150k_lr1e-4.pt [--device cuda]   (python: ~/venvs/gdino_fewshot)"""
import argparse, json, sys, collections
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import linkformer as LF, label_placement_check as LP
ROOT = LP.ROOT; D = LP.D
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
BIAS = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1]

@torch.no_grad()
def run(model, vocab, boards, dev):
    out = []
    for b in boards:
        if not (b["parts"] and b["texts"]): continue
        e = LF.encode(b, vocab); lg = model(LF.to_dev(e, dev)).float().cpu()
        out.append((b, e, lg))
    return out

def decode_bias(lg, bias):
    x = lg.clone(); x[:, -1] += bias; return LF.decode(x)

def analyse(R):
    c = collections.Counter(); feat = collections.defaultdict(list); rows = []
    for b, e, lg in R:
        L = LF.decode(lg); lp = F.log_softmax(lg, 1).numpy(); m = e["m"]; tg, ig = e["tgt"], e["ign"]
        G = LP.gaps(b); pl = LP.placement(b)
        P = np.array([p["box"] for p in b["parts"]], float); size = np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]); unit = float(np.median(size))
        taken = {j: i for i, j in L.items() if j is not None}
        for i, t in enumerate(tg):
            if ig[i] or not t: continue
            j = L.get(i); top = int(lp[i].argmax()); g = float(G[t[0], i]); mis = bool(pl[i] and pl[i][0])
            rec = dict(gap=g, size=float(size[i]/unit), cls=CLS[b["parts"][i]["cls"]], mis=mis, top_none=top == m)
            if j in t: c["right"] += 1; feat["right"].append(rec); continue
            if j is not None: continue                                        # wrong text: not this analysis
            c["none"] += 1
            if top == t[0]: k = "lost_true"
            elif top != m: k = "lost_other"
            else: k = "wants_none_near" if lp[i, m] - lp[i, t[0]] < 1 else "wants_none_far"
            c[k] += 1; feat[k].append(rec)
            rows.append(dict(board=b["id"], part=i, case=k, gap=round(g, 2), size=round(float(size[i]/unit), 2), cls=rec["cls"], misplaced=mis,
                             p_none=round(float(np.exp(lp[i, m])), 3), p_true=round(float(np.exp(lp[i, t[0]])), 3),
                             true_taken_by=taken.get(t[0])))
    def desc(rs):
        if not rs: return {}
        g = np.array([r["gap"] for r in rs]); s = np.array([r["size"] for r in rs]); cl = collections.Counter(r["cls"] for r in rs)
        return dict(n=len(rs), gap_median=round(float(np.median(g)), 2), gap_over_1=round(100*float((g > 1).mean()), 1), gap_over_2=round(100*float((g > 2).mean()), 1),
                    size_median=round(float(np.median(s)), 2), big_part_pct=round(100*float((s >= 2).mean()), 1),
                    misplaced_pct=round(100*float(np.mean([r["mis"] for r in rs])), 1), classes={k: round(100*v/len(rs), 1) for k, v in cl.most_common(5)})
    wn = feat["wants_none_near"] + feat["wants_none_far"]
    return dict(counts=dict(c), right=desc(feat["right"]), wants_none=desc(wn), wants_none_near=desc(feat["wants_none_near"]),
                wants_none_far=desc(feat["wants_none_far"])), rows

def sweep(R):
    out = []
    for bias in BIAS:
        c = collections.Counter()
        for b, e, lg in R:
            L = decode_bias(lg, bias)
            for i, t in enumerate(e["tgt"]):
                if e["ign"][i]: continue
                j = L.get(i); c["links"] += j is not None
                if t: c["linkable"] += 1; c["right"] += j in t; c["none"] += j is None
        acc, prec = c["right"]/c["linkable"], c["right"]/max(c["links"], 1)
        out.append(dict(bias=bias, acc=round(100*acc, 2), prec=round(100*prec, 2), f1=round(2*acc*prec/(acc+prec), 4), none_errors=c["none"], wrong=c["linkable"]-c["right"]))
    return out

def split_vote(R):
    """None errors where the model's total probability on texts beats none (p_none < 0.5) although no single text does; and a
    decoder that links every part whose p_none < 0.5 (none column closed for it), the rest as before."""
    c = collections.Counter(); dec = collections.Counter()
    for b, e, lg in R:
        p = F.softmax(lg, 1).numpy(); m = e["m"]; L = LF.decode(lg)
        x = lg.clone(); force = p[:, m] < 0.5; x[torch.from_numpy(force), -1] = -1e4; L2 = LF.decode(x)
        for i, t in enumerate(e["tgt"]):
            if e["ign"][i]: continue
            for name, LL in (("now", L), ("mass", L2)):
                j = LL.get(i); dec[name+"_links"] += j is not None
                if t: dec[name+"_linkable"] += 1; dec[name+"_right"] += j in t
            if t and L.get(i) is None and int(p[i].argmax()) == m:
                c["wants_none"] += 1; c["text_mass_wins"] += int(p[i, m] < 0.5); c["true_is_best_text"] += int(p[i, :m].argmax()) == t[0]
    out = dict(wants_none=c["wants_none"], text_mass_beats_none=c["text_mass_wins"], true_label_is_best_text=c["true_is_best_text"])
    for name in ("now", "mass"):
        acc, prec = dec[name+"_right"]/dec[name+"_linkable"], dec[name+"_right"]/max(dec[name+"_links"], 1)
        out[name] = dict(acc=round(100*acc, 2), prec=round(100*prec, 2), f1=round(2*acc*prec/(acc+prec), 4))
    return out

def examples(R, want):
    """Probability rows of a few real parts for the model page: top texts + none."""
    out = []
    for b, e, lg in R:
        if b["id"] not in want: continue
        p = F.softmax(lg, 1).numpy(); m = e["m"]; L = LF.decode(lg)
        for i, t in enumerate(e["tgt"]):
            row = sorted(((float(p[i, j]), b["texts"][j]["text"], j in t) for j in range(m)), reverse=True)[:4]
            out.append(dict(board=b["id"], part=i, cls=CLS[b["parts"][i]["cls"]], truth=b["texts"][t[0]]["text"] if t else "none",
                            pick=b["texts"][L[i]]["text"] if L.get(i) is not None else "none", none=round(float(p[i, m]), 3),
                            top=[dict(text=s, p=round(q, 3), true=tr) for q, s, tr in row]))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu"); A = ck["args"]; vocab = LF.Vocab(ck["vocab"]); d, Lr = LF.SIZES[A["size"]]; dev = torch.device(a.device)
    model = LF.LinkFormer(d, Lr, len(vocab.idx)+2, A["tokens"] == "E2").to(dev); model.load_state_dict(ck["model"]); model.eval(); run_ = Path(a.ckpt).stem
    val = set(json.load(open(D/"split.json"))["val_groups"])
    sets = {"kicad_val": [b for b in json.load(open(D/"kicad.json")) if b["group"] in val],
            "wacv_clean41": [b for b in json.load(open(D/"wacv_clean_v1.json")) if b["id"] not in LF.DEV]}
    res = {}
    for name, bs in sets.items():
        R = run(model, vocab, bs, dev); s, rows = analyse(R); s["bias_sweep"] = sweep(R); s["split_vote"] = split_vote(R); res[name] = s; res[name + "_rows"] = rows
        if name == "wacv_clean41": res["examples"] = examples(R, {"ML450_Top2", "Zedboard", "ATTIOT_Bottom"})
        print(name, json.dumps({k: v for k, v in s.items()}, ensure_ascii=False, indent=1), flush=True)
    json.dump(res, open(ROOT/f"results/linking/none_{run_}.json", "w"), indent=1, ensure_ascii=False)
