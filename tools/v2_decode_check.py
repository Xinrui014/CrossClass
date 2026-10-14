#!/usr/bin/env python3
"""Decoding variants for a v2 (--head lg) checkpoint, scored on KiCad val (or kvalsub) and WACV-clean v1 test (clean41).
  hung     Hungarian on -log P_ij, none cost -log(1 - sigma_i)                  (linkformer default)
  sig      parts with sigma_i >= 0.5 must link, the others take none
  thr<t>   link only if P_ij >= t (Hungarian on -log P, none cost -log t), as LightGlue thresholds P
  nocol    P'_ij = sigma_i tau_j softmax_row(S)_ij (column softmax dropped), none cost -log(1 - sigma_i)
Usage: v2_decode_check.py --ckpt ckpt/smoke/s15k_A.pt [--kicad kvalsub] [--device cuda]   (python: ~/venvs/gdino_fewshot)"""
import argparse, json, sys, math, hashlib, collections
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import linkformer as LF

def hung(cost_text, cost_none):
    n, m = cost_text.shape; C = np.full((n, m+n), 1e6); C[:, :m] = cost_text; C[np.arange(n), m+np.arange(n)] = cost_none
    r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}

MODES = ["hung", "sig", "thr0.2", "thr0.1", "thr0.05", "nocol"]
def decode_all(out):
    S, ls, lt = (o.float() for o in out)
    lp = (F.logsigmoid(ls)[:, None] + F.logsigmoid(lt)[None, :] + F.log_softmax(S, 1) + F.log_softmax(S, 0)).cpu().numpy()
    lnc = (F.logsigmoid(ls)[:, None] + F.logsigmoid(lt)[None, :] + F.log_softmax(S, 1)).cpu().numpy()
    lnone = F.logsigmoid(-ls).cpu().numpy(); sig = torch.sigmoid(ls).cpu().numpy()
    res = {"hung": hung(-lp, -lnone), "sig": hung(-lp, np.where(sig >= 0.5, 1e3, 0.0)), "nocol": hung(-lnc, -lnone)}
    for t in (0.2, 0.1, 0.05): res[f"thr{t}"] = hung(-lp, np.full(len(lnone), -math.log(t)))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--kicad", default="kvalsub"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); dev = torch.device(a.device); model, vocab, A = LF.load_model(a.ckpt, dev)
    D = LF.ROOT/"linkdata/CrossClass_data_v1"; val = set(json.load(open(D/"split.json"))["val_groups"])
    kv = [b for b in json.load(open(D/"kicad.json")) if b["group"] in val]
    if a.kicad == "kvalsub": kv = [b for b in kv if int(hashlib.md5(b["id"].encode()).hexdigest(), 16) % 4 == 0]
    sets = {a.kicad: kv, "clean41": [b for b in json.load(open(D/"wacv_clean_v1.json")) if b["id"] not in LF.DEV]}
    for name, bs in sets.items():
        tot = {k: collections.Counter() for k in MODES}
        with torch.no_grad():
            for b in bs:
                if not (b["parts"] and b["texts"]): continue
                e = LF.encode(b, vocab); res = decode_all(model(LF.to_dev(e, dev)))
                for k in MODES: tot[k].update(LF.count(e, res[k]))
        print(name, " | ".join(f"{k}: {LF.summarize({'x': tot[k]})['acc']*100:.1f} / {LF.summarize({'x': tot[k]})['prec']*100:.1f}" for k in MODES), flush=True)
