#!/usr/bin/env python3
"""Test-time board offset prior for the linker (2026-10-16, owner idea: add a distance / clustering prior to push link accuracy).
On one board the labels sit at similar offsets from their parts (e.g. all above). First pass: the model's links; the confident ones
(p >= CONF) give the board's offset distribution (text centre - part centre, in median-part units, per part orientation when there are
enough samples; Gaussian kernel density, leave-one-out). Second pass: logit_ij + beta * log(prior_ij / max_j prior_ij) for every text
column (the none column unchanged), Hungarian again. beta / sigma are chosen on clean_dev3 and reported on clean_test41 (grid printed).
Usage: CROSSCLASS=... link_prior.py CKPT [--txtfeat_wacv NPZ] [--device cuda]"""
import argparse, itertools, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.environ.get("CROSSCLASS", os.path.expanduser("~/projects/crossclass")), "tools")); import linkformer as LF
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--txtfeat_wacv", default="linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz")
ap.add_argument("--device", default="cuda"); ap.add_argument("--conf", type=float, default=0.9); a = ap.parse_args()
dev = torch.device(a.device); model, vocab, A = LF.load_model(a.ckpt, dev); LF.TXT_DIM = A.get("txtdim", 0); LF.IMG_DIM = 0
sets = LF.load_sets(A["data"]); Z = np.load(LF.ROOT/a.txtfeat_wacv) if LF.TXT_DIM else None
cache = {}
for name in ("clean_dev3", "clean_test41"):
    out = []
    for b in sets[name]:
        if not (b["parts"] and b["texts"]): continue
        if Z is not None:
            for t, v in zip(b["texts"], Z[f"{b['id']}__t"]): t["tf"] = v.astype(np.float32)
        e = LF.encode(b, vocab)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(LF.to_dev(e, dev)).float().cpu().numpy()
        B = e["box"].numpy(); n = e["n"]; c = (B[:, :2] + B[:, 2:]) / 2; vert = (B[:n, 3]-B[:n, 1]) > (B[:n, 2]-B[:n, 0])
        out.append(dict(e=e, lg=lg, off=c[None, n:] - c[:n, None], vert=vert))
    cache[name] = out

def run(boards, beta, sigma):
    per = {}
    for k, r in enumerate(boards):
        lg = r["lg"].copy(); n, m1 = lg.shape; m = m1 - 1
        if beta:
            L0 = LF.decode(torch.tensor(lg)); p = torch.softmax(torch.tensor(lg), 1).numpy()
            conf = [(i, j) for i, j in L0.items() if j is not None and p[i, j] >= a.conf]
            if len(conf) >= 3:
                for i in range(n):
                    same = [(ii, jj) for ii, jj in conf if ii != i and r["vert"][ii] == r["vert"][i]]
                    pool = same if len(same) >= 5 else [(ii, jj) for ii, jj in conf if ii != i]
                    if not pool: continue
                    V = np.array([r["off"][ii, jj] for ii, jj in pool])                    # (K, 2) confident offsets
                    d2 = ((r["off"][i][:, None, :] - V[None]) ** 2).sum(-1)                 # (m, K)
                    pr = np.exp(-d2 / (2*sigma**2)).mean(1) + 1e-6
                    lg[i, :m] += beta * np.log(pr / pr.max())
        per[r["e"]["id"]] = LF.count(r["e"], LF.decode(torch.tensor(lg)))
    return LF.summarize(per)

grid = [(0.0, 1.0)] + list(itertools.product([0.25, 0.5, 1.0, 2.0], [0.25, 0.5, 1.0, 2.0]))
res = {g: (run(cache["clean_dev3"], *g), run(cache["clean_test41"], *g)) for g in grid}
print(f"{'beta':>5} {'sigma':>5}   dev3 acc / prec    test41 acc / prec")
for (be, si), (d, t) in res.items(): print(f"{be:5.2f} {si:5.2f}   {d['acc']:.4f} / {d['prec']:.4f}   {t['acc']:.4f} / {t['prec']:.4f}")
best = max(grid[1:], key=lambda g: (res[g][0]["acc"], res[g][0]["prec"]))
print(f"chosen on dev3: beta {best[0]} sigma {best[1]} -> clean_test41 acc {res[best][1]['acc']:.4f} (no prior {res[grid[0]][1]['acc']:.4f}), "
      f"prec {res[best][1]['prec']:.4f} ({res[grid[0]][1]['prec']:.4f})")
