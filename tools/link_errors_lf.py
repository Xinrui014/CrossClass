#!/usr/bin/env python3
"""What the remaining link errors on WACV-clean are (2026-10-16, owner target 98 %). Per labelled part of clean_test41 / clean_dev3:
right / none (the model gave no text) / wrong text; whether the label is 'messy' (printed >= 0.5 median part sizes nearer, by edge gap,
to another part: linkformer.messy) and whether the chosen wrong text is the nearest text to the part; errors per board.
Usage: CROSSCLASS=... link_errors_lf.py CKPT [--txtfeat_wacv NPZ] [--device cpu]"""
import argparse, collections, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.environ.get("CROSSCLASS", os.path.expanduser("~/projects/crossclass")), "tools")); import linkformer as LF
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--txtfeat_wacv", default="linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz")
ap.add_argument("--device", default="cpu"); a = ap.parse_args()
dev = torch.device(a.device); model, vocab, A = LF.load_model(a.ckpt, dev); LF.TXT_DIM = A.get("txtdim", 0); LF.IMG_DIM = 0
sets = LF.load_sets(A["data"]); Z = np.load(LF.ROOT/a.txtfeat_wacv) if LF.TXT_DIM else None
for name in ("clean_test41", "clean_dev3"):
    c = collections.Counter(); byb = collections.Counter(); nb = collections.Counter()
    for b in sets[name]:
        if not (b["parts"] and b["texts"]): continue
        if Z is not None:
            for t, v in zip(b["texts"], Z[f"{b['id']}__t"]): t["tf"] = v.astype(np.float32)
        e = LF.encode(b, vocab); ms = LF.messy(b)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            L = LF.decode(model(LF.to_dev(e, dev)).float().cpu())
        P = np.array([p["box"] for p in b["parts"]], float); T = np.array([t["box"] for t in b["texts"]], float)
        gx = np.maximum(np.maximum(T[None, :, 0]-P[:, None, 2], P[:, None, 0]-T[None, :, 2]), 0); gy = np.maximum(np.maximum(T[None, :, 1]-P[:, None, 3], P[:, None, 1]-T[None, :, 3]), 0)
        near = np.hypot(gx, gy).argmin(1)
        for i, t in enumerate(e["tgt"]):
            if (e["ign"] and e["ign"][i]) or not t: continue
            j = L.get(i); k = "messy" if ms[i] else "clean"; nb[b["id"]] += 1
            if j in t: c[f"{k}: right"] += 1
            elif j is None: c[f"{k}: none"] += 1; byb[b["id"]] += 1
            else: c[f"{k}: wrong text ({'nearest' if j == near[i] else 'not nearest'})"] += 1; byb[b["id"]] += 1
    n = sum(c.values()); right = sum(v for k, v in c.items() if k.endswith("right"))
    print(f"== {name}: {n} labelled parts, acc {right/n:.4f}")
    for k in sorted(c): print(f"  {k:36s} {c[k]:5d}")
    mess = sum(v for k, v in c.items() if k.startswith("messy")); mr = c["messy: right"]
    print(f"  messy labels {mess} ({mess/n:.1%}), acc on messy {mr/max(mess,1):.3f}, on clean {c['clean: right']/max(n-mess,1):.3f}")
    print("  errors per board (top 8):", ", ".join(f"{s} {v}/{nb[s]}" for s, v in byb.most_common(8)))
