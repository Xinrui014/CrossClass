#!/usr/bin/env python3
"""Soft link output for the soft cascade (2026-10-17): like the end-to-end inference, but instead of the decoded link it keeps, for every
part (detector box), the full softmax over the board's text boxes + none (texts with p >= 0.005 kept). Text indices are converted to the
original index of e2e_boards_all.json when the boards carry an 'orig' field (rule-filtered boards). Output: {board: {part index:
{"none": p_none, "texts": [[text index, p], ...]}}}.
Usage: CROSSCLASS=... e2e_infer_soft.py CKPT BOARDS FEATS.npz OUT [--device cpu]   (default device: cuda)"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF, run_meta as RM
av = sys.argv[1:]; dv = "cuda"
if "--device" in av: k = av.index("--device"); dv = av[k+1]; del av[k:k+2]
ck, inp, fe, out = av[:4]; dev = torch.device(dv)
model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = getattr(model, "txtdim", 0); Z = np.load(fe); res = {}
with torch.no_grad():
    for b in json.load(open(inp)):
        if not (b["parts"] and b["texts"]): res[b["id"]] = {}; continue
        for t, f in zip(b["texts"], Z[b["id"]]): t["tf"] = f.astype(np.float32)
        lg = model(LF.to_dev(LF.encode(b, vocab), dev)).float(); pr = torch.softmax(lg, 1).cpu().numpy(); m = len(b["texts"])
        orig = [t.get("orig", j) for j, t in enumerate(b["texts"])]
        res[b["id"]] = {i: dict(none=round(float(pr[i, m]), 5), texts=[[int(orig[j]), round(float(pr[i, j]), 5)] for j in np.where(pr[i, :m] >= 0.005)[0]])
                        for i in range(pr.shape[0])}
json.dump(res, open(out, "w")); print("wrote", out, len(res))
RM.sidecar(out, [ck], dev, dict(boards=inp, feats=fe), decode="softmax (no decoding)", none_bias=0)
