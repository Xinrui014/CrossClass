#!/usr/bin/env python3
"""End-to-end class-head output of a proposal-B linker (2026-10-17): parts = detector boxes (e2e boards), their part tokens carry the
detector's real 9 class scores (data/preds_wacv_1280_cls9.json) when the model was trained with --detscore; texts = OCR boxes with PARSeq
features. Writes {board: {part index: {"cls": [9 class probabilities], "none": p_none, "texts": [[orig text index, p], ...]}}}.
Usage: CROSSCLASS=... e2e_infer_b.py CKPT BOARDS FEATS.npz OUT [--device cpu]   (default device: cuda)"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF, run_meta as RM
av = sys.argv[1:]; dv = "cuda"
if "--device" in av: k = av.index("--device"); dv = av[k+1]; del av[k:k+2]
ck, inp, fe, out = av[:4]; dev = torch.device(dv)
model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = getattr(model, "txtdim", 0); LF.DET_DIM = getattr(model, "detdim", 0)
p9 = os.environ.get("P9", os.path.join(os.environ["CROSSCLASS"], "data/preds_wacv_1280_cls9.json"))   # env P9: other test sets
LF.PIMG_DIM = getattr(model, "pimgdim", 0); pim = os.environ.get("PIMG", "")   # env PIMG: part image features of these boards' parts
assert not LF.PIMG_DIM or pim, "the model has part image features: set PIMG=data/partfeat_photo/<set>_det.npz"
Z = np.load(fe); P9 = json.load(open(p9)); res = {}; boards = json.load(open(inp))
if LF.PIMG_DIM: print("part image features:", LF.attach_pimg(boards, pim), "of", len(boards), "boards")
with torch.no_grad():
    for b in boards:
        if not (b["parts"] and b["texts"]): res[b["id"]] = {}; continue
        for t, f in zip(b["texts"], Z[b["id"]]): t["tf"] = f.astype(np.float32)
        if LF.DET_DIM:
            for p, k in zip(b["parts"], b["det_index"]):
                if P9[b["id"]][k][6] is not None: p["ds"] = np.array(P9[b["id"]][k][6:15], np.float32)
        e = LF.to_dev(LF.encode(b, vocab), dev); outs, cls = model.forward_batch([e], with_cls=True)
        pr = torch.softmax(outs[0].float(), 1).cpu().numpy(); pc = torch.softmax(cls[0].float(), 1).cpu().numpy(); m = len(b["texts"])
        orig = [t.get("orig", j) for j, t in enumerate(b["texts"])]
        res[b["id"]] = {i: dict(cls=[round(float(v), 5) for v in pc[i]], none=round(float(pr[i, m]), 5),
                                texts=[[int(orig[j]), round(float(pr[i, j]), 5)] for j in np.where(pr[i, :m] >= 0.005)[0]]) for i in range(pr.shape[0])}
json.dump(res, open(out, "w")); print("wrote", out, len(res))
RM.sidecar(out, [ck], dev, dict(boards=inp, feats=fe, preds_cls9=p9, **({"pimg": pim} if LF.PIMG_DIM else {})), decode="softmax (no decoding)", none_bias=0)
