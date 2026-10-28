#!/usr/bin/env python3
"""Hungarian link per part of an e2e class-head run (task 26, 2026-10-24): the same forward pass as tools/e2e_infer_b.py, then
LF.decode(link logits, "hung") (Hungarian on -log p, own none column per part). For the sel class heads this is the link the class head
used (model._links). Writes {board: {part index: [orig text index or -1, p of that text, p_none]}} to <BOUT>.hung.json, and checks that
the class probabilities equal the stored bout file (max abs difference printed).
Usage: CROSSCLASS=... P9=... PIMG=... e2e_infer_hung.py CKPT BOARDS FEATS.npz BOUT [--device cpu]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF
av = sys.argv[1:]; dv = "cpu"
if "--device" in av: k = av.index("--device"); dv = av[k+1]; del av[k:k+2]
ck, inp, fe, bout = av[:4]; dev = torch.device(dv)
model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = getattr(model, "txtdim", 0); LF.DET_DIM = getattr(model, "detdim", 0)
P9 = json.load(open(os.environ["P9"])); LF.PIMG_DIM = getattr(model, "pimgdim", 0); pim = os.environ.get("PIMG", "")
assert not LF.PIMG_DIM or pim, "the model has part image features: set PIMG"
Z = np.load(fe); boards = json.load(open(inp)); ref = json.load(open(bout)); res = {}; dmax = 0.0
if LF.PIMG_DIM: LF.attach_pimg(boards, pim)
with torch.no_grad():
    for b in boards:
        if not (b["parts"] and b["texts"]): res[b["id"]] = {}; continue
        for t, f in zip(b["texts"], Z[b["id"]]): t["tf"] = f.astype(np.float32)
        if LF.DET_DIM:
            for p, k in zip(b["parts"], b["det_index"]):
                if P9[b["id"]][k][6] is not None: p["ds"] = np.array(P9[b["id"]][k][6:15], np.float32)
        e = LF.to_dev(LF.encode(b, vocab), dev); outs, cls = model.forward_batch([e], with_cls=True)
        links = model._links[0] if getattr(model, "clsmode", "") == "sel" else LF.decode(outs[0])
        pr = torch.softmax(outs[0].float(), 1).cpu().numpy(); pc = torch.softmax(cls[0].float(), 1).cpu().numpy(); m = len(b["texts"])
        orig = [t.get("orig", j) for j, t in enumerate(b["texts"])]
        res[b["id"]] = {i: [int(orig[j]) if j is not None else -1, round(float(pr[i, j]), 5) if j is not None else 0.0, round(float(pr[i, m]), 5)]
                        for i, j in ((i, links.get(i)) for i in range(pr.shape[0]))}
        r = ref.get(b["id"], {})
        for i in range(pr.shape[0]):
            if str(i) in r: dmax = max(dmax, float(np.abs(np.array(r[str(i)]["cls"]) - pc[i]).max()))
json.dump(res, open(bout + ".hung.json", "w")); print("wrote", bout + ".hung.json", len(res), f"max |cls - stored cls| {dmax:.2e}")
