#!/usr/bin/env python3
"""Class-head linker on end-to-end boards (e2e_build.py): per part the Hungarian link, its probability, and the class head's argmax and
probability. Writes {board: {part index: [text index or null, link prob, class, class prob]}}. Usage: CROSSCLASS=... e2e_infer_cls.py CKPT BOARDS OUT"""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF
ck, inp, out = sys.argv[1:4]; dev = torch.device("cuda")
model, vocab, A = LF.load_model(ck, dev); res = {}
with torch.no_grad():
    for b in json.load(open(inp)):
        if not b["parts"]: res[b["id"]] = {}; continue
        if not b["texts"]: b = dict(b, texts=[dict(box=[-1e4, -1e4, -1e4+1, -1e4+1], text="", angle=None)])   # a dummy far text keeps the forward valid
        e = LF.to_dev(LF.encode(b, vocab), dev); outs, cls = model.forward_batch([e], with_cls=True)
        lg = outs[0].float(); L = LF.decode(lg); pr = torch.softmax(lg, 1).cpu(); pc = torch.softmax(cls[0], 1).cpu(); m = len(b["texts"])
        res[b["id"]] = {i: [j, float(pr[i, j if j is not None else m]), int(pc[i].argmax()), float(pc[i].max())] for i, j in L.items()}
json.dump(res, open(out, "w")); print("wrote", out, len(res))
