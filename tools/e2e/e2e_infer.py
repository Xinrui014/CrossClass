#!/usr/bin/env python3
"""Run a linkformer checkpoint on end-to-end boards (e2e_build.py): per board, the Hungarian links and the probability of the chosen
column (softmax over the part's texts + none). Writes {board: {part index: [text index or null, prob]}}.
Usage: CROSSCLASS=... e2e_infer.py CKPT BOARDS.json OUT.json"""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF
ck, inp, out = sys.argv[1:4]; dev = torch.device("cuda")
model, vocab, A = LF.load_model(ck, dev); res = {}
with torch.no_grad():
    for b in json.load(open(inp)):
        if not (b["parts"] and b["texts"]): res[b["id"]] = {}; continue
        lg = model(LF.to_dev(LF.encode(b, vocab), dev)).float(); L = LF.decode(lg); pr = torch.softmax(lg, 1).cpu()
        m = len(b["texts"]); res[b["id"]] = {i: [j, float(pr[i, j if j is not None else m])] for i, j in L.items()}
json.dump(res, open(out, "w")); print("wrote", out, len(res))
