#!/usr/bin/env python3
"""Linker with PARSeq text-image features on end-to-end boards: attach each text's 688-d feature ("tf"); with --parseq_strings the text
string becomes PARSeq's own reading (argmax of the feature's character probabilities), for string models. Writes {board: {part: [text
index or null, link prob]}}. Usage: CROSSCLASS=... e2e_infer_tf.py CKPT BOARDS FEATS.npz OUT [--parseq_strings]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["CROSSCLASS"], "tools")); import linkformer as LF
ck, inp, fe, out = sys.argv[1:5]; ps = "--parseq_strings" in sys.argv; dev = torch.device("cuda")
model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = getattr(model, "txtdim", 0); Z = np.load(fe); res = {}
CL = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"; read = lambda r: "".join(CL[c] for c in r[:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
with torch.no_grad():
    for b in json.load(open(inp)):
        if not (b["parts"] and b["texts"]): res[b["id"]] = {}; continue
        F = Z[b["id"]]
        for t, f in zip(b["texts"], F):
            t["tf"] = f.astype(np.float32)
            if ps: t["text"] = read(f)
        lg = model(LF.to_dev(LF.encode(b, vocab), dev)).float(); L = LF.decode(lg); pr = torch.softmax(lg, 1).cpu(); m = len(b["texts"])
        res[b["id"]] = {i: [j, float(pr[i, j if j is not None else m])] for i, j in L.items()}
json.dump(res, open(out, "w")); print("wrote", out, len(res))
