#!/usr/bin/env python3
"""In-domain check of the noise boxes (2026-10-17): KiCad val sides (kval, held out by repo group) with R noise boxes per label added
(the side's own noise boxes from linkdata/kicad_noise + PARSeq features, sampled with a fixed seed per side; no other augmentation).
Reports link accuracy / precision on the labelled parts and how many noise boxes get linked to a part (false links).
Usage: CROSSCLASS=... eval_kval_noise.py CKPT [CKPT ...] [--ratios 0,0.5,1] [--device cuda]"""
import argparse, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.environ.get("CROSSCLASS", os.path.expanduser("~/projects/crossclass")), "tools")); import linkformer as LF
ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--ratios", default="0,0.5,1"); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); R = [float(x) for x in a.ratios.split(",")]
sets = LF.load_sets("CrossClass_data_v2"); kval = []
for b in sets["kval"]:
    f = LF.ROOT/f"linkdata/kicad_textfeat/{b['id']}.npy"; nb, nf = LF.ROOT/f"linkdata/kicad_noise/{b['id']}.json", LF.ROOT/f"linkdata/kicad_noisefeat/{b['id']}.npy"
    if not (f.exists() and nb.exists() and nf.exists() and b["parts"] and b["texts"]): continue
    F = np.load(f)
    if len(F) != len(b["texts"]): continue
    for t, v in zip(b["texts"], F): t["tf"] = v.astype(np.float32)
    bx, fx = json.load(open(nb))["boxes"], np.load(nf)
    b["noise"] = [dict(box=x, text="", angle=None, noise=True, tf=v.astype(np.float32)) for x, v in zip(bx, fx)]; kval.append(b)
print(f"kval sides with features and noise boxes: {len(kval)}")
for ck in a.ckpts:
    model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = A.get("txtdim", 0); LF.IMG_DIM = 0
    for r in R:
        per = {}; nz = nzl = 0
        for b in kval:
            rng = random.Random(b["id"]); k = min(len(b["noise"]), int(round(r*len(b["texts"])))); m0 = len(b["texts"])
            bb = dict(b, texts=b["texts"] + rng.sample(b["noise"], k)); e = LF.encode(bb, vocab)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                L = LF.decode(model(LF.to_dev(e, dev)).float().cpu())
            per[b["id"]] = LF.count(e, L); nz += k; nzl += sum(1 for j in L.values() if j is not None and j >= m0)
        s = LF.summarize(per)
        print(f"{os.path.basename(ck):32s} noise {r:.1f}: link acc {100*s['acc']:.1f} prec {100*s['prec']:.1f}  noise boxes {nz}, linked to a part {nzl} ({100*nzl/max(nz,1):.1f} %)", flush=True)
