#!/usr/bin/env python3
"""Pack / unpack the KiCad noise boxes + PARSeq features for the slow bastion copy to intern02 (2026-10-16).
pack (L40):    linkdata/kicad_noise/*.json + kicad_noisefeat/*.npy -> OUT.npz: boxes (N,4) f32, probs (N,304) u8 (= p*255), tok (N,384) u8
               (per-dimension min / max over all boxes), lo / hi (384,), ids, offsets. Feature error: probs <= 0.002, tok <= range/510.
unpack (intern02): OUT.npz -> linkdata/kicad_noise/<side>.json {id, boxes} and kicad_noisefeat/<side>.npy (fp16), as linkformer --noise reads.
Usage: noise_pack.py pack|unpack ROOT NPZ"""
import glob, json, os, sys
import numpy as np
mode, R, Z = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "pack":
    ids, B, F = [], [], []
    for f in sorted(glob.glob(f"{R}/linkdata/kicad_noise/*.json")):
        s = json.load(open(f)); a = np.load(f"{R}/linkdata/kicad_noisefeat/{s['id']}.npy"); assert len(a) == len(s["boxes"])
        ids.append(s["id"]); B.append(np.array(s["boxes"], np.float32).reshape(-1, 4)); F.append(a.astype(np.float32))
    off = np.cumsum([0] + [len(b) for b in B]); B = np.concatenate(B); F = np.concatenate(F)
    probs = np.clip(np.round(F[:, :304]*255), 0, 255).astype(np.uint8); t = F[:, 304:]; lo, hi = t.min(0), t.max(0)
    tok = np.clip(np.round((t-lo)/np.maximum(hi-lo, 1e-6)*255), 0, 255).astype(np.uint8)
    np.savez(Z, boxes=B, probs=probs, tok=tok, lo=lo, hi=hi, ids=np.array(ids), off=off); print("packed", len(ids), "sides", len(B), "boxes")
else:
    d = np.load(Z); lo, hi, off = d["lo"], d["hi"], d["off"]; os.makedirs(f"{R}/linkdata/kicad_noise", exist_ok=True); os.makedirs(f"{R}/linkdata/kicad_noisefeat", exist_ok=True)
    F = np.concatenate([d["probs"].astype(np.float32)/255, d["tok"].astype(np.float32)/255*(hi-lo)+lo], 1).astype(np.float16); B = d["boxes"]
    for k, sid in enumerate(d["ids"]):
        sid = str(sid); a, b = off[k], off[k+1]
        json.dump(dict(id=sid, boxes=np.round(B[a:b], 3).tolist()), open(f"{R}/linkdata/kicad_noise/{sid}.json", "w")); np.save(f"{R}/linkdata/kicad_noisefeat/{sid}.npy", F[a:b])
    print("unpacked", len(d["ids"]), "sides", len(B), "boxes")
