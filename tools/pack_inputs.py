#!/usr/bin/env python3
"""Pack the per-side input folders of LinkFormer into ONE file each (task 36, owner 2026-10-25 17:05: the per-file md5 fingerprints and the
small-file reads were very slow on cluster02 network storage). The folders are kept. Packed formats (read by linkformer.Store):
  .npy folders  -> uncompressed .npz, key = side id                      (kicad_textfeat: only <id>.npy; <id>.stat.json is not read)
  .npz folders  -> uncompressed .npz, keys <side id>__<array name>       (part image features: feat, ok)
  .json folders -> one .json {side id: content}                          (noise boxes, fptype v2 / v3)
After writing, every side is read back through linkformer.Store from the folder and from the packed file and compared exactly (same keys,
dtypes and values). Usage: CROSSCLASS=... pack_inputs.py [name ...]   (default: all)"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); sys.path.insert(0, str(ROOT/"tools"))
import linkformer as LF
JOBS = {"kicad_textfeat": ("linkdata/kicad_textfeat", "npy", "linkdata/kicad_textfeat.npz"),
        "kicad_noisefeat": ("linkdata/kicad_noisefeat", "npy", "linkdata/kicad_noisefeat.npz"),
        "kicad_noise": ("linkdata/kicad_noise", "json", "linkdata/kicad_noise.json"),
        "out_paste_v1": ("data/partfeat_render/out_paste_v1", "npz", "data/partfeat_render/out_paste_v1.npz"),
        "out_paste_v1_e32": ("data/partfeat_render/out_paste_v1_e32", "npz", "data/partfeat_render/out_paste_v1_e32.npz"),   # Syn pilot (2026-10-26): KiCad v8l-P2 e32 features
        "out_paste_v1_e60e31": ("data/partfeat_render/out_paste_v1_e60e31", "npz", "data/partfeat_render/out_paste_v1_e60e31.npz"),   # final Syn detector (2026-10-27): det_kicad_v8lp2_e60 best.pt = e31
        "kicad_fptype_v2": ("linkdata/kicad_fptype_v2", "json", "linkdata/kicad_fptype_v2.json"),
        "kicad_fptype_v3": ("linkdata/kicad_fptype_v3", "json", "linkdata/kicad_fptype_v3.json")}
def same(a, b):
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b
for name in (sys.argv[1:] or list(JOBS)):
    src, ext, dst = JOBS[name]; S, D = ROOT/src, ROOT/dst; t0 = time.time()
    ids = sorted(f.name[:-len(ext) - 1] for f in S.iterdir() if f.name.endswith("." + ext) and not f.name.endswith(".stat.json"))
    assert not D.exists(), f"{D} exists"
    if ext == "json": json.dump({k: json.load(open(S/f"{k}.json")) for k in ids}, open(D, "w"))
    elif ext == "npy": np.savez(D, **{k: np.load(S/f"{k}.npy") for k in ids})
    else:
        arr = {}
        for k in ids:
            z = np.load(S/f"{k}.npz"); arr.update({f"{k}__{n}": z[n] for n in z.files})
        np.savez(D, **arr)
    t1 = time.time(); Fs, Ps = LF.Store(S, ext), LF.Store(D, ext)
    bad = [k for k in ids if not same(Fs.get(k), Ps.get(k))]
    print(f"{name}: {len(ids)} sides -> {dst} ({D.stat().st_size / 1e9:.2f} GB) in {t1 - t0:.0f} s; check {time.time() - t1:.0f} s: "
          f"{len(ids) - len(bad)} equal, {len(bad)} different {bad[:3]}; missing id -> {Ps.get('no_such_side')}", flush=True)
