#!/usr/bin/env python3
"""Link accuracy on 21PCB, oracle track (GT parts + GT texts), same counting as WACV-clean clean_test41 (tools/linkformer.py count /
summarize: acc = right / linkable parts, prec = right / links, Hungarian decoding, one-to-one, none allowed).
GT: data/21pcb/link_gt_v1.json (owner, 2026-10-21; texts = every box the owner kept, linked or not). 28 photos; batch 1 = 12
design-aligned photos (owner-checked text boxes), batch 2 = 16 photos (texts = OCR boxes the owner kept).
  prep            -> data/21pcb/link_eval/boards.json (WACV-clean board format: parts {box, des, cls, link}, texts {box, text})
  then            text_features.py boxes --boards data/21pcb/link_eval/boards.json --imgdir data/21pcb/images --out data/21pcb/link_eval/tf.npz
  eval CKPT TF    -> results/pcb21/link_eval_<ckpt>.json: LinkFormer (Hungarian) and a nearest-text baseline (Hungarian on edge gap /
                  part size, gap < 2), total and per batch
  pcb21_link_eval.py prep | eval ckpt/intern02/v2d2_I_noise05_100k.pt data/21pcb/link_eval/tf.npz   (env ~/venvs/gdino_fewshot)"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; OUT = D/"link_eval"
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]

def prep():
    L = json.load(open(D/"link_gt_v1.json")); boards = []
    for st, g in sorted(L.items()):
        boards.append(dict(id=st, source="21pcb", board=st.split("_")[0], W=g["W"], H=g["H"], batch=g["batch"], clean="21",
                           parts=[dict(box=p["box"], des=p.get("des", ""), cls=CLS.index(p["cls"]) if p["cls"] in CLS else -1,
                                       link=(p["text"] if p["text"] >= 0 else None)) for p in g["parts"]],
                           texts=[dict(box=t["box"], text=t.get("str", ""), angle=None) for t in g["texts"]]))
    OUT.mkdir(parents=True, exist_ok=True); json.dump(boards, open(OUT/"boards.json", "w"))
    print(len(boards), "boards,", sum(len(b["parts"]) for b in boards), "parts,", sum(len(b["texts"]) for b in boards), "texts,",
          sum(1 for b in boards for p in b["parts"] if p["link"] is not None), "linkable parts ->", OUT/"boards.json")

def nearest(b):
    from scipy.optimize import linear_sum_assignment
    P, T = b["parts"], b["texts"]; L = {}
    if not (P and T): return L
    C = np.full((len(P), len(T)), 1e6)
    for i, p in enumerate(P):
        sz = max(p["box"][2]-p["box"][0], p["box"][3]-p["box"][1], 1e-6)
        for j, t in enumerate(T):
            a, c = p["box"], t["box"]; g = np.hypot(max(0, max(a[0], c[0]) - min(a[2], c[2])), max(0, max(a[1], c[1]) - min(a[3], c[3]))) / sz
            if g < 2: C[i, j] = g
    for i, j in zip(*linear_sum_assignment(C)):
        if C[i, j] < 1e6: L[i] = j
    return L

def evaluate(ck, tf):
    import torch
    sys.path.insert(0, str(ROOT/"tools")); import linkformer as LF
    dev = torch.device("cpu"); torch.set_num_threads(16)
    model, vocab, A = LF.load_model(ck, dev); LF.TXT_DIM = getattr(model, "txtdim", 0); LF.DET_DIM = getattr(model, "detdim", 0)
    boards = json.load(open(OUT/"boards.json")); Z = np.load(tf)
    for b in boards:
        for t, f in zip(b["texts"], Z[b["id"]]): t["tf"] = f.astype(np.float32)
    encs = [LF.encode(b, vocab) for b in boards]
    _, per = LF.evaluate(model, encs, dev, "hung")
    pn = {b["id"]: LF.count(e, nearest(b)) for b, e in zip(boards, encs)}
    res = {}
    for name, sel in (("all", lambda b: True), ("batch1", lambda b: b["batch"] == 1), ("batch2", lambda b: b["batch"] == 2)):
        ids = [b["id"] for b in boards if sel(b)]
        res[name] = dict(linkformer=LF.summarize({i: per[i] for i in ids}), nearest=LF.summarize({i: pn[i] for i in ids}), photos=len(ids))
    res["per_photo"] = {i: dict(linkformer=per[i], nearest=pn[i]) for i in per}
    out = ROOT/"results/pcb21"/f"link_eval_{Path(ck).stem}.json"; json.dump(res, open(out, "w"), indent=1)
    for name in ("all", "batch1", "batch2"):
        r = res[name]; print(f"{name:7s} ({r['photos']:2d} photos)  LinkFormer acc {r['linkformer']['acc']:.4f} prec {r['linkformer']['prec']:.4f} "
                             f"({r['linkformer']['right']}/{r['linkformer']['linkable']})   nearest acc {r['nearest']['acc']:.4f} prec {r['nearest']['prec']:.4f}")
    print("->", out)

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep()
    elif sys.argv[1] == "eval": evaluate(sys.argv[2], sys.argv[3])
