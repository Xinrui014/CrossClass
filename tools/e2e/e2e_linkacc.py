#!/usr/bin/env python3
"""End-to-end link accuracy on WACV-clean v1 test photos (41), same metric as the oracle 'acc' (parts with a label: right = linked to
the label's text box). Chain per labelled part: detector box (IoU >= 0.5) -> OCR candidate box on the label (covers >= 50 % of the
annotated text box) -> linker picks that box (I_100k, p >= 0.5). Also the oracle-box comparison: the same linker on annotated boxes.
Writes per-part rows to OUT (json) for the example figures. Usage: DIR OUT"""
import json, os, sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(os.path.expanduser("~/projects/crossclass")); LD = Path(sys.argv[1]); OUT = sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else "all"; RUN = sys.argv[4] if len(sys.argv) > 4 else "I100k"
sys.path.insert(0, str(ROOT/"tools")); os.chdir(ROOT/"data"); from analyze9 import iou
DEV = {"ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"}
W = {w["id"]: w for w in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json"))}
B = {b["id"]: b for b in json.load(open(LD/f"e2e_boards_{MODE}.json"))}; Lk = json.load(open(LD/f"e2e_tf_links_{RUN}_{MODE}.json"))
def cover(a, b):  # fraction of box b covered by box a
    a, b = np.array(a, float), np.array(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[2], b[:, 2])-np.maximum(a[0], b[:, 0]), 0, None); ih = np.clip(np.minimum(a[3], b[:, 3])-np.maximum(a[1], b[:, 1]), 0, None)
    return iw*ih/np.maximum((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]), 1e-9)
cnt = Counter(); rows = []
for st, w in W.items():
    if st in DEV or st not in B: continue
    b = B[st]; db = np.array([p["box"] for p in b["parts"]]).reshape(-1, 4); ob = [t["box"] for t in b["texts"]]
    I = iou(np.array([p["box"] for p in w["parts"]]).reshape(-1, 4), db) if len(db) else None
    for pi, p in enumerate(w["parts"]):
        if p.get("ign") or p.get("link") is None: continue
        cnt["labelled parts"] += 1; gt = w["texts"][p["link"]]
        row = dict(img=st, part=pi, des=p["des"], cls=p["cls"], pbox=p["box"], tbox=gt["box"], gtext=gt["text"])
        if I is None or I[pi].max() < 0.5: row["stage"] = "detector missed the part"; rows.append(row); cnt[row["stage"]] += 1; continue
        i = int(np.argmax(I[pi])); row["det"] = i; row["dbox"] = b["parts"][i]["box"]; row["dcls"] = b["parts"][i]["cls"]
        ok = [j for j, o in enumerate(ob) if cover(o, gt["box"])[0] >= 0.5 or cover(gt["box"], o)[0] >= 0.5]
        j, pr = Lk[st].get(str(i), (None, 0.0)); row["raw"] = j; j = j if (j is not None and pr >= 0.5) else None
        row.update(link=j, prob=pr, lbox=(ob[j] if j is not None else None), ltext=(b["texts"][j]["text"] if j is not None else None))
        if not ok: row["stage"] = "OCR has no box on the label"
        elif j in ok: row["stage"] = "linked to the right text"
        elif j is None: row["stage"] = "linked to none"
        else: row["stage"] = "linked to a wrong text"
        cnt[row["stage"]] += 1; rows.append(row)
for k, v in cnt.most_common(): print(f"{k:32s} {v}")
n = cnt["labelled parts"]; f = n-cnt["detector missed the part"]; o = f-cnt["OCR has no box on the label"]
print(f"e2e link acc (all labelled parts): {cnt['linked to the right text']/n:.3f}; given detector found: {cnt['linked to the right text']/f:.3f};"
      f" given detector + OCR found: {cnt['linked to the right text']/o:.3f}")
import re
DES = re.compile(r"^[A-Z]+\d+[A-Z]?$")
wr = [r for r in rows if r["stage"] == "linked to a wrong text"]; nn = [r for r in rows if r["stage"] == "linked to none"]
print("wrong text is designator-like:", sum(bool(DES.match(r["ltext"] or "")) for r in wr), "of", len(wr))
print("none: linker chose a text but p < 0.5:", sum(r["raw"] is not None for r in nn), "of", len(nn))
json.dump(rows, open(OUT, "w"))
