#!/usr/bin/env python3
"""Build the final 21PCB ground truth (28 photos with IC annotation) from the owner's labelme files.
- Components: owner_final2/pcb21_batch1_class/ (12 design-aligned photos) or owner_final2/pcb21_batch2/ (16 photos).
- Text boxes (designator labels): owner_final2/pcb21_batch1/ for the 12 design-aligned photos only (strings still to confirm).
- The owner's class is final. Where a box's designator disagrees with the BoM class of that designator (results/pcb21/final_check.json,
  check A), the designator is wrong (it came from a neighbouring label) and is cleared, except the stems/refs listed in KEEP
  (open questions for the owner).
Outputs: data/21pcb/final_v1/labelme/<stem>.{json,jpg} (owner's files + cleared designators + texts merged) and
data/21pcb/final_v1/gt9_21pcb.json = {stem: {"W", "H", "boxes": [[x1,y1,x2,y2, cls9 (-1 = ignore), label, designator]],
"texts": [[x1,y1,x2,y2, string]]}} in native photo pixels (same box format as gdino_fewshot/err9/build_gt9.py).
  pcb21_build_final.py
Env PCB21_SRC as in tools/pcb21_final_check.py (run that first with the same value): components from one folder
(2026-10-20: owner_final3/pcb21_final, the owner's GT check; it overwrites final_v1, owner's choice)."""
import json, os, shutil
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; OUT = D/"final_v1"
SRC = os.environ.get("PCB21_SRC", "")
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
KEEP = {("Card2_BTMSPAUDSOURCE_T", "J3")}
CHK = json.load(open(ROOT/"results/pcb21/final_check.json")); PL = json.load(open(D/"prelabel.json"))["photos"]
(OUT/"labelme").mkdir(parents=True, exist_ok=True); gt = {}; n_clear = 0
for st, chk in CHK.items():
    b1 = D/"owner_final2/pcb21_batch1_class"/f"{st}.json"; src = b1 if b1.exists() else D/"owner_final2/pcb21_batch2"/f"{st}.json"
    if SRC: src = D/SRC/f"{st}.json"
    doc = json.load(open(src)); wrong = {a["ref"] for a in chk["A"] if (st, a["ref"]) not in KEEP}
    comps = [s for s in doc["shapes"] if s["label"] != "text"]
    for s in comps:
        if (s.get("description") or "").strip().upper() in wrong: s["description"] = ""; n_clear += 1
    texts = []
    if b1.exists():
        texts = [s for s in json.load(open(D/"owner_final2/pcb21_batch1"/f"{st}.json"))["shapes"] if s["label"] == "text"]
    doc["shapes"] = comps + texts
    json.dump(doc, open(OUT/"labelme"/f"{st}.json", "w"), indent=1)
    jpg = D/"labelme"/f"{st}.jpg"
    if jpg.exists(): shutil.copy2(jpg, OUT/"labelme"/f"{st}.jpg")
    sc = doc["imageWidth"] / PL[st]["W"]
    def nat(s):
        (x1, y1), (x2, y2) = s["points"][:2]
        return [round(min(x1, x2)/sc, 1), round(min(y1, y2)/sc, 1), round(max(x1, x2)/sc, 1), round(max(y1, y2)/sc, 1)]
    gt[st] = dict(W=PL[st]["W"], H=PL[st]["H"],
                  boxes=[nat(s) + [CLS.index(s["label"]) if s["label"] in CLS else -1, s["label"], (s.get("description") or "").strip()] for s in comps],
                  texts=[nat(s) + [(s.get("description") or "").strip()] for s in texts])
json.dump(gt, open(OUT/"gt9_21pcb.json", "w"))
print(len(gt), "photos,", sum(len(v["boxes"]) for v in gt.values()), "components,", sum(len(v["texts"]) for v in gt.values()), "text boxes;",
      n_clear, "wrong designators cleared")
