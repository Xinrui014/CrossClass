#!/usr/bin/env python3
"""Data for the 21PCB labelling page (page/label21/): downscaled photos img/<stem>.jpg (long side <= MAXS px) and pre21.json
(from data/21pcb/prelabel.json; boxes stay in native photo pixels, the page scales them by img width / W).
  pcb21_page_data.py [Card6_,Card19_]   (env ~/venvs/gdino_fewshot)"""
import json, os, sys
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; OUT = ROOT/"page/label21"
MAXS = 6000
only = [s for s in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if s]
P = json.load(open(D/"prelabel.json")); (OUT/"img").mkdir(parents=True, exist_ok=True)
photos = {st: v for st, v in P["photos"].items() if not only or any(st.startswith(o) for o in only)}
for st, v in photos.items():
    im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); s = min(1.0, MAXS/max(im.size))
    im.resize((round(im.size[0]*s), round(im.size[1]*s)), Image.LANCZOS).save(OUT/"img"/f"{st}.jpg", quality=82)
    v["parts"] = [dict(id=f"p{i}", box=p["box"], cls=p["cls"], text=(f"t{p['text']}" if p["text"] is not None else None),
                       src=p["src"], det_cls=p["det_cls"], score=p["score"]) for i, p in enumerate(v["parts"])]
    v["texts"] = [dict(id=f"t{j}", box=t["box"], str=t["text"], ocr=t["text"]) for j, t in enumerate(v["texts"])]
    print(st, im.size, round(s, 3), len(v["parts"]), len(v["texts"]), flush=True)
cards = {st.split("_")[0] for st in photos}
bom = {c: {r: dict(cls=b["cls"], raw=b["raw"], value=b["value"], desc=b["desc"], dni=b["dni"]) for r, b in P["bom"][c].items()} for c in cards}
json.dump(dict(classes=P["classes"], photos=photos, bom=bom), open(OUT/"pre21.json", "w"), separators=(",", ":"))
print("wrote", OUT/"pre21.json", os.path.getsize(OUT/"pre21.json"))
