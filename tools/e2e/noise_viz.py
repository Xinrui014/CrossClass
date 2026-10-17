# Draw kicad_noise boxes (red print / orange via / magenta pad) and the side's label boxes (green) on its render; prints counts.
import json, sys, glob, os
from PIL import Image, ImageDraw
R = os.path.expanduser("~/projects/crossclass/linkdata"); out = sys.argv[1]; fs = sorted(glob.glob(f"{R}/kicad_noise/*.json"))
tot = {}
for f in fs:
    n = json.load(open(f))
    for k in n["kind"]: tot[k] = tot.get(k, 0)+1
print("sides", len(fs), "boxes", tot, "per side", round(sum(tot.values())/max(len(fs), 1), 1))
big = sorted(fs, key=lambda f: -len(json.load(open(f))["kind"]))[:4]
for i, f in enumerate(big):
    n = json.load(open(f)); r = json.load(open(f"{R}/kicad_render/{n['id']}.json")); im = Image.open(f"{R}/kicad_render/{n['id']}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
    for b in r["texts"]: d.rectangle(b, outline=(0, 255, 0), width=2)
    for b, k in zip(n["px"], n["kind"]): d.rectangle(b, outline={"print": (255, 0, 0), "via": (255, 160, 0), "pad": (255, 0, 255)}[k], width=2)
    im.thumbnail((1400, 1400)); im.save(f"{out}/noise_{i}.jpg", quality=85); print(n["id"], n["n_cand"], len(n["kind"]), len(r["texts"]))
