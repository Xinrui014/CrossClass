#!/usr/bin/env python3
"""Input for the blind 'LLM decides what each designator prefix means on each board' step on 21PCB (same as tools/llm_naming_prep.py on
WACV, inference only). Per photo and per prefix of the linked readings (hard links of v2d2_I_noise05_100k on the rule boxes, p >= 0.5;
reading = PARSeq argmax): number of parts, example designators, the detector's votes, median part size (px), the fixed prefix table's
class. No ground truth and no BoM are used. Writes OUT/summary.txt, OUT/pairs.json and crop sheets (part box blue, linked text red,
6 examples per pair) for the pairs where the table and the detector majority disagree, the prefix is unknown, or the votes split.
  pcb21_llm_prep.py OUT   (env ~/venvs/gdino_fewshot)"""
import json, os, sys, collections, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.path.expanduser("~/projects/crossclass")); LD = ROOT/"data/21pcb/e2e"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import PFX
TAB = dict(PFX); PRE = re.compile(r"^([A-Z]+)\d"); NM = ["Resistor", "Capacitor", "Inductor", "Connector", "Diode", "Switch", "Transistor", "IC", "Oscillator"]
P9 = json.load(open(LD/"preds_cls9.json")); B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}
BI = {b["id"]: b for b in json.load(open(LD/"e2e_boards_infil.json"))}; L = json.load(open(LD/"links_noise05_infil_raw.json")); Z = np.load(LD/"e2e_tf_all.npz")
CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
read = lambda st, j: "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
pairs, lines = [], []
for st in sorted(B):
    groups = collections.defaultdict(list)
    for i, (j, p) in L.get(st, {}).items():
        if j is None or p < 0.5: continue
        jo = BI[st]["texts"][j]["orig"]; k = B[st]["det_index"][int(i)]; r = read(st, jo); m = PRE.match(r)
        if m: groups[m.group(1)].append((k, jo, r))
    lines.append(f"\n== {st}")
    for pre, it in sorted(groups.items(), key=lambda x: -len(x[1])):
        D = np.array([P9[st][k][6:15] for k, _, _ in it], float); D = D / D.sum(1, keepdims=True); votes = collections.Counter(NM[int(v)] for v in D.argmax(1))
        wh = np.array([[P9[st][k][2]-P9[st][k][0], P9[st][k][3]-P9[st][k][1]] for k, _, _ in it]); tab = NM[TAB[pre]] if pre in TAB and TAB[pre] <= 8 else "-"
        top, topn = votes.most_common(1)[0]; flag = (tab == "-") or (top != tab) or (topn / len(it) < 0.7)
        ex = sorted({r for _, _, r in it})[:8]
        lines.append(f"  {pre:6s} n={len(it):3d} table={tab:10s} detector votes={dict(votes.most_common(3))} size(px)={int(np.median(wh[:,0]))}x{int(np.median(wh[:,1]))} "
                     f"examples={ex}{'  *' if flag else ''}")
        pairs.append(dict(board=st, prefix=pre, n=len(it), table=tab, votes=dict(votes), examples=ex, flag=flag, items=[[k, j] for k, j, _ in it]))
open(OUT/"summary.txt", "w").write("\n".join(lines)); json.dump(pairs, open(OUT/"pairs.json", "w"))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
fl = [q for q in pairs if q["flag"]]; CW, CHh = 240, 200; cache = {}
for s0 in range(0, len(fl), 4):
    sheet = Image.new("RGB", (6*CW, 4*(CHh+22)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for r_, q in enumerate(fl[s0:s0+4]):
        st = q["board"]
        if st not in cache: cache.clear(); cache[st] = Image.open(ROOT/"data/21pcb/images"/f"{st}.jpg").convert("RGB")
        im = cache[st]; y = r_*(CHh+22); d.text((4, y+2), f"[{s0+r_}] {st}  prefix {q['prefix']}  n={q['n']}  table={q['table']}  detector votes={q['votes']}", fill=(0, 0, 0), font=font)
        for c_, (k, j) in enumerate(q["items"][:6]):
            pb = P9[st][k][:4]; tb = B[st]["texts"][j]["box"]; x0, y0 = min(pb[0], tb[0]), min(pb[1], tb[1]); x1, y1 = max(pb[2], tb[2]), max(pb[3], tb[3])
            w, h = max(x1-x0, 20), max(y1-y0, 20); x0 -= 0.6*w; y0 -= 0.6*h; x1 += 0.6*w; y1 += 0.6*h
            cr = im.crop((int(x0), int(y0), int(x1), int(y1))); sc = min(CW/cr.width, CHh/cr.height); cr = cr.resize((max(1, int(cr.width*sc)), max(1, int(cr.height*sc))))
            cd = ImageDraw.Draw(cr); T = lambda b: [(b[0]-int(x0))*sc, (b[1]-int(y0))*sc, (b[2]-int(x0))*sc, (b[3]-int(y0))*sc]
            cd.rectangle(T(pb), outline=(30, 110, 255), width=2); cd.rectangle(T(tb), outline=(230, 20, 20), width=2); sheet.paste(cr, (c_*CW, y+20))
    sheet.save(OUT/f"sheet_{s0//4:02d}.jpg", quality=85)
print(len(pairs), "pairs,", len(fl), "flagged,", (len(fl)+3)//4, "sheets")
