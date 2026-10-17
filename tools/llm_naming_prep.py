#!/usr/bin/env python3
"""Input for 'an LLM decides what each designator prefix means on each board' (2026-10-17, owner: the text -> class step is world
knowledge). Per WACV photo and per prefix of the linked readings (links RUN, p >= 0.5; reading = Claude where read, else PARSeq):
number of parts, example designators, the detector's votes (argmax class counts) and mean class distribution, median part size, and the
fixed prefix table's class. No ground truth is used. Writes OUT/summary.txt and OUT/pairs.json; crop sheets (part box blue, linked text
red, 6 examples per pair) for the pairs where the table and the detector majority disagree, the prefix is unknown, or the votes split.
Usage: llm_naming_prep.py E2EDIR RUN TODO READS OUT [SELECT.json = indices into pairs.json to draw instead of the flagged pairs]"""
import json, os, sys, collections, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(os.path.expanduser("~/projects/crossclass")); LD = Path(sys.argv[1]); RUN = sys.argv[2]; OUT = Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)
CL = {tuple(t): ("" if r == "-" else r) for t, r in zip(json.load(open(sys.argv[3])), json.load(open(sys.argv[4])))}
sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import PFX, RAW
TAB = dict(PFX); PRE = re.compile(r"^([A-Z]+)\d"); NM = ["Resistor", "Capacitor", "Inductor", "Connector", "Diode", "Switch", "Transistor", "IC", "Oscillator"]
P9 = json.load(open(ROOT/"data/preds_wacv_1280_cls9.json")); B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}
L = json.load(open(LD/f"e2e_tf_links_{RUN}_all.json")); RD = json.load(open(LD/"e2e_read_all.json"))
photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
pairs, lines = [], []
for st in sorted(B):
    groups = collections.defaultdict(list)
    for i, (j, p) in L.get(st, {}).items():
        if j is None or p < 0.5: continue
        k = B[st]["det_index"][int(i)]; r = CL.get((st, j), RD[st][j]) or ""; m = PRE.match(r)
        if m and P9[st][k][6] is not None: groups[m.group(1)].append((k, j, r))
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
try: font = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError: font = ImageFont.load_default()
fl = [pairs[i] for i in json.load(open(sys.argv[6]))] if len(sys.argv) > 6 else [q for q in pairs if q["flag"]]; CW, CH = 240, 200; cache = {}
for s0 in range(0, len(fl), 4):                          # 4 pairs per sheet, one row of 6 crops each
    sheet = Image.new("RGB", (6*CW, 4*(CH+22)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for r_, q in enumerate(fl[s0:s0+4]):
        st = q["board"]
        if st not in cache: cache.clear(); cache[st] = Image.open(photo(st)).convert("RGB")
        im = cache[st]; y = r_*(CH+22); d.text((4, y+2), f"[{s0+r_}] {st}  prefix {q['prefix']}  n={q['n']}  table={q['table']}  detector votes={q['votes']}", fill=(0, 0, 0), font=font)
        for c_, (k, j) in enumerate(q["items"][:6]):
            pb = P9[st][k][:4]; tb = B[st]["texts"][j]["box"]; x0, y0 = min(pb[0], tb[0]), min(pb[1], tb[1]); x1, y1 = max(pb[2], tb[2]), max(pb[3], tb[3])
            w, h = max(x1-x0, 20), max(y1-y0, 20); x0 -= 0.6*w; y0 -= 0.6*h; x1 += 0.6*w; y1 += 0.6*h
            cr = im.crop((int(x0), int(y0), int(x1), int(y1))); sc = min(CW/cr.width, CH/cr.height); cr = cr.resize((max(1, int(cr.width*sc)), max(1, int(cr.height*sc))))
            cd = ImageDraw.Draw(cr); T = lambda b: [(b[0]-int(x0))*sc, (b[1]-int(y0))*sc, (b[2]-int(x0))*sc, (b[3]-int(y0))*sc]
            cd.rectangle(T(pb), outline=(30, 110, 255), width=2); cd.rectangle(T(tb), outline=(230, 20, 20), width=2); sheet.paste(cr, (c_*CW, y+20))
    sheet.save(OUT/f"sheet_{s0//4:02d}.jpg", quality=85)
print(len(pairs), "pairs,", len(fl), "flagged,", (len(fl)+3)//4, "sheets")
