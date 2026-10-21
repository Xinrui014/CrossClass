#!/usr/bin/env python3
"""Browser for ALL soft-cascade C errors on 21PCB (owner request 2026-10-19), grouped by the causes of tools/pcb21_c_analysis.py.
For every error: a crop (part box magenta, linked texts cyan with their link probability), GT class + designator, C's class, the
detector's class, the top linked texts with PARSeq reading and link probability, and C's top-3 probabilities.
Writes page/c_errors/index.html + img/<n>.jpg (git-ignored). Serve on the L40:
  cd ~/projects/crossclass/page/c_errors && python3 -m http.server 8765 --bind 127.0.0.1
and open http://localhost:8765 after `ssh -L 8765:localhost:8765 l40`.
  pcb21_c_errors_html.py   (env ~/venvs/gdino_fewshot, CPU)"""
import json, os, sys, html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(os.path.expanduser("~/projects/crossclass")); OUTD = ROOT/"page/c_errors"; (OUTD/"img").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"tools")); sys.argv = ["x"]
src = open(ROOT/"tools/pcb21_c_analysis.py").read().split("font = ImageFont")[0]
src = src.replace('rec = dict(st=st, gt=NM[g[4]], des=des, det=NM[pd], C=NM[pc], box=dets[k][:4], link=link, mass=round(mass, 2))',
                  'rec = dict(st=st, gt=NM[g[4]], des=des, det=NM[pd], C=NM[pc], box=dets[k][:4], link=link, mass=round(mass, 2), '
                  'texts=sorted(r["texts"], key=lambda x: -x[1])[:3] if r else [], none=round(r["none"], 3) if r else 1.0, P=P.tolist(), d=d.tolist())')
exec(src)
ORDER = ["right label, OCR misread", "right label, prefix not in table", "right label, table class differs", "wrong text linked",
         "label found, linker said none", "label missed by OCR", "no printed label"]
EXPL = {"right label, OCR misread": "The part's own label is linked (checked against the link GT), but PARSeq reads it wrong (rotated / tiny text): no known prefix, C falls back to the detector.",
        "right label, prefix not in table": "The own label is linked and read as a designator whose prefix the fixed table does not know (B, F, TVS, FL, M ...): C falls back to the detector.",
        "right label, table class differs": "The own label is linked and read, but the prefix table class differs from the GT class (T, JP ...).",
        "wrong text linked": "The strongest link goes to another box than the part's label (link GT).",
        "label found, linker said none": "An OCR box covers the part's label, but the link mass is below 0.5: C keeps the detector class.",
        "label missed by OCR": "The part has a label, but no OCR box covers it, so nothing can be linked: C keeps the detector class.",
        "no printed label": "The part has no label on the board (link GT none): C keeps the detector class, or links a wrong text."}
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); cache = {}
errs.sort(key=lambda e: (ORDER.index(e["cause"]), e["st"], e["gt"]))
for n, e in enumerate(errs):
    st = e["st"]
    if st not in cache: cache.clear(); cache[st] = Image.open(E/"21pcb/images"/f"{st}.jpg").convert("RGB")
    im = cache[st]; b = e["box"]; tb = [(B[st]["texts"][j]["box"], read(st, j), p) for j, p in e["texts"]]; bs = [b] + [t[0] for t in tb if t[2] >= 0.05]
    x0 = min(x[0] for x in bs); y0 = min(x[1] for x in bs); x1 = max(x[2] for x in bs); y1 = max(x[3] for x in bs)
    psz = max(b[2]-b[0], b[3]-b[1]); cx, cy = (x0+x1)/2, (y0+y1)/2
    # card image: the part, its linked labels and the neighbouring parts (at least 6 part sizes); click -> a 3x wider view
    for side, px, suf in ((max(max(x1-x0, y1-y0) * 2.2, 6 * psz) + 300, 520, ""), (max(max(x1-x0, y1-y0) * 2.2, 6 * psz) * 3 + 600, 1100, "_wide")):
        X0, Y0 = cx-side/2, cy-side/2; c = im.crop((int(X0), int(Y0), int(X0+side), int(Y0+side))); z = px/side; c = c.resize((px, px), Image.LANCZOS); d = ImageDraw.Draw(c)
        R = lambda q: [(q[0]-X0)*z, (q[1]-Y0)*z, (q[2]-X0)*z, (q[3]-Y0)*z]
        for q, txt, p in tb:
            if p >= 0.05: d.rectangle(R(q), outline=(0, 230, 255), width=2); d.text((R(q)[0], R(q)[3]+1), f"{p:.2f}", fill=(0, 230, 255), font=font)
        d.rectangle(R(b), outline=(255, 0, 255), width=3); c.save(OUTD/"img"/f"{n}{suf}.jpg", quality=86)
    e["img"] = f"img/{n}.jpg"; e["wide"] = f"img/{n}_wide.jpg"
PB = json.load(open(E/"21pcb/prelabel.json"))["bom"]
def suspect(e):   # the linked text names a real BoM part whose class equals C's answer: the GT designator / class may be wrong
    if e["cause"] != "wrong text linked" or not e["link"]: return False
    b = PB[e["st"].split("_")[0]].get(e["link"][1]); return bool(b and b["cls"] >= 0 and NM[b["cls"]] == e["C"])
top3 = lambda v: ", ".join(f"{NM[i]} {v[i]:.2f}" for i in np.argsort(-np.array(v))[:3])
sec = []
for c in ORDER:
    items = [e for e in errs if e["cause"] == c]
    cards = "".join(f'''<div class="card"><a href="{e['wide']}" target="_blank"><img src="{e['img']}" loading="lazy" alt="{html.escape(e['st'])}"></a>
<div class="t">{'<span class="sus">possible GT error</span><br>' if suspect(e) else ''}<b>GT {e['gt']}</b> &rarr; <span class="bad">C {e['C']}</span> <span class="dim">(detector {e['det']})</span></div>
<div class="m">{html.escape(e['st'])} &middot; GT designator <b>{html.escape(e['des'] or '-')}</b></div>
<div class="m">links: {html.escape("; ".join(f"'{read(e['st'], j)}' {p:.2f}" for j, p in e['texts']) or 'none')} &middot; none {e['none']:.2f}</div>
<div class="m">C: {top3(e['P'])}<br>detector: {top3(e['d'])}</div></div>''' for e in items)
    sec.append(f'<section id="s{ORDER.index(c)}"><h2>{html.escape(c)} <span class="dim">({len(items)})</span></h2><p class="dim">{html.escape(EXPL[c])}</p><div class="grid">{cards}</div></section>')
nav = " &middot; ".join(f'<a href="#s{i}">{html.escape(c)} ({sum(e["cause"] == c for e in errs)})</a>' for i, c in enumerate(ORDER))
page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>21PCB C Errors</title>
<style>body{{background:#14171a;color:#e3e8ec;font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}}h1{{font-size:20px;margin:0 0 6px}}h2{{font-size:17px;margin:28px 0 4px}}
a{{color:#7cc7ff}}.dim{{color:#93a0ab}}.bad{{color:#ff8a80;font-weight:600}}.sus{{background:#5a4a00;color:#ffd54a;padding:1px 6px;border-radius:4px;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:#1d2226;border:1px solid #2c343a;border-radius:8px;overflow:hidden}}.card img{{width:100%;display:block}}.t{{padding:6px 8px 0}}.m{{padding:2px 8px;font-size:12.5px;color:#c3ccd3}}.card .m:last-child{{padding-bottom:8px}}
nav{{position:sticky;top:0;background:#14171a;padding:6px 0;border-bottom:1px solid #2c343a}}</style></head><body>
<h1>21PCB: all {len(errs)} errors of the soft cascade C</h1><p class="dim">Causes checked against the owner's link GT (2026-10-21). Yellow tag 'possible GT error' = a wrong-text case whose linked text is a real BoM designator with the class C answered. Parts the detector found (IoU &ge; 0.5, score &ge; 0.25) whose top-1 class under C is wrong. Magenta = part box, cyan = linked texts with link probability. Click an image for a 3x wider view.</p>
<nav>{nav}</nav>{"".join(sec)}</body></html>'''
open(OUTD/"index.html", "w").write(page); print(len(errs), "errors ->", OUTD/"index.html")
