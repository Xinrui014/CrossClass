#!/usr/bin/env python3
"""Turn a vector assembly drawing (designators printed inside the part outlines, e.g. HuMANDATA "PCB Drawing" PDFs with a top
and a bottom view side by side) into a centroid file that tools/design_check.py reads.
Words between the "View from" headers and the title block that look like designators become parts; the left half of the page is
the top view, the right half the bottom view (drawn as seen from below -> x is mirrored back to top-view coordinates, and
design_check.py --side bottom mirrors it again). Output: Allegro-style "REFDES,…,SYM_X,SYM_Y,SYM_ROTATE,SYM_MIRROR" CSV, y up.
Usage: pdf_to_centroid.py designfiles/HUMANDATA/XCM307-PCB-B-B.pdf [--page 0] > designfiles/HUMANDATA/XCM307-PCB-B-B.cen
--altium: an Altium layout PDF keeps invisible text objects "CO<designator>" per component and "PA<designator>0<pin>" per pad
          (Avnet MicroZed layouts). A part's position = mean of its pads (the CO object sits on the footprint origin, not the centre).
          The objects carry no board side, so every part is written for --side; --exclude <design_check json of the other photo>
          drops the parts that photo already confirmed ("agree"), i.e. parts on the other side.
       pdf_to_centroid.py designfiles/MicroZed/MicroZed_RevB_Layout_130617_0.pdf --altium --side top \
           [--exclude results/linking/design_check/MicroZed_Bottom.json] > designfiles/MicroZed/RevB_top.cen
(python: an env with PyMuPDF, e.g. ~/miniconda3/envs/ocr_ab)"""
import argparse, re
import fitz
ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--page", type=int, default=0)
ap.add_argument("--altium", action="store_true"); ap.add_argument("--side", default="top"); ap.add_argument("--exclude", default="")
a = ap.parse_args()
doc = fitz.open(a.pdf)
if a.altium:
    import json, collections
    W = doc[a.page].get_text("words")
    refs = {m.group(1) for x in W for m in [re.fullmatch(r"CO([A-Z]+\d+)", x[4])] if m}
    skip = {o["design"] for o in json.load(open(a.exclude))["parts"] if o["status"] == "agree"} if a.exclude else set()
    pads = collections.defaultdict(list)
    for x0, y0, x1, y1, t, *_ in W:                      # "PA" + ref + "0" + pin, pin never starts with 0 -> unique split
        if not t.startswith("PA"): continue
        rest = t[2:]; hit = [r for r in refs if rest.startswith(r + "0") and len(rest) > len(r) + 1 and rest[len(r) + 1] != "0"]
        if hit: pads[max(hit, key=len)].append(((x0 + x1) / 2, (y0 + y1) / 2))
    print("REFDES,COMP_DEVICE_TYPE,COMP_VALUE,COMP_TOL,SYM_NAME,SYM_X,SYM_Y,SYM_ROTATE,SYM_MIRROR")
    for r, v in sorted(pads.items()):
        if r in skip: continue
        print(f"{r},,,,,{sum(q[0] for q in v) / len(v):.2f},{-sum(q[1] for q in v) / len(v):.2f},0,{'YES' if a.side == 'bottom' else 'NO'}")
    raise SystemExit
p = doc[a.page]; W = p.get_text("words")
top = min(w[3] for w in W if w[4] == "View")                                   # just below the view headers
bot = min((w[1] for w in W if w[4] == "PCB" and w[1] > top), default=p.rect.height)   # "PCB Version" / title block
mid = p.rect.width / 2
print("REFDES,COMP_DEVICE_TYPE,COMP_VALUE,COMP_TOL,SYM_NAME,SYM_X,SYM_Y,SYM_ROTATE,SYM_MIRROR")
seen = set()
for x0, y0, x1, y1, t, *_ in W:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if not (top < cy < bot) or not re.fullmatch(r"[A-Z]{1,4}\d{0,3}", t) or len(t) < 2: continue
    bottom = cx > mid
    if (t, bottom) in seen: continue                                            # keep the first if a name repeats on a side
    seen.add((t, bottom))
    print(f"{t},,,,,{(-(cx - mid) if bottom else cx):.2f},{-cy:.2f},0,{'YES' if bottom else 'NO'}")
