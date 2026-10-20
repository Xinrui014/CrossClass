#!/usr/bin/env python3
"""Pre-annotation of 21PCB photos for the human review (CPU). Inputs: data/21pcb/preds_det.json (pcb21_detect.py),
data/21pcb/ocr.json (pcb21_ocr.py), the parsed BoMs (tabular_rows.json, pdf_rows.json).
Link rule (deliberately simple and NOT our learned linker, so the labels do not favour our method):
  only OCR texts that match a BoM designator take part; cost(part, text) = edge gap / part size, + 1.0 when the BoM class of the
  designator disagrees with the detector's class; pairs with gap > 2 part sizes are not allowed; one-to-one by Hungarian.
  Each designator is used at most once per photo (best-scoring text kept).
Class: linked part -> class from the BoM row (bom_lib.classify -> 9 classes, else ignore -1); unlinked part -> detector class.
Parts: detector boxes with score >= 0.25, plus boxes with score >= 0.05 that get a designator link.
Output data/21pcb/prelabel.json = {stem: {"W", "H", "parts": [{"box", "cls", "src": "bom"|"det", "det_cls", "score", "ref", "text"}],
  "texts": [{"box", "text", "ref", "score"}], "bom_missing": [refs of the board seen on neither photo]}}"""
import json, os, re, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.dont_write_bytecode = True                     # tcassi is read-only: no __pycache__ there
sys.path.insert(0, str(Path.home()/"projects/tcassi/bom_analysis")); import bom_lib as B
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
C9 = {"resistor": 0, "capacitor": 1, "inductor": 2, "filter": 2, "connector": 3, "diode/LED": 4, "switch": 5, "transistor": 6,
      "IC": 7, "oscillator/crystal": 8}
NAMES = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]

bom = {}
for f in ("tabular_rows.json", "pdf_rows.json"):
    for card, v in json.load(open(D/f)).items():
        for r in v["rows"]:
            if not r.get("ref"): continue
            raw = B.classify(r["ref"], r.get("desc", ""), r.get("package_raw", ""))
            txt = " ".join(str(r.get(k, "")) for k in ("desc", "value", "mpn")).upper()
            # owner decisions 2026-10-18: EMI filters (FL*, NFM*) count as inductor; modules described as connectors (RJ-45
            # MagJack "M1") count as connector
            if (re.match(r"^FL", r["ref"].upper()) or re.match(r"^NFM", str(r.get("mpn", "") or r.get("value", "")).upper())
                    or "EMIFIL" in txt): raw = "filter"       # Card13/15 F1-F4: 3-terminal EMI filter capacitors (TDK CKD510)
            elif re.search(r"TEST ?POINT|TST ?PNT", txt): raw = "testpoint/mech"   # Card14: test points named S1-S8, T\CS, ...
            elif re.match(r"^\s*LED\b", str(r.get("desc", "")).upper()): raw = "diode/LED"   # Card10: LEDs named B1-B4 / M0-M4
            elif re.search(r"RESISTOR LINK|ZERO ?OHM|0 ?OHM JUMPER", txt): raw = "resistor"   # Card18 JP1/3/4, Card10 R36
            elif re.search(r"TERMINAL ?BLOCK|COMBICON", txt): raw = "connector"   # Card10 OUT
            elif re.search(r"SOLDER LINK.*RESISTOR|0 ?(OHM|Ω).*RESISTOR", txt): raw = "resistor"   # SLx: 0-ohm links
            elif (raw in ("other", "testpoint/mech") and re.search(r"\bCONN|RJ-?45|MAGJACK|JACK\b", txt)
                  and not re.match(r"^TP", r["ref"].upper()) and not re.search(r"TST|TEST", txt)): raw = "connector"
            dni = bool(r.get("dni")) or bool(re.search(r"NOT INSERTED|DO NOT INSERT|PLACE ?HOLDER|\bDNI\b|\bDNP\b", txt))
            bom.setdefault(card, {})[r["ref"].strip().upper()] = dict(raw=raw, cls=C9.get(raw, -1), dni=dni,
                                                                      value=r.get("value", ""), desc=r.get("desc", ""))
DET = json.load(open(D/"preds_det.json")); OCR = json.load(open(D/"ocr.json")); out = {}; seen = {}

def gap(p, t):
    dx = max(0, max(p[0], t[0]) - min(p[2], t[2])); dy = max(0, max(p[1], t[1]) - min(p[3], t[3])); return float(np.hypot(dx, dy))

for st in sorted(DET):
    if st not in OCR: continue
    card = st.split("_")[0]; bm = bom.get(card, {}); W, H = Image.open(D/"images"/f"{st}.jpg").size
    det = [d for d in DET[st] if d[4] >= 0.05]
    texts = [dict(box=o["box"], text=o["text"], ref=o["ref"], score=o["score"]) for o in OCR[st]]
    best = {}
    for j, t in enumerate(texts):
        if t["ref"] and (t["ref"] not in best or t["score"] > texts[best[t["ref"]]]["score"]): best[t["ref"]] = j
    tj = sorted(best.values())
    C = np.full((len(det), len(tj)), 1e6)
    for i, d in enumerate(det):
        sz = max(d[2]-d[0], d[3]-d[1])
        for k, j in enumerate(tj):
            g = gap(d, texts[j]["box"]) / sz
            if g <= 2: C[i, k] = g + (1.0 if bm[texts[j]["ref"]]["cls"] != int(d[5]) else 0.0) + (0.3 if d[4] < 0.25 else 0.0)
    link = {}
    if len(det) and len(tj):
        for i, k in zip(*linear_sum_assignment(C)):
            if C[i, k] < 1e6: link[i] = tj[k]
    parts = []
    for i, d in enumerate(det):
        if i not in link and d[4] < 0.25: continue
        j = link.get(i); ref = texts[j]["ref"] if j is not None else None
        cls = bm[ref]["cls"] if ref else int(d[5])
        parts.append(dict(box=[round(v, 1) for v in d[:4]], cls=cls, src="bom" if ref else "det", det_cls=int(d[5]),
                          score=round(d[4], 3), ref=ref, text=j))
    out[st] = dict(W=W, H=H, parts=parts, texts=texts)
    seen.setdefault(card, set()).update(t["ref"] for t in texts if t["ref"])
for st in out:
    card = st.split("_")[0]
    out[st]["bom_missing"] = sorted(r for r, v in bom.get(card, {}).items() if not v["dni"] and r not in seen[card])
json.dump(dict(classes=NAMES, bom={c: bom[c] for c in {s.split("_")[0] for s in out}}, photos=out), open(D/"prelabel.json", "w"))
for st, v in out.items():
    n = len(v["parts"]); nb = sum(p["src"] == "bom" for p in v["parts"])
    print(f"{st}: {n} parts ({nb} with a BoM designator), {len(v['texts'])} texts, board refs not seen on either photo: {len(v['bom_missing'])}")
