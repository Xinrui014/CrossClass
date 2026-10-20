#!/usr/bin/env python3
"""Export the 21PCB pre-labels as labelme files (owner wants an existing annotation tool, 2026-10-18), and read them back.
Convention (owner chose "option B", 2026-10-18; one rectangle per object):
  component : label = class (resistor capacitor inductor connector diode switch transistor IC oscillator, or ignore =
              test point / mech / fuse / other); description = its printed designator, e.g. R16 (empty = no printed label)
  text      : label = text; description = the printed designator, e.g. R16. OCR strings that match no BoM designator start
              with "?" (e.g. ?R1G): fix or delete the box.
  link      : a component and a text box with the same description belong together.
  Boards with design files (Card19): component boxes and designators come from the design check (pcb21_design_check.py);
  description "?extra" = a detector box with no design part at that place (a false box, or a part missing from the design).
Images are downscaled copies (long side <= MAXS px); read-back restores native photo pixels (scale = imageWidth / native width).
Open with:  labelme pcb21_labelme --labels pcb21_labelme/labels.txt --nodata --autosave
  export: pcb21_labelme.py export [Card6_,Card19_]   -> data/21pcb/labelme/<stem>.{jpg,json} + data/21pcb/labelme_<tag>.zip
  import: pcb21_labelme.py import DIR                -> DIR/labels.json {stem: {"parts", "texts", "conflicts"}} (native px)"""
import json, os, re, sys, zipfile
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; MAXS = 6000
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
LABELS = CLS + ["ignore", "text"]
DESLIKE = re.compile(r"^[A-Z]{1,4}[0-9]{1,4}[A-Z]?$")

def rect(label, b, s, desc=""):
    return dict(label=label, points=[[round(b[0]*s, 1), round(b[1]*s, 1)], [round(b[2]*s, 1), round(b[3]*s, 1)]], group_id=None,
                description=desc, shape_type="rectangle", flags={}, mask=None)

def export(only):
    global FIX
    FIX = json.load(open(D/"design_fixed.json")) if (D/"design_fixed.json").exists() else {}
    P = json.load(open(D/"prelabel.json")); out = D/"labelme"; out.mkdir(exist_ok=True); files = []
    TF = json.load(open(D/"text_fill.json")) if (D/"text_fill.json").exists() else {}   # designator texts found by pcb21_text_fill.py
    for st, v in P["photos"].items():
        if only and not any(st.startswith(o) for o in only): continue
        im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); s = min(1.0, MAXS/max(im.size))
        im = im.resize((round(im.size[0]*s), round(im.size[1]*s)), Image.LANCZOS); im.save(out/f"{st}.jpg", quality=85)
        shapes, used, tfix = [], set(), {}
        if st in FIX:                                   # board with design files: designators + boxes from the design check
            pre_text = {tuple(p["box"]): p["text"] for p in v["parts"]}
            for p in FIX[st]["parts"]:
                lab = CLS[p["cls"]] if p["cls"] >= 0 else "ignore"
                shapes.append(rect(lab, p["box"], s, p["ref"] if p["ref"] else "?extra"))
                j = pre_text.get(tuple(p["box"]))
                if j is not None and p["ref"]:
                    used.add(j); t = v["texts"][j]["ref"]
                    if t and t != p["ref"] and p["ref"].startswith(t): tfix[j] = p["ref"]     # OCR cut the designator short
        else:
            for p in v["parts"]:
                shapes.append(rect(CLS[p["cls"]] if p["cls"] >= 0 else "ignore", p["box"], s, p["ref"] or ""))
                if p["text"] is not None: used.add(p["text"])
        for j, t in enumerate(v["texts"]):
            if j in tfix: shapes.append(rect("text", t["box"], s, tfix[j]))
            elif t["ref"]: shapes.append(rect("text", t["box"], s, t["ref"]))
            elif j in used or DESLIKE.match(re.sub(r"[^A-Z0-9]", "", t["text"].upper())):
                shapes.append(rect("text", t["box"], s, "?" + t["text"].strip()))
        for t in TF.get(st, []): shapes.append(rect("text", t["box"], s, t["ref"]))
        doc = dict(version="5.4.1", flags={}, shapes=shapes, imagePath=f"{st}.jpg", imageData=None,
                   imageHeight=im.size[1], imageWidth=im.size[0])
        json.dump(doc, open(out/f"{st}.json", "w"), indent=1); files += [out/f"{st}.jpg", out/f"{st}.json"]
        print(st, len(shapes), "shapes, scale", round(s, 3), flush=True)
    (out/"README.txt").write_text(__doc__.split("  export:")[0]); (out/"labels.txt").write_text("\n".join(LABELS) + "\n")
    tag = ("_".join(o.strip("_") for o in only) if len(only) <= 3 else f"subset{len(only)}") or "all"; z = D/f"labelme_{tag}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
        for p in files + [out/"README.txt", out/"labels.txt"]: f.write(p, f"pcb21_labelme/{p.name}")
    print("wrote", z, os.path.getsize(z))

def read_back(d):
    P = json.load(open(D/"prelabel.json")); d = Path(d); res = {}
    for jf in sorted(d.glob("*.json")):
        if jf.name == "labels.json": continue
        doc = json.load(open(jf)); s = doc["imageWidth"] / Image.open(D/"images"/f"{jf.stem}.jpg").size[0]
        bom = P["bom"].get(jf.stem.split("_")[0], {}); parts, texts, conf = [], [], []
        for sh in doc["shapes"]:
            (x1, y1), (x2, y2) = sh["points"][:2]; b = [min(x1, x2)/s, min(y1, y2)/s, max(x1, x2)/s, max(y1, y2)/s]
            lab, des = sh["label"].strip(), (sh.get("description") or "").strip().upper()
            if lab == "text": texts.append(dict(box=b, text=des.lstrip("?"), unsure=des.startswith("?")))
            else:
                cls = CLS.index(lab) if lab in CLS else -1; unsure = des.startswith("?"); des = "" if unsure else des
                parts.append(dict(box=b, cls=cls, label=lab, ref=des or None, unsure=unsure))
                if des and des in bom and bom[des]["cls"] != cls: conf.append(dict(ref=des, label=lab, bom=bom[des]["raw"]))
                if des and des not in bom: conf.append(dict(ref=des, label=lab, bom="not in BoM"))
        res[jf.stem] = dict(parts=parts, texts=texts, conflicts=conf)
        print(jf.stem, len(parts), "parts", len(texts), "texts", len(conf), "class/BoM conflicts")
    json.dump(res, open(d/"labels.json", "w")); print("->", d/"labels.json")

if __name__ == "__main__":
    if sys.argv[1] == "export": export([s for s in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if s])
    else: read_back(sys.argv[2])
