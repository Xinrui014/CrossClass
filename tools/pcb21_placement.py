#!/usr/bin/env python3
"""Extract component placement of the 21PCB boards from the vendor design files only (no model, no detector, no OCR, no photo).
One common JSON per board: designfiles/21pcb/placement/<CardN>.json =
  {card, source: [files], units: "mil"|"mm", frame: text, parts: [{ref, x, y, side, rot, box, box_from, footprint}], checks: {...}}
Frame: the design's own coordinates, y up, seen from the TOP side (bottom parts are NOT mirrored). x, y = placement point of the
file (Allegro symbol origin / body centre, PADS part origin, Altium component origin); for IPC-only boards x, y = pad extent centre.
box = [x1, y1, x2, y2] in the same frame: pad extent (+ half pad size, pad rotation applied) from IPC-D-356 or the layout's own pads;
Altium falls back to the 3D body outline (box_from "body").
Readers:
  Allegro "Placed Component Report" (.placement: REFDES,...,SYM_X,SYM_Y,SYM_ROTATE,SYM_MIRROR; units from the IPC file of the same board)
  Allegro Place_Body_Center.txt (Card19; refdes x y rot [m] footprint, mil, "m" = mirrored = bottom)
  PADS PowerPCB ASCII (.asc, *PART* section; BASIC units = 1/38100 mil; MIRROR "M" = bottom)
  PADS XY export (.xlsx: PartType RefDes PartDecal Pins Layer Orient. X Y SMD Glued; mil)
  Altium ASCII PcbDoc (|RECORD=Component| X Y ROTATION LAYER; designator from the Text record with DESIGNATOR=True; pads COMPONENT=index)
  IPC-D-356 (317/327 records: refdes cols 21-26, access code A00 = through, A01 = top, A<n layers> = bottom; UNITS CUST 0/2 = 0.1 mil,
             CUST 1 = 0.001 mm)
Checks per board: parts per side, placement refs in the parsed BoM (data/21pcb/tabular_rows.json, pdf_rows.json), populated BoM refs
missing from the placement, pad-centre vs placement-point offset (frame sanity).
  pcb21_placement.py   (any python3; pure standard library)"""
import json, os, re, math, zipfile, collections, statistics
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); DF = ROOT/"designfiles/21pcb"; U = DF/"unz"
OUT = DF/"placement"; OUT.mkdir(parents=True, exist_ok=True)
C19 = DF/"Card19_TMDSIDK437X/tidrc77a/tidrc77a/AssemblyPackage/Revision 1p4A/AM437X_INDUSTRIAL_EVM_3K0008_PCB_REV1_4A_ASSEMBLY_PACKAGE"
MIL_PER_MM = 1/0.0254

# ---------------- IPC-D-356 ----------------
def load_ipc(f, to="mil"):
    """pads per refdes: list of (access, x, y, w, h) with the pad rotation already applied to w, h (axis-aligned extent)."""
    pads = collections.defaultdict(list); unit = None; rev = None; nlay = None
    for l in open(f, encoding="latin-1"):
        if l.startswith("P  UNITS") or l.startswith("P UNITS"):
            unit = int(l.split()[-1])
        elif re.match(r"^P\s+REV\b", l): rev = l.split()[2] if len(l.split()) > 2 else None
        elif "Number of etch Layers" in l: nlay = int(l.split(":")[1])
        if l[:3] not in ("317", "327"): continue
        r = l[20:26].strip()
        m = re.search(r"A(\d\d)X([+-]?\d+)Y([+-]?\d+)X(\d+)(?:Y(\d*))?\s*(?:R(\d+))?", l)
        if not r or r == "VIA" or not m: continue
        w = int(m.group(4)); h = int(m.group(5)) if m.group(5) else 0
        if h == 0: h = w                                                    # Y omitted or 0 = round pad
        a = math.radians(int(m.group(6))) if m.group(6) else 0.0
        W = abs(w*math.cos(a)) + abs(h*math.sin(a)); H = abs(w*math.sin(a)) + abs(h*math.cos(a))
        pads[r.upper()].append((int(m.group(1)), int(m.group(2)), int(m.group(3)), W, H))
    # scale: CUST 0 / 2 = 0.0001 inch, CUST 1 = 0.001 mm
    s_mm = 0.001 if unit == 1 else 0.00254; s = s_mm if to == "mm" else s_mm*MIL_PER_MM
    pads = {r: [(a, x*s, y*s, w*s, h*s) for a, x, y, w, h in v] for r, v in pads.items()}
    acc = [p[0] for v in pads.values() for p in v if p[0] > 0]
    bot = nlay if nlay else (max(acc) if acc and max(acc) > 1 else None)               # bottom access code = number of etch layers
    return pads, dict(unit=unit, rev=rev, layers=nlay, bottom_code=bot)

def pad_box(v):
    xs = [q[1]-q[3]/2 for q in v] + [q[1]+q[3]/2 for q in v]; ys = [q[2]-q[4]/2 for q in v] + [q[2]+q[4]/2 for q in v]
    return [min(xs), min(ys), max(xs), max(ys)]

def r3(b): return None if b is None else [round(float(x), 4) for x in b]

# ---------------- readers ----------------
def read_allegro_placement(f):
    import csv
    out = []; lines = open(f, encoding="latin-1").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("REFDES,"))
    for row in csv.DictReader(lines[i:]):
        out.append(dict(ref=row["REFDES"].strip().upper(), x=float(row["SYM_X"]), y=float(row["SYM_Y"]),
                        side="bottom" if row["SYM_MIRROR"].strip().upper() == "YES" else "top",
                        rot=float(row["SYM_ROTATE"]), footprint=row["SYM_NAME"].strip()))
    return out

def read_place_body_center(f):
    out = []
    for l in open(f, encoding="latin-1"):
        m = re.match(r"^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(m\s+)?(\S+)", l)
        if m: out.append(dict(ref=m.group(1).upper(), x=float(m.group(2)), y=float(m.group(3)), side="bottom" if m.group(5) else "top",
                              rot=float(m.group(4)), footprint=m.group(6)))
    return out

def read_pads_ascii(f):
    """*PART* section of a PADS PowerPCB ASCII file. Returns parts in mil (BASIC units /38100) and the decal per part type."""
    L = open(f, encoding="latin-1").read().splitlines()
    head = L[0]; basic = "BASIC" in head.upper()
    units = next((int(l.split()[1]) for l in L[:20] if l.startswith("UNITS")), 0)
    if not basic and units != 0: raise ValueError(f"PADS units {units} not handled")
    sec = None; ptypes = {}; out = []; skip = 0
    for l in L:
        if l.startswith("*") and not l.startswith("*REMARK*"):
            sec = l.split()[0]; continue
        if not l.strip() or l.startswith("*REMARK*"): continue
        t = l.split()
        # PARTTYPE header: NAME DECALNM(alt1:alt2:..) TYPE GATES SIGPINS UNUSEDPINNMS FLAGS ECO
        if sec == "*PARTTYPE*" and len(t) >= 7 and all(re.match(r"^-?\d+$", x) for x in t[3:7]):
            ptypes.setdefault(t[0], t[1])
        if sec != "*PART*": continue
        if skip: skip -= 1; continue
        if len(t) >= 12 and re.match(r"^-?\d+$", t[2]) and re.match(r"^-?\d+$", t[3]) and t[6] in ("N", "M"):
            pt = t[1]; dec = None
            if "@" in pt: pt, dec = pt.split("@", 1)
            alt = int(t[7]) if re.match(r"^-?\d+$", t[7]) else 0
            if dec is None and pt in ptypes:
                ds = ptypes[pt].split(":"); dec = ds[alt] if 0 <= alt < len(ds) else ds[0]
            sc = 1/38100 if basic else 1.0
            out.append(dict(ref=t[0].upper(), x=int(t[2])*sc, y=int(t[3])*sc, side="bottom" if t[6] == "M" else "top",
                            rot=float(t[4]), footprint=dec or pt))
            skip = 3*int(t[-1])                                              # each label = 3 lines
    return out

def read_xlsx_rows(f):
    z = zipfile.ZipFile(f); ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ss = [("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"]))) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter("{%s}row" % ns["m"]):
        vals = {}
        for c in r.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group(0); v = c.find("m:v", ns)
            if v is None: continue
            vals[col] = ss[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(vals)
    return rows

def read_pads_xy(f):
    rows = read_xlsx_rows(f); hdr = {v: k for k, v in rows[0].items()}; out = []
    for r in rows[1:]:
        if not r.get(hdr["RefDes"]): continue
        out.append(dict(ref=r[hdr["RefDes"]].strip().upper(), x=float(r[hdr["X"]]), y=float(r[hdr["Y"]]),
                        side="bottom" if r[hdr["Layer"]].strip().lower().startswith("bot") else "top",
                        rot=float(r[hdr["Orient."]]), footprint=r.get(hdr["PartDecal"])))
    return out

def _mil(v):
    v = v.strip()
    if v.endswith("mil"): return float(v[:-3])
    if v.endswith("mm"): return float(v[:-2])*MIL_PER_MM
    return float(v)

def read_altium_ascii(f):
    comps = []; texts = {}; pads = collections.defaultdict(list); bodies = collections.defaultdict(list)
    for l in open(f, encoding="latin-1"):
        if not l.startswith("|RECORD="): continue
        d = {}
        for kv in l.rstrip("\r\n").split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1); d.setdefault(k, v)
        rec = d.get("RECORD")
        if rec == "Component": comps.append(d)
        elif rec == "Text" and d.get("DESIGNATOR", "").upper() == "TRUE" and "COMPONENT" in d: texts[int(d["COMPONENT"])] = d.get("TEXT", "")
        elif rec == "Pad" and int(d.get("COMPONENT", -1)) >= 0:
            a = math.radians(float(d.get("ROTATION", "0"))); w = _mil(d["XSIZE"]); h = _mil(d["YSIZE"])
            W = abs(w*math.cos(a)) + abs(h*math.sin(a)); H = abs(w*math.sin(a)) + abs(h*math.cos(a))
            pads[int(d["COMPONENT"])].append((d.get("LAYER"), _mil(d["X"]), _mil(d["Y"]), W, H))
        elif rec == "ComponentBody" and int(d.get("COMPONENT", -1)) >= 0:
            n = int(d.get("MAINCONTOURVERTEXCOUNT", 0))
            bodies[int(d["COMPONENT"])] += [(_mil(d[f"VX{i}"]), _mil(d[f"VY{i}"])) for i in range(n) if f"VX{i}" in d]
    out = []
    for i, c in enumerate(comps):
        ref = (texts.get(i) or c.get("SOURCEDESIGNATOR") or "").strip().upper()
        box, frm = None, None
        if pads.get(i): box, frm = pad_box(pads[i]), "pads"
        elif bodies.get(i):
            xs, ys = zip(*bodies[i]); box, frm = [min(xs), min(ys), max(xs), max(ys)], "body"
        out.append(dict(ref=ref, x=_mil(c["X"]), y=_mil(c["Y"]), side="bottom" if c.get("LAYER") == "BOTTOM" else "top",
                        rot=float(c.get("ROTATION", "0")) % 360, footprint=c.get("PATTERN"), box=box, box_from=frm,
                        src_designator=c.get("SOURCEDESIGNATOR"), pad_layers={q[0] for q in pads.get(i, [])}))
    return out

def parts_from_ipc_only(pads, meta):
    out = []
    for r, v in sorted(pads.items()):
        b = pad_box(v); acc = {p[0] for p in v}
        if meta["bottom_code"] in acc and 1 not in acc: side, doubt = "bottom", None
        elif 1 in acc: side, doubt = "top", ("pads on both outer layers" if meta["bottom_code"] in acc else None)
        else: side, doubt = "top", "through-hole only: side assumed top"
        p = dict(ref=r, x=(b[0]+b[2])/2, y=(b[1]+b[3])/2, side=side, rot=None, footprint=None, box=b, box_from="pads")
        if doubt: p["doubt"] = doubt
        out.append(p)
    return out

def fix_truncated(pads, known):
    """Allegro IPC-D-356 keeps only 6 characters of a refdes (cols 21-26). Map a 6-character IPC ref to the one longer known ref
    (placement or BoM) that starts with it; returns the list of repairs."""
    fixed = []
    for r in [r for r in pads if r not in known and len(r) == 6]:
        c = [k for k in known if len(k) > 6 and k.startswith(r)]
        if len(c) == 1 and c[0] not in pads: pads[c[0]] = pads.pop(r); fixed.append((r, c[0]))
    return fixed

def attach_ipc(parts, pads):
    """box = pad extent from IPC; returns median offset pad centre - placement point (frame sanity)."""
    off = []
    for p in parts:
        v = pads.get(p["ref"])
        if v:
            p["box"], p["box_from"] = pad_box(v), "pads"
            off.append(((p["box"][0]+p["box"][2])/2 - p["x"], (p["box"][1]+p["box"][3])/2 - p["y"]))
        else: p.setdefault("box", None); p.setdefault("box_from", None)
    return off

# ---------------- board table ----------------
def F(*a): return U.joinpath(*a)
CN540 = F("cn0540-DesignSupport/CN0540-DesignSupport/CN0540 CAD Files")
CN506 = F("cn0506-DesignSupport/cn0506-designsupport")
CN216 = F("cn0216-sdp-designsupport/CN0216-Design Support"); CN272 = F("CN0272-DesignSupport/CN0272-DesignSupport")
BOARDS = {
    "Card6":  dict(kind="allegro_placement", place=CN540/"assy/01_059358b.placement", ipc=CN540/"fab/08-059358b-ipc356.ipc", rev="B (08-059358b)", bom_rev="CN0540 BOM.csv (no rev in name; identical file)"),
    "Card8":  dict(kind="allegro_placement", place=CN506/"CN0506 assembly files/01_054610b.placement", ipc=CN506/"CN0506 fab files/08_054610b-ipc356.ipc", rev="B (08_054610b)", bom_rev="CN0506 BOM.csv (identical file)"),
    "Card17": dict(kind="pads_xy", place=F("CN0321-DesignSupport/CN0321-DesignSupport/EVAL-CN0321-SDPZ_XY.xlsx"), ipc=None, rev="0 (Layout_Rev0)", bom_rev="BOM_Rev0 (identical file)"),
    "Card9":  dict(kind="pads_ascii", place=CN216/"EVAL-CN0216-SDPZ-PADSLayout-RevA.asc", ipc=CN216/"EVAL-CN0216-SDPZ-GBR-RevA/CN0216-NET-A.IPC", rev="A", bom_rev="BOM-RevA (same content, only saved-by name differs)"),
    "Card16": dict(kind="pads_ascii", place=CN272/"EVAL-CN0272-SDPZ-PADSLayout-RevC.asc", ipc=CN272/"EVAL-CN0272-SDPZ-GBR-RevC/CN0272-NET-C.IPC", rev="C", bom_rev="BOM-RevC (same content, only saved-by name differs)"),
    "Card7":  dict(kind="ipc_only", place=None, ipc=F("cn0395-DesignSupport/CN0395-DesignSupport/EVAL-CN0395-ARDZ-GRB-RevB/08_043603b-ipc356.ipc"), rev="B (08_043603b; IPC header says REV A)", bom_rev="BOM-RevB (identical file)"),
    "Card5":  dict(kind="altium_ascii", place=DF/"Card5_TPS65982EVM/slvc618/Altium Project/HVL117D_ASCII.PcbDoc", ipc=None, rev="HVL117D (rev D)", bom_rev="TPS65982EVM(inc BoM).pdf"),
    "Card19": dict(kind="place_body_center", place=C19/"Place_Body_Center.txt", ipc=C19/"AM437X_INDUSTRIAL_EVM_3K0008_PCB_REV1_4A.ipc", rev="1.4A", bom_rev="REV1_4A (identical file)"),
}

def bom_rows(card):
    for fn in ("tabular_rows.json", "pdf_rows.json"):
        d = json.load(open(ROOT/"data/21pcb"/fn))
        if card in d: return d[card]["file"], d[card]["rows"]
    return None, []

summary = {}
for card, b in BOARDS.items():
    k = b["kind"]; src = [str(p.relative_to(DF)) for p in (b["place"], b["ipc"]) if p]; notes = []
    units = "mil"; pads = None; meta = {}
    if b["ipc"]:
        # Allegro .placement files carry no unit; take it from the IPC file of the same board (CUST 1 = mm)
        unit_code = load_ipc(b["ipc"])[1]["unit"]
        units = "mm" if (k == "allegro_placement" and unit_code == 1) else "mil"
        pads, meta = load_ipc(b["ipc"], to=units)
    bf, rows = bom_rows(card)
    if pads is not None:
        known = {r["ref"].upper() for r in rows} if k == "ipc_only" else None
        if k != "ipc_only":
            known = {p["ref"] for p in (read_allegro_placement(b["place"]) if k == "allegro_placement" else read_place_body_center(b["place"])
                                         if k == "place_body_center" else read_pads_ascii(b["place"]))}
        fx = fix_truncated(pads, known)
        if fx: notes.append(f"IPC refdes truncated to 6 characters, repaired by unique prefix ({'BoM' if k == 'ipc_only' else 'placement'}): {fx}")
    if k == "allegro_placement": parts = read_allegro_placement(b["place"])
    elif k == "place_body_center": parts = read_place_body_center(b["place"])
    elif k == "pads_ascii": parts = read_pads_ascii(b["place"])
    elif k == "pads_xy": parts = read_pads_xy(b["place"])
    elif k == "altium_ascii": parts = read_altium_ascii(b["place"])
    elif k == "ipc_only": parts = parts_from_ipc_only(pads, meta)
    off = []
    if pads is not None and k != "ipc_only":
        off = attach_ipc(parts, pads)
        extra = sorted(set(pads) - {p["ref"] for p in parts})
        if extra: notes.append(f"IPC refs with no placement: {extra[:20]}{' ...' if len(extra) > 20 else ''} ({len(extra)})")
    for p in parts:
        p.setdefault("box", None); p.setdefault("box_from", None)
    if k == "altium_ascii":
        off = [((p["box"][0]+p["box"][2])/2 - p["x"], (p["box"][1]+p["box"][3])/2 - p["y"]) for p in parts if p["box_from"] == "pads"]
    # frame sanity: offset of pad-extent centre from placement point
    if off:
        dx = statistics.median(o[0] for o in off); dy = statistics.median(o[1] for o in off)
        d = sorted(math.hypot(*o) for o in off); tol = 0.5 if units == "mm" else 20
        frame = dict(median_dx=round(dx, 3), median_dy=round(dy, 3), median_dist=round(d[len(d)//2], 3),
                     frac_within=round(sum(x <= tol for x in d)/len(d), 3), tol=tol)
    else: frame = None
    # side check: placement side vs the outer layer of its SMD pads (IPC access code or Altium pad layer)
    conf = []
    for p in parts:
        if k == "altium_ascii": L = p.get("pad_layers", set()); t, bt = "TOP" in L, "BOTTOM" in L
        elif pads is not None and k != "ipc_only" and p["ref"] in pads:
            A = {q[0] for q in pads[p["ref"]]}; t, bt = 1 in A, meta["bottom_code"] in A
        else: continue
        if (p["side"] == "top" and bt and not t) or (p["side"] == "bottom" and t and not bt): conf.append(p["ref"])
    if conf: notes.append(f"placement side disagrees with SMD pad layer: {conf}")
    # duplicate refs
    cnt = collections.Counter(p["ref"] for p in parts); dup = [r for r, n in cnt.items() if n > 1]
    if dup: notes.append(f"duplicate refs: {dup}")
    bom_all = {r["ref"].upper() for r in rows}; bom_pop = {r["ref"].upper() for r in rows if not r["dni"]}
    refs = {p["ref"] for p in parts}
    side = collections.Counter(p["side"] for p in parts)
    chk = dict(parts=len(parts), top=side["top"], bottom=side["bottom"], with_box=sum(p["box"] is not None for p in parts),
               box_from=dict(collections.Counter(p["box_from"] for p in parts)),
               placement_refs_in_bom=len(refs & bom_all), bom_refs=len(bom_all), bom_populated=len(bom_pop),
               populated_bom_in_placement=len(bom_pop & refs), populated_bom_missing=sorted(bom_pop - refs),
               placement_not_in_bom=sorted(refs - bom_all), bom_file=bf, design_rev=b["rev"], bom_rev=b["bom_rev"],
               ipc_meta=meta or None, frame_check=frame, side_pad_conflicts=len(conf), notes=notes)
    if k == "altium_ascii":
        mism = [(p["ref"], p["src_designator"]) for p in parts if p["src_designator"] and p["src_designator"].upper() != p["ref"]]
        chk["altium_pcb_vs_schematic_designator_diff"] = mism[:30]
    frame_txt = (f"design frame of the file, {units}, y up, seen from top (bottom parts not mirrored); x, y = "
                 + {"allegro_placement": "Allegro symbol origin (SYM_X, SYM_Y)", "place_body_center": "Allegro body centre",
                    "pads_ascii": "PADS part origin", "pads_xy": "PADS part origin (XY export)", "altium_ascii": "Altium component origin (absolute, before board origin offset)",
                    "ipc_only": "pad extent centre (no placement file)"}[k])
    out_parts = [dict(ref=p["ref"], x=round(p["x"], 4), y=round(p["y"], 4), side=p["side"], rot=p["rot"], box=r3(p["box"]),
                      box_from=p["box_from"], footprint=p["footprint"], **({"doubt": p["doubt"]} if p.get("doubt") else {})) for p in parts]
    json.dump(dict(card=card, source=src, units=units, frame=frame_txt, parts=out_parts, checks=chk), open(OUT/f"{card}.json", "w"), indent=1)
    summary[card] = dict(kind=k, **{kk: vv for kk, vv in chk.items() if kk not in ("placement_not_in_bom",)})
    print(card, k, json.dumps({kk: chk[kk] for kk in ("parts", "top", "bottom", "with_box", "placement_refs_in_bom", "bom_populated",
                                                       "populated_bom_in_placement", "frame_check")}))
    print("   missing populated:", chk["populated_bom_missing"][:25], "| not in BoM:", chk["placement_not_in_bom"][:25], notes)
json.dump(summary, open(OUT/"summary.json", "w"), indent=1)
