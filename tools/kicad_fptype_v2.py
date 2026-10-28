#!/usr/bin/env python3
"""Final class target v2 of KiCad parts (owner OK 2026-10-23 16:13; rules and probe numbers in results/analysis/2026-10-23_fptype_rules_v2.md).
No pcbnew: reads the stored footprint / Value strings of linkdata/kicad_fptype/<side>.json (v1, made by tools/kicad_fptype.py with the
old RULES of tools/kicad_world_probe.py) and the designators of kicad.json, and writes linkdata/kicad_fptype_v2/<side>.json
{id, fptype: [final class 0-8, or 9 = ignored, per kicad.json part], fpclass: [footprint / Value class or -1], rule: [...], fp, value}.
"fptype" is the final target, so `linkformer.py --fptype linkdata/kicad_fptype_v2` uses it as is (9 = no class loss). The v1 folder is not changed.
Rules, first match wins (rule number stored per part):
  1 Value part numbers (transistor, diode, IC); skipped on connector footprints of J / P / CN / CON / X parts; the transistor and
    diode patterns are skipped for thermistors (TH / RT / NTC / PTC)
  2 SOT / TO package families (incl. SC-59): class from a designator prefix whitelist; any other prefix falls through to rule 3
  3 footprint / Value keywords: feedthrough -> C, the old RULES with transformer / magnetics -> L (footprint name, Value, KiCad Transformer_SMD / _THT library) inserted before the inductor rule
  4 no footprint / Value class: the designator prefix table (kicad_links.PFX + additions)
  5 no footprint / Value class and prefix T / TR (transformers, terminals, test points): ignored (9)
  6 no footprint / Value class and a prefix outside the table: ignored (9)
Usage: python3 tools/kicad_fptype_v2.py [--out linkdata/kicad_fptype_v2] [--check]   (CROSSCLASS = project root)
--check: no writes; asserts v1 fptype == old RULES on the stored strings and prints the counts."""
import argparse, collections, hashlib, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kicad_world_probe import RULES as OLD
from kicad_links import PFX, prefix
I = re.I
V1 = [(re.compile(r"\bBSS\d|\b2N\d{4}|\bMMBT|\bBC\d{3}|\bBCX\d|\bAO\d{4}|\bSi(?!70\d\d|8\d{3}|5351)\d{4}|\bIRF|\bIRL(?!81)|\bFDN|\bDMG\d|\bDMN\d|\bDMP\d|\bPMV\d|\bNTR\d|\bTIP\d|"
                  r"\bS8050|\bS8550|\bSS8050|MOSFET|\bNPN\b|\bPNP\b|transistor|\bBJT\b|\bQ_[NP]MOS|\bQ_NPN|\bQ_PNP|\bFZT|\bZXM", I), 6),
      (re.compile(r"\bBAT\d|\bBAV\d|\b1N\d{4}|\bSS\d{2}\b|\bSM[ABC]J|\bMBR\d|\bUS1|\bRB\d{3}|\bPMEG|\bSMAJ|\bSMBJ|\bD_Small|\bD_Schottky", I), 4),
      (re.compile(r"AMS1117|LM1117|NCP1117|LD1117|AZ1117|\b78L?\d\d|\bLM78|\bLM317|\bAP2112|\bMCP73|\bXC6206|\bLMV321|\bLM321|USBLC6|SRV05|74LVC|74HC|"
                  r"\bDS18B20|\bA3144|Hall_Sensor", I), 7)]
CONN = re.compile(r"pin_?header|pin_?socket|jst|terminal|molex|connector|\bconn", I); CONN_P = {"J", "P", "CN", "CON", "X"}; THERM_P = {"TH", "RT", "NTC", "PTC"}
PKG = re.compile(r"(?<![a-z])SOT[-_]?23(?!\d)|SOT[-_]?323|(?<![a-z])SC[-_]?70|SOT[-_]?363|(?<![a-z])SC[-_]?88|(?<![a-z])SC[-_]?59(?!\d)|SOT[-_]?416|SOT[-_]?89(?!\d)|SOT[-_]?223|"
                 r"(?<![a-z])TO[-_]?92|(?<![a-z])TO[-_]?220|(?<![a-z])TO[-_]?252|(?<![a-z0-9])D2?PAK|(?<![a-z])TO[-_]?263|(?<![a-z])TO[-_]?247|(?<![a-z])TO[-_]?3(?![\d])|"
                 r"Package_TO_SOT_(SMD|THT)", I)
PKG_P = {**{p: 7 for p in ("U", "IC", "VR", "REG", "LDO", "PS")}, **{p: 4 for p in ("D", "DB", "DD", "TVS", "ZD", "LED", "CR", "DZ")},
         **{p: 6 for p in ("Q", "T", "TR", "FET", "VT", "QR", "QL", "M", "PQ", "QA")}}
XFMR = (re.compile(r"transformer|xfmr|toroid|\bbalun|\bADT\d-\d", I), 2)
R3 = [(re.compile(r"feedthrough", I), 1)] + [(re.compile(p, I), c) for p, c in OLD[:4]] + [XFMR] + [(re.compile(p, I), c) for p, c in OLD[4:]]
assert OLD[4][1] == 2, "OLD[4] must be the inductor rule"
PFX2 = dict(PFX, MX=5, CH=5, LC=1, CB=1, RR=0, RD=0, KD=4, DR=4, DP=4); RESTRICT = {"T", "TR"}
OLDC = [(re.compile(p, I), c) for p, c in OLD]

def old_type(fp, v):
    s = f"{fp} {v}"
    return next((c for p, c in OLDC if p.search(s)), -1)

def fp_class(fp, v, des):
    """(footprint / Value class or -1, rule 1 / 2 / 3, or 0 when no rule fires) for footprint 'lib:name', Value v and designator des."""
    if not fp: return -1, 0
    pre = prefix(des)
    if not (pre in CONN_P and CONN.search(fp)):
        for p, c in V1:
            if c in (4, 6) and pre in THERM_P: continue
            if p.search(v): return c, 1
    if PKG.search(fp) and pre in PKG_P: return PKG_P[pre], 2
    s = f"{fp} {v}"; lib, _, name = fp.rpartition(":")   # the transformer keyword ignores project library names (for example "LWA Balun 2020"),
    s_name = f"{lib if re.fullmatch(r'Transformer_(SMD|THT)', lib, I) else ''} {name} {v}"   # but keeps the KiCad libraries Transformer_SMD / _THT
    return next(((c, 3) for p, c in R3 if p.search(s_name if p is XFMR[0] else s)), (-1, 0))

def target(fp, v, des):
    """(final class 0-8 or 9 = ignored, footprint class, rule 1-6)."""
    t, r = fp_class(fp, v, des)
    if t >= 0: return t, t, r
    pre = prefix(des)
    if pre in RESTRICT: return 9, -1, 5
    c = PFX2.get(pre, 9)
    return c, -1, (4 if c < 9 else 6)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="linkdata/kicad_fptype_v2"); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); SRC = ROOT/"linkdata/kicad_fptype"; OUT = ROOT/a.out
    if not a.check:
        if OUT.exists(): sys.exit(f"refused: {OUT} exists")
        OUT.mkdir(parents=True)
    t0 = time.time(); c = collections.Counter(); h = []
    for b in json.load(open(ROOT/"linkdata/CrossClass_data_v2/kicad.json")):
        F = json.load(open(SRC/f"{b['id']}.json")); assert len(F["fptype"]) == len(b["parts"]), b["id"]
        ft, fc, ru = [], [], []
        for p, st, fp, v in zip(b["parts"], F["fptype"], F["fp"], F["value"]):
            assert (old_type(fp, v) if fp else -1) == st, (b["id"], p["des"])
            y, t, r = target(fp, v, p["des"]); ft.append(y); fc.append(t); ru.append(r)
            c[f"rule{r}"] += 1; c["parts"] += 1; c[f"class{y}"] += 1; c["fpclass_changed_vs_v1"] += t != st
        c["sides"] += 1
        if not a.check:
            s = json.dumps(dict(id=b["id"], fptype=ft, fpclass=fc, rule=ru, fp=F["fp"], value=F["value"])); open(OUT/f"{b['id']}.json", "w").write(s)
            h.append(f"{b['id']}.json {hashlib.md5(s.encode()).hexdigest()}")
    print(dict(sorted(c.items())), f"{time.time()-t0:.0f} s")
    if not a.check:
        md5 = hashlib.md5("\n".join(sorted(h)).encode()).hexdigest()
        json.dump(dict(rules="final v2 (owner OK 2026-10-23 16:13; main 16:30)", script_md5=hashlib.md5(open(__file__, "rb").read()).hexdigest(), source=str(SRC),
                       counts=dict(sorted(c.items())), files=len(h), md5_of_sorted_file_md5s=md5), open(ROOT/f"{a.out}.meta.json", "w"), indent=1)
        print(f"wrote {len(h)} files to {OUT}; md5 of the sorted '<file> <md5>' lines {md5}")
