#!/usr/bin/env python3
"""Feasibility probe for a 'design knowledge' model from our KiCad corpus (2026-10-17, owner idea 3). Our extracted dataset keeps only
box / designator / prefix class per part; the .kicad_pcb files also hold the footprint library name, the Value field and the net of every
pad. On a random sample of CrossClass_data_v2 layouts this reports: how often the footprint name / value give the part type (keyword
rules), how often that type disagrees with the designator-prefix class (board naming conventions, e.g. L for LEDs), and simple circuit
facts from the nets (LED sharing a net with a resistor, crystal sharing nets with capacitors).
Usage: /usr/bin/python3 kicad_world_probe.py [--n 300] [--workers 8]   (system python with pcbnew; CROSSCLASS project root)"""
import argparse, collections, json, os, random, re, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))
NM = ["Resistor", "Capacitor", "Inductor", "Connector", "Diode", "Switch", "Transistor", "IC", "Oscillator"]
RULES = [(r"\bLED\b|LED_|_LED|WS2812|SK6812", 4), (r"crystal|xtal|oscillator|resonator|\bosc", 8), (r"button|switch|\bSW_|tact|relay", 5),
         (r"connector|pin_?header|pinheader|socket|usb|jst|terminal|molex|\bjack\b|rj45|\bconn", 3), (r"inductor|ferrite|\bL_\d|choke|\bbead", 2),
         (r"resistor|\bR_\d|potentiometer|trimmer|\bR_Array|resistor_array", 0), (r"capacitor|\bC_\d|\bCP_|elec|tantal|\bcap", 1),
         (r"diode|\bD_SOD|\bD_SMA|\bD_SMB|\bD_SMC|\bD_MiniMELF|schottky|zener|\bTVS", 4),
         (r"Package_SO|SOIC|TSSOP|SSOP|QFN|QFP|BGA|DFN|DIP-\d|\bMSOP|LGA|PLCC|\bTSOT-23-[5-9]|SOT-23-[5-9]", 7)]
def fp_type(lib, name, value):
    s = f"{lib}:{name} {value}"
    for pat, c in RULES:
        if re.search(pat, s, re.I): return c
    return None
def probe(key):
    import pcbnew
    from kicad_links import DES, is_component, pcls
    try: brd = pcbnew.LoadBoard(str(ROOT/f"kicad_scan/boards/{key}.kicad_pcb"))
    except Exception: return None
    parts = []
    for fp in brd.GetFootprints():
        ref = fp.GetReference()
        if not DES.match(ref) or not is_component(fp, ref): continue
        fid = fp.GetFPID(); lib, name = str(fid.GetLibNickname()), str(fid.GetLibItemName())
        nets = sorted({p.GetNetname() for p in fp.Pads() if p.GetNetname()})
        parts.append(dict(ref=ref, pc=pcls(ref), ft=fp_type(lib, name, fp.GetValue()), lib=lib, name=name, value=fp.GetValue(), nets=nets, npads=len(fp.Pads())))
    return key, parts
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    keys = sorted({b["id"].rsplit("_", 1)[0] for b in json.load(open(ROOT/"linkdata/CrossClass_data_v2/kicad.json"))}); random.Random(0).shuffle(keys)
    with Pool(a.workers) as pool: res = [r for r in pool.imap_unordered(probe, keys[:a.n]) if r]
    P = [p for _, ps in res for p in ps]; n = len(P); known = [p for p in P if p["ft"] is not None]
    print(f"layouts {len(res)}, component parts {n}; footprint/value gives a type for {len(known)} ({100*len(known)/n:.1f} %); pads with a net: "
          f"{100*sum(1 for p in P if p['nets'])/n:.1f} % of parts")
    both = [p for p in known if 0 <= p["pc"] <= 8]; agree = sum(p["pc"] == p["ft"] for p in both)
    print(f"parts with a prefix class (0-8) and a footprint type: {len(both)}; agree {agree} ({100*agree/len(both):.1f} %)")
    dis = collections.Counter((re.match(r"[A-Z]+", p["ref"]).group(0), NM[p["pc"]], NM[p["ft"]]) for p in both if p["pc"] != p["ft"])
    boards = collections.defaultdict(set)
    for key, ps in res:
        for p in ps:
            if p["ft"] is not None and 0 <= p["pc"] <= 8 and p["pc"] != p["ft"]: boards[(re.match(r"[A-Z]+", p["ref"]).group(0), NM[p["ft"]])].add(key)
    print("prefix class != footprint type (prefix, prefix says, footprint says): parts, layouts")
    for k, v in dis.most_common(15): print(f"  {k}: {v}, {len(boards[(k[0], k[2])])}")
    other = collections.Counter((re.match(r"[A-Z]+", p["ref"]).group(0), NM[p["ft"]]) for p in known if not 0 <= p["pc"] <= 8)
    print("prefix class 'other' (9) but footprint gives a type:", dict(other.most_common(10)))
    # circuit facts from nets
    led = ser = xt = xtc = 0
    for key, ps in res:
        netparts = collections.defaultdict(list)
        for p in ps:
            for nn in p["nets"]: netparts[nn].append(p)
        for p in ps:
            if p["ft"] == 4 and re.search(r"LED", f"{p['lib']}{p['name']}{p['value']}", re.I):
                led += 1; ser += any(q["ft"] == 0 for nn in p["nets"] if not re.search(r"GND|VCC|VDD|\+\d", nn, re.I) for q in netparts[nn] if q is not p)
            if p["ft"] == 8:
                xt += 1; xtc += sum(1 for nn in p["nets"] if not re.search(r"GND", nn, re.I) for q in netparts[nn] if q is not p and q["ft"] == 1) >= 2
    print(f"LEDs {led}: share a signal net with a resistor {ser} ({100*ser/max(led,1):.0f} %); crystals {xt}: >= 2 capacitors on their nets {xtc} ({100*xtc/max(xt,1):.0f} %)")
    json.dump({k: ps for k, ps in res}, open("/tmp/xinrui_world_probe.json", "w"))
