#!/usr/bin/env python3
"""Stage 1 of the KiCad link set: pick WACV-like open-schematics boards and write their .kicad_pcb text.
Candidates = the same filter as gdino_fewshot/kicad_render/select_boards.py (usable, routed, unique placement, 20-350 parts,
2.1-11.2 parts/cm^2, R+C share >= 0.4, >= 1 IC) -> 2,677 boards. Per board (largest .kicad_pcb of the project): top-side references
visible on F.SilkS >= 70 % and SMD pad share >= 60 %. No 3D-model check (layout only, nothing is rendered).
Output kicad_boards/<id>.kicad_pcb + kicad_boards/selected.json (git-ignored).
Usage: kicad_select.py [workers]   (python: ~/miniconda3/envs/pcbprior)"""
import json, re, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq
H = Path.home(); SP = H/"projects/PCB_structure_prior"; sys.path.insert(0, str(SP/"tools")); import parse_kicad_pcb as P
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass")); OUT = ROOT/"kicad_boards"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(H/"projects/gdino_fewshot/kicad_render"))

REF_NEW = re.compile(r'\(property\s+"Reference"\s+"([^"]*)"(.*?)\(effects(.*?)\)\s*\)', re.S)
REF_OLD = re.compile(r'\(fp_text\s+reference\s+"?([^"\s)]*)"?(.*?)\(effects(.*?)\)\s*\)', re.S)
def board_checks(txt):   # same as select_boards.py
    chunks = re.split(r'\n\s*\((?:footprint|module)\s', txt)[1:]
    top = vis = smd = th = 0
    for ch in chunks:
        on_top = bool(re.search(r'\(layer\s+"?F\.Cu"?\)', ch[:400]))
        smd += len(re.findall(r'\(pad\s+"?[^"\s]*"?\s+smd\b', ch)); th += len(re.findall(r'\(pad\s+"?[^"\s]*"?\s+(?:thru_hole|np_thru_hole)\b', ch))
        if not on_top: continue
        top += 1; m = REF_NEW.search(ch) or REF_OLD.search(ch)
        if not m: continue
        body = m.group(2) + m.group(3)
        if re.search(r'layer\s+"?F\.SilkS"?', body) and not re.search(r'\bhide\b(?!\s+no)', body) and not re.search(r'\(hide\s+yes\)', body): vis += 1
    return dict(top_fp=top, vis_share=vis/max(top, 1), smd_share=smd/max(smd+th, 1))

def candidates():
    d = pd.read_parquet(SP/"results/boards.parquet"); d = d[(d.usable == True) & (d.routed == True)].drop_duplicates("place_hash")
    d["area_cm2"] = d.width_mm*d.height_mm/100; d = d[(d.area_cm2 > 1) & (d.area_cm2 < 2000)]; d["dens"] = d.n_components/d.area_cm2
    pre = d.prefixes.apply(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    d["rc"] = pre.apply(lambda p: (p.get("R", 0)+p.get("C", 0))/max(sum(p.values()), 1)); d["has_U"] = pre.apply(lambda p: p.get("U", 0) > 0)
    return d[d.n_components.between(20, 350) & d.dens.between(2.1, 11.2) & (d.rc >= 0.4) & d.has_U]

def do_shard(args):
    shard, rows = args; out = []
    try:
        pf = pq.ParquetFile(SP/"data/open_schematics/data"/shard); starts = np.cumsum([0]+[pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        for g in range(pf.num_row_groups):
            want = [r for r in rows if starts[g] <= r < starts[g+1]]
            if not want: continue
            t = pf.read_row_group(g, columns=["name", "pcb_files"]).to_pylist()
            for r in want:
                row = t[r-starts[g]]; best, btxt = None, None
                for txt in (row["pcb_files"] or [])[:8]:
                    if not txt or "kicad_pcb" not in txt[:200]: continue
                    b = P.parse_board(txt)
                    if not b.get("ok"): continue
                    s = P.board_stats(b)
                    if best is None or s["n_components"] > best["n_components"]: best, btxt = s, txt
                if btxt is None: continue
                chk = board_checks(btxt)
                if chk["vis_share"] >= 0.7 and chk["smd_share"] >= 0.6:
                    key = f"{shard.replace('.parquet', '')}_r{r}"; (OUT/f"{key}.kicad_pcb").write_text(btxt)
                    out.append(dict(key=key, name=row["name"], shard=shard, row=int(r), **chk))
    except Exception as ex:
        print(shard, "FAILED", repr(ex)[:120], flush=True)
    return out

if __name__ == "__main__":
    c = candidates(); print("candidates", len(c), flush=True)
    jobs = sorted(c.groupby("shard").row.apply(lambda s: sorted(int(v) for v in s)).items())
    sel = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for k, res in enumerate(pool.imap_unordered(do_shard, jobs)):
            sel += res
            if k % 50 == 0: print(f"shards {k+1}/{len(jobs)}, kept {len(sel)}", flush=True)
    json.dump(sorted(sel, key=lambda s: s["key"]), open(OUT/"selected.json", "w"), indent=0)
    print("kept", len(sel), "of", len(c))
