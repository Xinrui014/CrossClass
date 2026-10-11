#!/usr/bin/env python3
"""Full scan of open-schematics for the linking data (owner 2026-10-11: use much more of open-schematics).
Every .kicad_pcb file of every row (not only the largest board of a project) is parsed; per board file:
  layout hash (place_hash, from parse_kicad_pcb.board_stats) and netlist hash, part count, size, KiCad version,
  per side: footprints and references visible on that side's silkscreen; SMD pad share.
Boards with >= 10 visible references on a side are written once per layout hash to kicad_scan/boards/<place_hash>.kicad_pcb.
Per-shard results kicad_scan/rows/<shard>.jsonl (resumable); merge with --merge -> kicad_scan/scan.jsonl + summary.
Usage: kicad_scan.py [workers] | kicad_scan.py --merge   (python: ~/miniconda3/envs/pcbprior)"""
import json, re, sys, os
from pathlib import Path
from multiprocessing import Pool
import pyarrow.parquet as pq
H = Path.home(); SP = H/"projects/PCB_structure_prior"; sys.path.insert(0, str(SP/"tools")); import parse_kicad_pcb as P
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass")); OUT = ROOT/"kicad_scan"
SHARDS = SP/"data/open_schematics/data"
REF_NEW = re.compile(r'\(property\s+"Reference"\s+"([^"]*)"(.*?)\(effects(.*?)\)\s*\)', re.S)
REF_OLD = re.compile(r'\(fp_text\s+reference\s+"?([^"\s)]*)"?(.*?)\(effects(.*?)\)\s*\)', re.S)
DES = re.compile(r"^[A-Z]{1,4}\d{1,4}[A-Z]?$")
HIDE = re.compile(r'\bhide\b(?!\s+no)|\(hide\s+yes\)')

def side_counts(txt):
    """Per side: footprints, references that look like designators and are visible on that side's silkscreen; SMD pad share."""
    chunks = re.split(r'\n\s*\((?:footprint|module)\s', txt)[1:]
    c = dict(top_fp=0, bottom_fp=0, vis_top=0, vis_bottom=0); smd = th = 0
    for ch in chunks:
        head = ch[:600]; side = "top" if re.search(r'\(layer\s+"?F\.Cu"?\)', head) else "bottom" if re.search(r'\(layer\s+"?B\.Cu"?\)', head) else None
        smd += len(re.findall(r'\(pad\s+"?[^"\s]*"?\s+smd\b', ch)); th += len(re.findall(r'\(pad\s+"?[^"\s]*"?\s+(?:thru_hole|np_thru_hole)\b', ch))
        if side is None: continue
        c[f"{side}_fp"] += 1; m = REF_NEW.search(ch) or REF_OLD.search(ch)
        if not m or not DES.match(m.group(1)): continue
        body = m.group(2) + m.group(3); silk = "F.SilkS" if side == "top" else "B.SilkS"
        if re.search(rf'layer\s+"?{re.escape(silk)}"?', body) and not HIDE.search(body): c[f"vis_{side}"] += 1
    c["smd_share"] = round(smd/max(smd+th, 1), 3); return c

def do_shard(shard):
    out_f = OUT/"rows"/f"{shard.replace('.parquet', '')}.jsonl"
    if out_f.exists(): return shard, 0
    recs = []
    try:
        pf = pq.ParquetFile(SHARDS/shard); r0 = 0
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["name", "pcb_files"]).to_pylist()
            for i, row in enumerate(t):
                for fi, txt in enumerate((row["pcb_files"] or [])[:20]):
                    rec = dict(shard=shard, row=r0+i, fidx=fi, name=row["name"])
                    if not txt or "kicad_pcb" not in txt[:200]: continue
                    try:
                        b = P.parse_board(txt)
                        if not b.get("ok"): rec["ok"] = False; recs.append(rec); continue
                        s = P.board_stats(b)
                    except Exception:
                        rec["ok"] = False; recs.append(rec); continue
                    rec.update(ok=True, n_components=s["n_components"], n_footprints=s["n_footprints"], width_mm=s["width_mm"], height_mm=s["height_mm"],
                               place_hash=s["place_hash"], netlist_hash=s["netlist_hash"], version=s["version"], **side_counts(txt))
                    if max(rec["vis_top"], rec["vis_bottom"]) >= 10:
                        f = OUT/"boards"/f"{s['place_hash']}.kicad_pcb"
                        if not f.exists():
                            tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(txt); os.replace(tmp, f)
                        rec["saved"] = True
                    recs.append(rec)
            r0 += len(t)
    except Exception as ex:
        print(shard, "FAILED", repr(ex)[:120], flush=True); return shard, -1
    with open(out_f.with_suffix(".part"), "w") as fh:
        for r in recs: fh.write(json.dumps(r)+"\n")
    os.replace(out_f.with_suffix(".part"), out_f); return shard, len(recs)

def merge():
    recs = [json.loads(l) for f in sorted((OUT/"rows").glob("*.jsonl")) for l in open(f)]
    with open(OUT/"scan.jsonl", "w") as fh:
        for r in recs: fh.write(json.dumps(r)+"\n")
    ok = [r for r in recs if r.get("ok")]; uniq = {}
    for r in ok: uniq.setdefault(r["place_hash"], r)
    u = list(uniq.values()); vis = lambda r: max(r["vis_top"], r["vis_bottom"])
    summ = dict(shards=len(list((OUT/"rows").glob("*.jsonl"))), board_files=len(recs), parsed=len(ok), projects_with_board=len({(r["shard"], r["row"]) for r in ok}),
                unique_layouts=len(u), unique_ge20_parts=sum(r["n_components"] >= 20 for r in u),
                unique_ge10_visible_side=sum(vis(r) >= 10 for r in u), unique_ge10_visible_and_ge20_parts=sum(vis(r) >= 10 and r["n_components"] >= 20 for r in u),
                unique_netlists_among_ge10_visible=len({r["netlist_hash"] for r in u if vis(r) >= 10}),
                saved_files=len(list((OUT/"boards").glob("*.kicad_pcb"))))
    json.dump(summ, open(OUT/"summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    if "--merge" in sys.argv: merge(); sys.exit()
    (OUT/"rows").mkdir(parents=True, exist_ok=True); (OUT/"boards").mkdir(parents=True, exist_ok=True)
    shards = sorted(p.name for p in SHARDS.glob("*.parquet")); n = 0
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        for k, (s, cnt) in enumerate(pool.imap_unordered(do_shard, shards)):
            n += max(cnt, 0)
            if k % 25 == 0: print(f"shards {k+1}/{len(shards)}, board files so far {n}", flush=True)
    merge()
