#!/usr/bin/env python3
"""Merge the owner's link-tool edits (2026-10-21, labelled offline on the laptop) into a 21PCB link ground truth, and compare
them with the pre-links of tools/pcb21_link_prep.py.
Inputs: data/21pcb/linktool/data/<stem>.json (pre-links), owner edits: data/21pcb/linktool_edits_v2/<stem>.json when present (later
re-labelled photos, 2026-10-21: Card14_T), else data/21pcb/linktool_edits/<stem>.json (first md5-checked upload).
Outputs:
  data/21pcb/link_gt_v1.json = {stem: {W, H, batch, parts: [{box, cls, des, text (index into texts, -1 = no printed text),
      checked, flag, added}], texts: [{box, str, src, linked}]}} in native photo pixels (box / scale)
  results/pcb21/link_gt_v1_stats.json = per photo and total: checked, pre-link kept / changed by kind (match / rule / near / none),
      parts added / deleted / box changed / class changed, texts added / deleted / box changed / string changed, unsure flags
  pcb21_link_merge.py   (any python3)"""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"
PRE, EDS = D/"linktool/data", [D/"linktool_edits_v2", D/"linktool_edits"]

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = w*h
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

G = json.load(open(D/"final_v1/gt9_21pcb.json")); out, stats = {}, {}
KINDS = ("match", "rule", "near", "none")
for f in sorted(PRE.glob("*.json")):
    st = f.stem; a = json.load(open(f)); src = next(e/f"{st}.json" for e in EDS if (e/f"{st}.json").exists()); b = json.load(open(src)); sc = a["scale"]
    if src.parent.name != "linktool_edits": print("using", src.parent.name, "for", st)
    ta, tb = {t["id"]: t for t in a["texts"]}, {t["id"]: t for t in b["texts"]}
    pa, pb = {p["id"]: p for p in a["parts"]}, {p["id"]: p for p in b["parts"]}
    s = dict(parts=len(b["parts"]), checked=sum(1 for p in b["parts"] if p.get("checked")), flag=sum(1 for p in b["parts"] if p.get("flag")),
             parts_added=sum(1 for i in pb if i not in pa), parts_deleted=sum(1 for i in pa if i not in pb),
             parts_box=sum(1 for i in pb if i in pa and iou(pa[i]["box"], pb[i]["box"]) < 0.98),
             parts_cls=sum(1 for i in pb if i in pa and pa[i]["cls"] != pb[i]["cls"]),
             texts_added=sum(1 for i in tb if i not in ta), texts_deleted=sum(1 for i in ta if i not in tb),
             texts_box=sum(1 for i in tb if i in ta and iou(ta[i]["box"], tb[i]["box"]) < 0.98),
             texts_str=sum(1 for i in tb if i in ta and (ta[i]["str"] or "") != (tb[i]["str"] or "")))
    for k in KINDS:                                                  # how good was each kind of pre-link (parts present in both)
        ids = [i for i in pb if i in pa and pa[i]["how"] == k]
        s[f"{k}_n"] = len(ids); s[f"{k}_kept"] = sum(1 for i in ids if pa[i]["link"] == pb[i]["link"])
    texts = b["texts"]; idx = {t["id"]: j for j, t in enumerate(texts)}; linked = {p["link"] for p in b["parts"] if p.get("link")}
    N = lambda bx: [round(v / sc, 1) for v in bx]
    out[st] = dict(W=G[st]["W"], H=G[st]["H"], batch=a["batch"],
                   parts=[dict(box=N(p["box"]), cls=p["cls"], des=p.get("des", ""), text=idx.get(p.get("link"), -1),
                               checked=bool(p.get("checked")), flag=bool(p.get("flag")), added=bool(p.get("added"))) for p in b["parts"]],
                   texts=[dict(box=N(t["box"]), str=t.get("str", ""), src=t.get("src", ""), linked=t["id"] in linked) for t in texts])
    stats[st] = s
tot = {k: sum(v[k] for v in stats.values()) for k in next(iter(stats.values()))}
json.dump(out, open(D/"link_gt_v1.json", "w"))
json.dump(dict(total=tot, photos=stats), open(ROOT/"results/pcb21/link_gt_v1_stats.json", "w"), indent=1)
print(f"{'photo':38s} {'chk':>9s} {'match':>9s} {'rule':>9s} {'near':>9s} {'none':>9s}  parts +/-/box/cls   texts +/-/box/str  flag")
for st, s in list(stats.items()) + [("TOTAL", tot)]:
    kk = " ".join(f"{s[k+'_kept']:>4d}/{s[k+'_n']:<4d}" for k in KINDS)
    print(f"{st:38s} {s['checked']:>4d}/{s['parts']:<4d} {kk}  {s['parts_added']:>3d}/{s['parts_deleted']:>3d}/{s['parts_box']:>3d}/{s['parts_cls']:>3d}"
          f"   {s['texts_added']:>3d}/{s['texts_deleted']:>3d}/{s['texts_box']:>3d}/{s['texts_str']:>3d}  {s['flag']:>3d}")
nl = sum(1 for v in out.values() for p in v["parts"] if p["text"] >= 0)
print(f"link GT: {sum(len(v['parts']) for v in out.values())} parts, {nl} linked to a text, "
      f"{sum(len(v['texts']) for v in out.values())} texts ({sum(1 for v in out.values() for t in v['texts'] if t['linked'])} linked)")
