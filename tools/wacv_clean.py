#!/usr/bin/env python3
"""Build WACV-clean v1 (linkdata/CrossClass_data_v1/wacv_clean_v1.json) from wacv.json + the owner's review (page/review, exported db)
+ the design-file checks (results/linking/design_check/*.json). Same schema as wacv.json, plus per part:
  link  = index of the text that names the part, or null (the explicit ground truth; scorers use it instead of string equality,
          because truncated / repeated strings make string equality ambiguous)
  ign   = true when the true label could not be decided (owner answered "看不清" or a conflict is left open) -> left out of scoring
  src   = where the link comes from: wacv (string match) / design / owner / claude
Order of authority: wacv string match < design file (only where its designator is printed in the photo) < Claude's review answers
< the owner's review answers. ML365_Bottom is dropped (owner 2026-10-13: no design file).
Writes also results/linking/wacv_clean_v1_build.json (what changed, per image).
Usage: wacv_clean.py [review_db.json]   (python: ~/venvs/gdino_fewshot)"""
import json, os, sys, copy, collections
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"linkdata/CrossClass_data_v1"
DB = json.load(open(sys.argv[1] if len(sys.argv) > 1 else D/"review_db_2026-10-13.json"))
W = json.load(open(D/"wacv.json")); CANDS = json.load(open(ROOT/"results/linking/wacv_clean_candidates.json"))
DC = {f.stem: json.load(open(f)) for f in sorted((ROOT/"results/linking/design_check").glob("*.json"))}
CLS = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
DROPPED = {"ML365_Bottom"}
# open conflicts decided here until the owner answers: ML365_Top text 107 is linked by #155 (owner) and #156 (Claude, reads R180 = #156's
# designator); #156 kept, #155 left out of scoring
OVERRIDE = {("ML365_Top", 155): "ign"}
# Claude's fixes found while building (2026-10-13), original wacv.json indices. The ML450 clock card is in both ML450_Top1 and _Top2; where
# the photos' labels disagree the silkscreen decides: Top2 had C11 / C12 swapped (C11 is printed straight above the small capacitor #52,
# C12 beside the tantalum #51); Top1 has no box for the printed C22 next to #10 (copied from Top2 text 144, same card, shift 22.7 / 1123.2 px)
FIX = {"ML450_Top2": [("link", 51, 282), ("link", 52, 134)], "ML450_Top1": [("newtext", 10, [490, 643, 506, 672], "C22")]}
RANK = {"wacv": 0, "design": 1, "claude": 2, "owner": 3}
ctr = lambda b: np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
size = lambda b: max(b[2]-b[0], b[3]-b[1], 1)
union = lambda bs: [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]

def default_links(P, T):
    """WACV ground truth: same string. Groups with several parts or texts of one string -> nearest one-to-one (centre distance / part size)."""
    link, amb = [None]*len(P), 0
    ts = collections.defaultdict(list); ps = collections.defaultdict(list)
    for j, t in enumerate(T): ts[t["text"]].append(j)
    for i, p in enumerate(P):
        if p["des"]: ps[p["des"]].append(i)
    for s, I in ps.items():
        J = ts.get(s, [])
        if not J: continue
        if len(I) > 1 or len(J) > 1: amb += 1
        C = np.array([[np.linalg.norm(ctr(P[i]["box"])-ctr(T[j]["box"]))/size(P[i]["box"]) for j in J] for i in I])
        for r, c in zip(*linear_sum_assignment(C)): link[I[r]] = J[c]
    return link, amb

out, log = [], {}
labels = collections.defaultdict(dict)
for a in DB["labels"].values(): labels[a["image"]][a["part"]] = a
for b in W:
    img = b["id"]
    if img in DROPPED: continue
    P = copy.deepcopy(b["parts"]); T = copy.deepcopy(b["texts"]); n = len(P)
    link, amb = default_links(P, T); src = ["wacv" if l is not None else "" for l in link]; ign = [False]*n; dead = [False]*n
    L = collections.Counter(ambiguous_groups=amb)
    # design file: sure statuses only; a link moves only to a text that prints the design's designator
    if img in DC:
        ts = collections.defaultdict(list)
        for j, t in enumerate(T): ts[t["text"]].append(j)
        moved = set()
        for o in DC[img]["parts"]:
            i, true = o["part"], o["design"]
            if o["status"] == "agree" and link[i] is not None: src[i] = "design"
            if o["status"] in ("wrong", "missed") and ts.get(true):
                j = min(ts[true], key=lambda j: np.linalg.norm(ctr(P[i]["box"])-ctr(T[j]["box"])))
                if link[i] != j: L["design_link_moved"] += 1
                P[i]["des"], link[i], src[i] = true, j, "design"; moved.add(j)
        for o in DC[img]["parts"]:                     # a design-only name (later revision) keeps the printed link unless that text moved
            i = o["part"]
            if o["status"] == "wrong" and not ts.get(o["design"]) and link[i] is not None and src[i] == "wacv" and link[i] in moved:
                P[i]["des"], link[i], src[i] = o["design"], None, "design"; L["design_unlinked"] += 1
    # review answers
    newT = {}                                          # newtext id -> text index
    def text_of(a):
        if a.get("nid"):
            if a["nid"] not in newT:
                nt = DB["newtexts"][a["nid"]]; T.append(dict(box=nt["box"], text=nt["str"], angle=None)); newT[a["nid"]] = len(T)-1; L["new_texts"] += 1
            return newT[a["nid"]]
        return a["text"] if a.get("text", -1) is not None and a.get("text", -1) >= 0 else None
    groups = collections.defaultdict(list)
    for i, a in sorted(labels.get(img, {}).items()):
        who = "claude" if a.get("by") == "claude" else "owner"; k = a["kind"]; L[f"review_{k}"] += 1
        if a.get("cfix"): P[i]["cls"] = CLS.index(a["cfix"]); L["class_fixed"] += 1
        if k == "delete": dead[i] = True; continue
        if k == "unsure" or (k == "same" and a.get("unsure")): ign[i], link[i], src[i] = True, None, who
        elif k == "none" or (k == "same" and a.get("none")): link[i], src[i] = None, who
        else:
            j = text_of(a)
            if j is not None:
                if a.get("fix"): T[j]["text"] = a["fix"]; L["text_fixed"] += 1
                P[i]["des"] = a.get("fix") or a.get("pfix") or T[j]["text"]; link[i], src[i] = j, who
        if k == "same":
            gk = (a.get("cand"), a.get("nid") or a.get("text"), bool(a.get("none")), bool(a.get("unsure"))); groups[gk].append(i)
    for gk, I in groups.items():                       # "same component": one part, union box, the class of the biggest box
        if len(I) < 2: continue
        keep = max(I, key=lambda i: (P[i]["box"][2]-P[i]["box"][0])*(P[i]["box"][3]-P[i]["box"][1]))
        P[keep]["box"] = union([P[i]["box"] for i in I])
        for i in I:
            if i != keep: dead[i] = True
        L["merged_parts"] += len(I)-1
    for f in FIX.get(img, []):
        if f[0] == "link": link[f[1]], src[f[1]] = f[2], "claude"
        else: T.append(dict(box=f[2], text=f[3], angle=None)); link[f[1]], src[f[1]] = len(T)-1, "claude"; L["new_texts"] += 1
        P[f[1]]["des"] = T[link[f[1]]]["text"]; L["claude_fix"] += 1
    for pb in DB["partboxes"].values():
        if pb["image"] == img: P[pb["part"]]["box"] = pb["box"]; L["boxes_redrawn"] += 1
    for (im, i), v in OVERRIDE.items():
        if im == img and v == "ign": ign[i], link[i], src[i] = True, None, "override"; L["override_ign"] += 1
    for k_, np_ in sorted(DB["newparts"].items()):     # parts the owner drew (and Claude copied between photos of the same card)
        if np_["image"] != img: continue
        j = np_["text"] if np_.get("text", -1) >= 0 else None
        if j is not None and np_.get("str") and T[j]["text"] != np_["str"]: T[j]["text"] = np_["str"]; L["text_fixed"] += 1
        P.append(dict(box=np_["box"], des=np_.get("str", "") if j is not None else "", cls=CLS.index(np_["cls"])))
        link.append(j); src.append("claude" if np_.get("by") == "claude" else "owner"); ign.append(False); dead.append(False); L["new_parts"] += 1
    # one text names at most one part: the higher authority keeps it; the others lose it and their true label is unknown -> left out
    # of scoring; equal authority -> all left out
    by_t = collections.defaultdict(list)
    for i, j in enumerate(link):
        if j is not None and not dead[i]: by_t[j].append(i)
    for j, I in by_t.items():
        if len(I) < 2: continue
        top = max(RANK.get(src[i], 0) for i in I); win = [i for i in I if RANK.get(src[i], 0) == top]
        for i in I:
            if i not in win or len(win) > 1:
                link[i], ign[i] = None, True; L["conflict_tie" if i in win else "conflict_lost"] += 1
    # dropped texts (none marked "not a designator" in this review), then renumber
    drop_t = {int(k.split("__t")[1]) for k in DB["nontext"] if k.startswith(img + "__t")}
    tmap = {}; T2 = []
    for j, t in enumerate(T):
        if j not in drop_t: tmap[j] = len(T2); T2.append(t)
    P2 = []
    for i, p in enumerate(P):
        if dead[i]: continue
        j = tmap.get(link[i]) if link[i] is not None else None
        d = p["des"] if j is not None else (p["des"] if p["des"] not in {t["text"] for t in T2} else "")
        if j is None and d != p["des"]: L["des_blanked"] += 1
        q = dict(box=p["box"], des=d if j is None else T2[j]["text"], cls=p["cls"], link=j, src=src[i] or "wacv")
        if ign[i]: q["ign"] = True
        P2.append(q)
    for p in P2:                                       # the explicit link and the strings agree
        assert p["link"] is None or T2[p["link"]]["text"] == p["des"], (img, p)
    lk = [p["link"] for p in P2 if p["link"] is not None]; assert len(lk) == len(set(lk)), img
    L["parts"], L["texts"], L["links"], L["ignored"] = len(P2), len(T2), len(lk), sum(1 for p in P2 if p.get("ign"))
    L["links_wacv_before"] = sum(1 for p in b["parts"] if p["des"] and p["des"] in {t["text"] for t in b["texts"]})
    for p in P2: L["src_" + p["src"]] += p["link"] is not None
    out.append(dict(b, parts=P2, texts=T2, clean="v1")); log[img] = dict(sorted(L.items()))
json.dump(out, open(D/"wacv_clean_v1.json", "w"))
tot = collections.Counter()
for v in log.values(): tot.update(v)
json.dump(dict(total=dict(sorted(tot.items())), images=log), open(ROOT/"results/linking/wacv_clean_v1_build.json", "w"), indent=1)
print(len(out), "images |", dict(sorted(tot.items())))
