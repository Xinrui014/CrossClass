#!/usr/bin/env python3
"""Score the VLM class decider (qwen_class.py): replace the detector's class by Qwen's class under a policy, class-wise NMS,
standard metrics (det_metrics.py) on the 3 dev images and the 42 test images; policy chosen on dev, all reported.
Policies: H = Qwen confidence high; Hd = high AND detector score < 0.7; HM = high or medium; HMd = HM AND detector score < 0.7;
ALL = every answer (not unsure); Hd_nonRC = Hd but the new class may not be resistor/capacitor; Hd_look = Hd and the new class is one
of inductor/diode/switch/transistor/oscillator (policies fixed before scoring, chosen on dev). Also: class accuracy of Qwen alone vs detector on detections that sit on a GT part (IoU >= 0.5),
and the 9x9 confusion of Qwen's class. Usage (in err9/): qwen_class_score.py --tag qc2"""
import argparse, json, os, subprocess, sys, collections
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); E = ROOT/"data"; Q = ROOT/"answers"; R = ROOT/"results"
sys.path.insert(0, str(ROOT/"tools")); os.chdir(E)   # analyze9 / det_metrics read gt9_*.json + preds_*_1280.json from the cwd
from analyze9 import load, iou, NM
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--dev", nargs="*", default=["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"])
a = ap.parse_args()
ans = {}
for l in open(Q/f"answers_{a.tag}.jsonl"):
    r = json.loads(l); ans[r["key"]] = r
preds = json.load(open(E/"preds_wacv_1280.json")); G = json.load(open(E/"gt9_wacv.json"))
test_imgs = [s for s in G if s not in a.dev]; dev_imgs = [s for s in G if s in a.dev]

def nms_classwise(boxes, thr=0.6):
    if not boxes: return boxes
    b = np.array(boxes); keep = []
    for c in np.unique(b[:, 5]):
        idx = np.where(b[:, 5] == c)[0]; idx = idx[np.argsort(-b[idx, 4])]; sup = np.zeros(len(idx), bool); I = iou(b[idx, :4], b[idx, :4])
        for i in range(len(idx)):
            if sup[i]: continue
            keep.append(idx[i]); sup |= I[i] >= thr; sup[i] = True
    return [boxes[i] for i in sorted(keep)]

POL = {"H": (("high",), 1.01, None), "Hd": (("high",), 0.7, None), "HM": (("high", "medium"), 1.01, None), "HMd": (("high", "medium"), 0.7, None),
       "ALL": (("high", "medium", "low"), 1.01, None),
       "Hd_nonRC": (("high",), 0.7, {2, 3, 4, 5, 6, 7, 8}),      # Qwen may only move a part OUT of resistor / capacitor (its mass error is C -> R)
       "Hd_look": (("high",), 0.7, {2, 4, 5, 6, 8})}              # ... and only into the look-alike classes inductor / diode / switch / transistor / oscillator
def subset(name, imgs, P):
    json.dump({s: P[s] for s in imgs}, open(E/f"preds_{name}_1280.json", "w"))
    if not (E/f"gt9_{name}.json").exists(): json.dump({s: G[s] for s in imgs}, open(E/f"gt9_{name}.json", "w"))
subset("wacv42", test_imgs, preds); subset("wacvdev3", dev_imgs, preds)
changed = {}; sets_test = ["base=wacv42"]; sets_dev = ["base=wacvdev3"]
for pol, (confs, dmax, allow) in POL.items():
    out = {}; n = 0
    for st, bl in preds.items():
        nb = []
        for k, p in enumerate(bl):
            p = list(p); r = ans.get(f"{st}|{k}")
            if r and r["qwen_cls"] is not None and r["confidence"] in confs and p[4] < dmax and r["qwen_cls"] != int(p[5]) and (allow is None or r["qwen_cls"] in allow): p[5] = r["qwen_cls"]; n += 1
            nb.append(p)
        out[st] = nms_classwise(nb)
    changed[pol] = n; subset(f"wacv{a.tag}{pol}42", test_imgs, out); subset(f"wacv{a.tag}{pol}dev3", dev_imgs, out)
    sets_test.append(f"{pol}=wacv{a.tag}{pol}42"); sets_dev.append(f"{pol}=wacv{a.tag}{pol}dev3")
def metrics(sets):
    subprocess.run([sys.executable, str(ROOT/"tools/det_metrics.py")] + sets, cwd=E, check=True, stdout=subprocess.DEVNULL)
    return {r["set"]: r for r in json.load(open(E/"det_metrics_q38.json"))}
Mt, Md = metrics(sets_test), metrics(sets_dev)
best = max(POL, key=lambda p: Md[p]["map50"])
# Qwen's class alone vs the detector, on detections on a GT part (IoU >= 0.5), all 45 images
tot = collections.Counter(); conf_mat = np.zeros((9, 9), int); byconf = collections.defaultdict(lambda: [0, 0]); ev = collections.Counter()
for d in load("wacv"):
    for k, p in enumerate(preds[d["st"]]):
        r = ans.get(f"{d['st']}|{k}")
        if not r or not len(d["gm"]): continue
        I = iou(np.array([p[:4]]), d["gm"])[0]
        if I.max() < 0.5: continue
        gc = int(d["gc"][int(np.argmax(I))]); tot["on_part"] += 1; tot["det_right"] += int(p[5]) == gc; ev[r["evidence"]] += 1
        if r["qwen_cls"] is None: tot["qwen_unsure"] += 1; continue
        tot["qwen_right"] += r["qwen_cls"] == gc; conf_mat[gc, r["qwen_cls"]] += 1
        byconf[r["confidence"]][0] += r["qwen_cls"] == gc; byconf[r["confidence"]][1] += 1
        if int(p[5]) != gc: tot["det_wrong"] += 1; tot["det_wrong_qwen_right"] += r["qwen_cls"] == gc
        else: tot["det_right_qwen_wrong"] += r["qwen_cls"] != gc
L = [f"# VLM class decider on WACV — tag {a.tag}", "", f"Answers: {len(ans)}; class changes per policy: {changed}; policy chosen on the 3 dev images: **{best}** (dev mAP50)", "",
     "| 42 test images | precision | recall | F1 | macro-F1 | mAP50 | mAP50-95 | dev mAP50 |", "|---|---|---|---|---|---|---|---|"]
for n in ["base"] + list(POL):
    m = Mt[n]; L.append(f"| {'detector alone' if n == 'base' else 'policy ' + n}{' (chosen)' if n == best else ''} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['macro_f1']:.3f} | {m['map50']:.3f} | {m['map50_95']:.3f} | {Md[n]['map50']:.3f} |")
op = max(tot["on_part"], 1)
L += ["", "## Class accuracy on detections that sit on a GT part (IoU ≥ 0.5, all 45 images)", "",
      f"- detections on a part: {tot['on_part']}; detector class right {tot['det_right']} ({100*tot['det_right']/op:.1f} %); Qwen class right {tot['qwen_right']} ({100*tot['qwen_right']/op:.1f} %); Qwen unsure {tot['qwen_unsure']}",
      f"- detector wrong {tot['det_wrong']}: Qwen right on {tot['det_wrong_qwen_right']}; detector right but Qwen wrong: {tot['det_right_qwen_wrong']}",
      "- Qwen accuracy by its confidence: " + ", ".join(f"{c} {v[0]}/{v[1]} ({100*v[0]/max(v[1],1):.0f} %)" for c, v in sorted(byconf.items())),
      "- evidence Qwen says it used: " + ", ".join(f"{k} {v}" for k, v in ev.most_common()), "",
      "## Confusion of Qwen's class (rows = true class, columns = Qwen), on-part detections", "",
      "| true \\ Qwen | " + " | ".join(NM) + " | right |", "|---|" + "---|"*10]
for i, c in enumerate(NM):
    row = conf_mat[i]; L.append(f"| {c} | " + " | ".join(str(v) for v in row) + f" | {100*row[i]/max(row.sum(),1):.0f} % |")
open(R/f"score_{a.tag}.md", "w").write("\n".join(L)+"\n"); json.dump(dict(changed=changed, best=best, test=Mt, dev=Md, tot=dict(tot)), open(R/f"score_{a.tag}.json", "w"), indent=1)
import shutil; shutil.move(E/"det_metrics_q38.json", R/f"det_metrics_{a.tag}.json"); print("\n".join(L))
