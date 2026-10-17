import json, sys, numpy as np
sys.path.insert(0, "../tools"); from analyze9 import iou
G = json.load(open("gt9_wacv42.json"))
names = ["wacv42", "wacvlf_des0542", "wacvcls_des_headall42", "wacvcls_des_headlinked42", "wacvcls_des_rule0542", "wacvcls_gt_headall42", "wacvcls_gt_rule0542"]
for nm in names:
    pr = json.load(open(f"preds_{nm}_1280.json")); f = r = 0
    for st, g in G.items():
        gb = np.array([b[:4] for b in g["boxes"] if b[4] >= 0]).reshape(-1, 4); gc = [b[4] for b in g["boxes"] if b[4] >= 0]
        d = np.array([p for p in pr.get(st, []) if p[4] >= 0.25]).reshape(-1, 6)
        if not len(gb) or not len(d): continue
        I = iou(gb, d[:, :4])
        for k in range(len(gb)):
            j = int(np.argmax(I[k]))
            if I[k, j] >= 0.5: f += 1; r += int(d[j, 5]) == gc[k]
    print(f"{nm:28s} class right on found parts: {r}/{f} = {100*r/f:.1f} %")
