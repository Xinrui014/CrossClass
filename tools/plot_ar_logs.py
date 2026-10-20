#!/usr/bin/env python3
"""Training curves of the causal AR link head runs (logs/ar_night.log = A frozen backbone, logs/ar_night_b.log = B joint fine-tune):
train losses every 250 steps (matrix head 'link', AR head = the 'tcls' slot) and the evaluation link accuracy every 2,500 steps,
with the Hungarian baseline of the same checkpoint as a reference line. -> page/ar_curves/ar_curves.png
  plot_ar_logs.py   (env with matplotlib)"""
import re, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(os.path.expanduser("~/projects/crossclass")); OUT = ROOT/"page/ar_curves"; OUT.mkdir(parents=True, exist_ok=True)
def parse(f, tag):
    tr, ev = [], []
    for l in open(f):
        if f"[{tag}]" not in l: continue
        m = re.search(r"step (\d+) train_loss ([\d.]+) link ([\d.]+) cls [\d.]+ tcls ([\d.]+)", l)
        if m: tr.append(tuple(float(x) for x in m.groups()))
        m = re.search(r"step (\d+) loss [\d.]+ kvalsub: acc ([\d.]+) prec [\d.]+ clean_dev3: acc ([\d.]+)", l)
        if m: ev.append(tuple(float(x) for x in m.groups()))
    return tr, ev
A = parse(ROOT/"logs/ar_night.log", "v2d2_I_noise05_ar20k"); B = parse(ROOT/"logs/ar_night_b.log", "v2d2_I_noise05_arjoint20k")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for (tr, ev), name, c in ((A, "A: frozen backbone", "#1f77b4"), (B, "B: joint fine-tune", "#d62728")):
    s = [t[0] for t in tr]
    ax[0].plot(s, [t[2] for t in tr], color=c, alpha=.45, lw=1, ls="--", label=f"{name}: matrix head loss")
    ax[0].plot(s, [t[3] for t in tr], color=c, lw=1.6, label=f"{name}: AR head loss")
    ax[1].plot([e[0] for e in ev], [100*e[1] for e in ev], "o-", color=c, label=name)
    ax[2].plot([e[0] for e in ev], [100*e[2] for e in ev], "o-", color=c, label=name)
ax[1].axhline(97.84, color="k", ls=":", label="baseline (Hungarian), kval 97.84")
ax[2].axhline(90.66, color="k", ls=":", label="baseline (Hungarian) 90.66")
ax[0].set_title("Train loss (KiCad, every 250 steps)"); ax[0].set_ylim(0, 0.08)
ax[1].set_title("Link accuracy, KiCad val subset (kvalsub)"); ax[1].set_ylim(97, 98.5)
ax[2].set_title("Link accuracy, WACV clean_dev3 (3 photos)"); ax[2].set_ylim(86, 93)
for a in ax: a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)
ax[1].set_ylabel("%"); ax[2].set_ylabel("%")
fig.suptitle("Causal AR link head on v2d2_I_noise05_100k. Final clean_test41: baseline 92.74 %, A 92.26 %, B 92.50 %", fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"ar_curves.png", dpi=130); print(OUT/"ar_curves.png", len(A[0]), len(B[0]))
