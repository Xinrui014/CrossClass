# Whole-board transformer linker — overfit test (2026-10-10, cluster job 196808, 1× L40S, 15 min)

Owner: start large, first check the model can overfit a tiny set. `tools/linkformer.py`, job `tools/cluster/linkformer_overfit.sbatch`
(repo clone at cluster `~/xrssd/crossclass`, commit 8d93a22). Oracle track, exact-pair accuracy (a same-prefix neighbour counts as wrong).
L = 384 × 8 (14.7 M params), XL = 512 × 12 (38.8 M). E1 = geometry tokens, E2 = + prefix and digits. No augmentation, no regularisation.

| run | train boards | steps | train acc | one-to-one ceiling | held-out fold-0 test (15 images) |
|---|---|---|---|---|---|
| L E1 | 3 dev | 1,500 | 98.0 % | 98.7 % | – |
| L E2 | 3 dev | 1,500 | 98.6 % | 98.7 % | – |
| XL E1 | 3 dev | 1,500 | 98.1 % | 98.7 % | – |
| XL E2 | 3 dev | 1,500 | 98.6 % | 98.7 % | – |
| L E1 | fold-0 train (30) | 4,000 | 97.2 % | 98.4 % | 38.3 % (best 42.4 %) |
| L E2 | fold-0 train (30) | 4,000 | 98.4 % | 98.4 % | 69.5 % |
| XL E1 | fold-0 train (30) | 4,000 | 95.0 % | 98.4 % | 42.1 % (best 45.1 %) |
| XL E2 | fold-0 train (30) | 4,000 | 97.6 % | 98.4 % | 68.2 % (best 69.2 %) |
| B7 trees, geometry (same 30 boards) | | | | | 91.4 % |
| B8 trees + detector class (same 30 boards) | | | | | 93.3 % |

One-to-one ceiling = share of linkable parts that can get their own label when each printed label may be used once (parts sharing a
designator with fewer printed copies cannot). Readings: (1) the pipeline works and both sizes overfit to the ceiling; (2) trained on 30
boards the transformer memorises them and generalises far worse than the hand-feature trees (69.5 % vs 91–93 % on the same held-out
boards); (3) strings help a lot (E1 → E2 +27–31 points); (4) XL is not better than L here and converges slower. Next = much more board
variety (KiCad) + augmentation before any comparison on the 42 test images.

# Full runs (2026-10-10, cluster job 197413, 1× L40S, ~3 h; wandb https://wandb.ai/xinruiworkstation/crossclass, group linkformer_full_197413)
L model (14.7 M), dropout 0.1, wd 0.01, cosine lr, 4 boards per step, augmentation (90° rotations + mirror, box jitter 10 %, 15 % of
labels dropped, 5 % fake labels). Pre-train 20k steps on KiCad (`linkdata/kicad_all.json`, 1,749 train sides; 93 validation sides);
fine-tune 1k steps, lr 1e-4, per WACV fold; WACV-only 4k steps per fold. Settings fixed before the run; each test fold evaluated once.
Checkpoints: cluster `~/xrssd/crossclass/ckpt/P_{full_E2,full_E1,1000_E2,300_E2}.pt`. Report: `full_report.md` (`tools/linkformer_report.py`).

| method (oracle track, exact pairs) | training data | 42 test right label | precision | F1 | 3 dev right label | KiCad val |
|---|---|---|---|---|---|---|
| B7 trees, geometry | WACV folds | 82.5 % | 79.0 % | 0.807 | | |
| B8 trees + detector class (③) | WACV folds | 85.7 % | 82.7 % | 0.842 | | |
| transformer E1 (geometry), KiCad only | 1,749 KiCad sides | **86.9 %** | 85.0 % | **0.860** | 86.4 % | 94.6 % |
| transformer E1, KiCad + WACV fine-tune | KiCad + WACV folds | 85.2 % | 86.4 % | 0.858 | 82.9 % | |
| transformer E2 (+ strings), KiCad only | 1,749 KiCad sides | 80.3 % | 84.7 % | 0.825 | 83.5 % | 95.8 % |
| transformer E2, KiCad only | 1,000 projects | 77.9 % | 83.3 % | 0.805 | 83.1 % | 95.3 % |
| transformer E2, KiCad only | 300 projects | 66.3 % | 80.3 % | 0.726 | 72.6 % | 91.7 % |
| transformer E2, KiCad + WACV fine-tune | KiCad + WACV folds | 83.1 % | 86.9 % | 0.849 | 81.6 % | |
| transformer E2, WACV only | WACV folds | 61.2 % | 81.4 % | 0.698 | 63.2 % | |

Readings: (1) trained on KiCad layouts only, geometry-only tokens beat both trees on the photos (86.9 % vs B8 85.7 % / B7 82.5 %, higher
precision too), with no WACV training and no detector class; the 3 dev images rank the configurations the same way, so the choice of E1
is not a test-set pick. (2) KiCad data is what makes the transformer work (WACV only 61.2 %); the data curve is still rising
(66.3 → 77.9 → 80.3 % for 300 → 1,000 → all), so more boards should help. (3) Strings help on KiCad validation (95.8 vs 94.6 %) but hurt
on photos (80.3 vs 86.9 %): they transfer badly from layouts to photos (cause not yet checked). (4) The fixed WACV fine-tune lowered
right-label for E1 (86.9 → 85.2 %) while raising precision; it helped E2 (80.3 → 83.1 %). (5) Still below the 90 % target.
