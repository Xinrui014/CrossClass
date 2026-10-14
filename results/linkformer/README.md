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

# dv1_E1_XL_150k (2026-10-11, cluster job 198314, 2× RTX PRO 6000 DDP, 3 h 41 min; wandb crossclass/31xhz6bv, group dv1)
XL (39.3 M), geometry tokens (E1), CrossClass_data_v1 train (23,403 KiCad sides, 1.08 M links), 150k steps × 4 boards, bf16, same
augmentation / regularisation as before. Fine-tunes from the final checkpoint: 1k steps, lr 1e-4, WACV fold only (_ft) or WACV : KiCad 1:1
(_mix). Checkpoints: cluster `~/xrssd/crossclass/ckpt/dv1_E1_XL_150k{,_step25000..125000}.pt`. Report `dv1_E1_XL_150k_report.md`.

| 42 WACV test images, exact pairs | right label | precision | F1 | 3 dev right label |
|---|---|---|---|---|
| B7 trees, geometry | 82.5 % | 79.0 % | 0.807 | |
| B8 trees + detector class (③) | 85.7 % | 82.7 % | 0.842 | |
| P_full_E1 (L, old 1,749-side data, 20k steps) | 86.9 % | 85.0 % | 0.860 | 86.4 % |
| **dv1_E1_XL_150k, KiCad only** | **89.3 %** | 87.8 % | **0.885** | 87.7 % |
| + WACV fine-tune (_ft) | 87.8 % | 88.0 % | 0.879 | 87.6 % |
| + WACV / KiCad 1:1 fine-tune (_mix) | 88.3 % | 88.1 % | 0.882 | 88.7 % |
KiCad validation (1,369 sides, full): 96.2 % right label, 97.2 % precision.

Convergence (every 5k steps; kvalsub = 356 KiCad validation sides): kvalsub 86.1 (5k) → 90.7 (20k) → 94.3 (50k) → 95.5 (80k) →
95.7 (100k) → 96.1 (150k); dev3 72.8 → 82.3 → 86.0 → 87.3 → 87.5 → 87.7; training loss 0.61 → 0.29 → 0.24 → 0.19 → 0.17 → 0.15.
Both validation curves are flat (± noise) from ~80k steps, long before the cosine schedule reaches its minimum → converged; more steps
would not help. Readings: (1) 10× data + XL: 86.9 → 89.3 % on the photos without any WACV training (+3.6 over B8, +6.8 over B7), 0.7 below
the 90 % target. (2) WACV fine-tuning does not help on test (−1.5 / −1.0); dev3 would pick _mix (88.7 %), whose test number is 88.3 % —
the three are within ~1 point and 3 dev images cannot separate them, so the dev-selected (clean) result is 88.3 %, the KiCad-only 89.3 %.

# dv1_E1_XXL_150k (2026-10-12, cluster job 199198, 2× L40S DDP, 4 h 35 min; wandb crossclass/vr50dvcr, group dv1)
Same recipe as dv1_E1_XL_150k, only the size changed: XXL = 768 wide × 16 layers (~114 M). Zero-shot only.
| 42 WACV test images, exact pairs | right label | precision | F1 | 3 dev | KiCad val (full) | final training loss |
|---|---|---|---|---|---|---|
| dv1_E1_XL_150k (39 M) | **89.3 %** | **87.8 %** | **0.885** | 87.7 % | **96.2 %** | 0.15 |
| dv1_E1_XXL_150k (114 M) | 86.6 % | 85.7 % | 0.861 | 87.8 % | 95.1 % | 0.19 |
XXL trailed XL at every checkpoint on KiCad validation (e.g. 50k: 90.2 vs 94.3 %; 150k: 94.6 vs 96.1 %) and kept a higher training loss,
so with the XL settings (lr 3e-4, 4 boards per step) the bigger model trained worse even on KiCad, not just transferred worse. Under this
recipe size is not the lever; a fair size test would need its own learning rate / batch size. The 3 dev images rate the two runs equal
(87.7 vs 87.8 %) while the test differs by 2.7 points: dev3 is too small to choose between models.

# XXL learning-rate sweep (2026-10-12/13; only lr changed vs dv1_E1_XXL_150k)
lr 1e-4: cluster job 200658, 2× RTX 6000 Ada DDP, 4.4 h, wandb n2p8rs83. lr 2e-4: L40 server GPU 3, single GPU (4 boards per step
accumulated), env ~/miniconda3/envs/qwen_edit, 6.0 h, wandb pawomdw6. Zero-shot (KiCad only), 42 WACV test images, exact pairs.
| run | lr | right label | precision | F1 | wrong parts | KiCad val (full) | dev3 | final loss |
|---|---|---|---|---|---|---|---|---|
| dv1_E1_XL_150k (39 M) | 3e-4 | 89.3 % | 87.8 % | 0.885 | 490 | 96.2 % | 87.7 % | 0.15 |
| dv1_E1_XXL_150k (116 M) | 3e-4 | 86.6 % | 85.7 % | 0.861 | 614 | 95.1 % | 87.8 % | 0.19 |
| dv1_E1_XXL_150k_lr2e-4 | 2e-4 | 88.8 % | 87.2 % | 0.880 | 516 | 96.0 % | 88.0 % | 0.16 |
| **dv1_E1_XXL_150k_lr1e-4** | **1e-4** | **89.9 %** | **88.4 %** | **0.891** | **465** | **96.4 %** | **90.6 %** | **0.14** |
Curves (kvalsub / dev3 / loss): lr1e-4 is ahead of every other run from the first evaluation (5k: 90.1 / 77.9 / 0.45 vs XL 86.1 / 72.8 /
0.61) and ends highest on KiCad validation. Lower lr is monotonically better for XXL (3e-4 → 2e-4 → 1e-4: 86.6 → 88.8 → 89.9 %), so
the earlier "bigger is worse" was the learning rate. Tuned, XXL beats XL by 0.6 points on test and 0.2 on KiCad val — a small size gain.
Open: XL was never run at a lower lr, so the size effect at matched tuning is still unmeasured; the trend also leaves room below 1e-4.

# Rescored on WACV-clean v1 (2026-10-13, `tools/wacv_clean_score.py`, L40 GPU 0, fp32)
Same checkpoints, no training. clean41 = 41 test images of `wacv_clean_v1.json` (explicit links, ML365_Bottom dropped, 6 unreadable parts
left out); old41 = the old labels on the same 41 images. Details and B7 / B8: `LINKING.md` §10, `results/linking/wacv_clean_v1_scores.md`.
| run | old42 | old41 | clean41 | clean41 precision | clean41 F1 | clean41 wrong |
|---|---|---|---|---|---|---|
| dv1_E1_XL_150k | 89.4 % | 89.0 % | 90.5 % | 91.6 % | 0.911 | 397 |
| dv1_E1_XXL_150k | 86.5 % | 86.2 % | 87.8 % | 89.6 % | 0.887 | 510 |
| dv1_E1_XXL_150k_lr2e-4 | 88.8 % | 88.2 % | 90.0 % | 91.3 % | 0.906 | 419 |
| **dv1_E1_XXL_150k_lr1e-4** | **89.7 %** | **89.3 %** | **90.9 %** | **92.2 %** | **0.916** | **379** |
fp32 here vs bf16 in the training jobs: old42 counts differ by 4–8 parts from the tables above (465 → 473 wrong for lr1e-4).
`linkformer.py` now reads `link` / `ign` when a board has them (loss and scores); `--data CrossClass_data_v1` adds clean_dev3 / clean_test41 /
clean_all44.
