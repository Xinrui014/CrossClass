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
