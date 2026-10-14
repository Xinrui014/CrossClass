# Linker versions — timeline

The part ↔ designator linker (task and metric: `LINKING.md`). Each version is a git tag or branch; data and checkpoints are not in git
and live on the L40 and the cluster (paths below). Report on WACV-clean v1, 41 test images ("clean41"), from 2026-10-13 on.

| version | git | dates | model | training data | result (exact right label) |
|---|---|---|---|---|---|
| v0 baselines | `main` (commits up to `faab23e`) | 2026-10-10 | rules B1–B6, learned pair scorers B7 (geometry) / B8 (+ detector class, = ③) | WACV 3 board folds | old42: B7 82.5 %, B8 85.7 % · clean41: B7 84.0 %, B8 86.9 % |
| **v1** LinkFormer E1 | tag **`linker-v1`** on `main` | 2026-10-10 → 10-14 | whole-board transformer; token = [is_text, ln w, ln h, w ≥ h] (geometry only); 2D RoPE on box centres in median-part units; score = dot product + a none column; per-part softmax loss; Hungarian decoding | CrossClass_data_v1 KiCad train (23,403 sides), no WACV | best `dv1_E1_XXL_150k_lr1e-4`: old42 89.9 % (cluster, bf16) / 89.7 % (fp32 rescore), **clean41 90.9 %** (379 wrong, precision 92.2 %); KiCad val 96.4 % |
| **v2** LightGlue-style matcher | branch **`linker-v2`** | 2026-10-14 → | v1 + matchability per part (σ: has a label) and per text (τ: names a part), P = σ·τ·row-softmax·column-softmax, balanced loss (`--head lg`); attention bias from pair geometry in each box's own size (`--geobias`); OCR-string tokens with OCR-like noise and 30 % dropout (`--tokens E2 --strnoise`) | same KiCad train | experiments, each = v1 + one change, same recipe as v1 (XXL, lr 1e-4, 150k steps × 4 boards): **A** `--head lg` (job 203521, 2×pro6000, 2.4 h) = clean41 86.6 % (default decoding) / **89.7 %** (@sig), KiCad val 93.7 / 95.0 % → worse than v1 (90.9 / 96.4 %); **S** `--tokens E2 --strnoise` (203583, 2.2 h) = clean41 88.8 % (prec 91.1 %), KiCad val **98.0 %** → strings help on KiCad but hurt on photos: S breaks 300 parts v1 had right (174 of them become none), fixes 174; 115 of the new none-errors are labels with prefixes JP / CN / LD / T / BTN / DS / RM / RP / TP / F / DA that never name a part in CrossClass_data_v1 (kicad_links.py PFX keeps only R RN C L FB LED D SW S Q IC U Y X J P footprints as parts) → data fix proposed; **G** `--geobias` (203526, 3.5 h) = clean41 **92.2 %** (prec 92.8 %, F1 0.925), KiCad val 96.6 % → **first change that beats v1** (+1.3 / +0.6) |

**CrossClass_data_v2** (2026-10-14, `tools/kicad_links.py` v2 rule + `tools/make_data_v2.py`): a KiCad footprint is a part when it is a real,
populated component, whatever its prefix (v1 kept only R RN FB L C LED D SW S Q IC U Y X J P footprints); extended prefix → class table,
unknown prefix = class 9 "other". Train 24,152 sides / 1.40 M parts / 1.13 M links (v1 23,403 / 1.32 M / 1.08 M); WACV files unchanged.
Fast training recipe (`--bps 16 --size_ratio 1e9 --tok_budget 16384`): 16 boards per step in padded forwards (same math as one board at
a time), 37,500 steps = the same 600k boards, ~2× faster per board than `--batched`, ~3× than v1's one-board-at-a-time training.
Round 1 on data v2 (intern02 A800 GPUs 6/7, lr 2e-4): v2d2_G (E1 + geobias) vs v2d2_GS (G + strings with noise) — running.

v2 smoke tests (L40, size S, `tools/v2_decode_check.py`): at 15k steps the lg head trails the v1 head (KiCad val subset 88.4 % with its
best decoding vs 90.7 %; clean41 79.7 vs 81.2 %; the gap shrinks with training: −17 points at 2.5k steps, −8.5 at 15k with the default
decoding), while strings help early (lg + geobias + strings at 3k steps ≈ v1 at 3k). So the planned A → B → C chain was changed to three
separate one-change experiments A / G / S, to avoid mixing the lg head's handicap with the strings' gain.

Why v2: error analysis of v1 on clean41 (`page/errors/`, artifact Ffx2RnK2NbxmsKCxnqrAj3): 45 % of errors are on labels printed nearer
another part; none-errors come from the model's own "no label" choice (far labels, big parts, votes split over several texts); 37 % are
a neighbour's label. Design discussion and references: `page/model/` (artifact 41WS71ciNk2ovCsLuhSueY).

## Git workflow
- `main` holds finished versions; each finished version gets a tag (`linker-v1`, …).
- A new version is developed on its own branch (`linker-v2`) and merged into `main` only when the owner accepts it (it must beat the
  previous version on clean41).
- Commits are made and pushed only when the owner asks.

## Where the non-git files are
| what | L40 (`~/projects/crossclass/`) | cluster (`~/xrssd/crossclass/`) |
|---|---|---|
| CrossClass_data_v1: kicad.json, split.json, wacv.json, folds.json (frozen 2026-10-11) | `linkdata/CrossClass_data_v1/` | `linkdata/CrossClass_data_v1/` |
| CrossClass_data_v2 (2026-10-14; also on intern02 `/dev_vepfs/xiyu/crossclass/linkdata/`) | `linkdata/CrossClass_data_v2/` | – |
| WACV-clean v1: wacv_clean_v1.json + review db export review_db_2026-10-13.json (2026-10-13) | `linkdata/CrossClass_data_v1/` | `linkdata/CrossClass_data_v1/` (wacv_clean_v1.json) |
| WACV photos + XML | `~/projects/GLIGEN/igligen/dataset/pcb_wacv_2019/` (assoc_lib `RAW`) | – |
| KiCad source boards (25 GB) | `kicad_scan/boards/` | – |
| vendor design files | `designfiles/` | – |
| v1 checkpoints | `ckpt/` (XL, XXL lr 3e-4 / 2e-4 / 1e-4) | `ckpt/` (all v1 runs) |
| v2 checkpoints | `ckpt/` (v2_S) | `ckpt/` (`v2_A/G/S`); intern02 `/dev_vepfs/xiyu/crossclass/ckpt/` (`v2d2_*`) |
| training logs | – | `logs/linkformer/`; wandb project `crossclass` (group `dv1` = v1, `v2` = v2) |
