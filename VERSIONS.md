# Linker versions — timeline

The part ↔ designator linker (task and metric: `LINKING.md`). Each version is a git tag or branch; data and checkpoints are not in git
and live on the L40 and the cluster (paths below). Report on WACV-clean v1, 41 test images ("clean41"), from 2026-10-13 on.

| version | git | dates | model | training data | result (exact right label) |
|---|---|---|---|---|---|
| v0 baselines | `main` (commits up to `faab23e`) | 2026-10-10 | rules B1–B6, learned pair scorers B7 (geometry) / B8 (+ detector class, = ③) | WACV 3 board folds | old42: B7 82.5 %, B8 85.7 % · clean41: B7 84.0 %, B8 86.9 % |
| **v1** LinkFormer E1 | tag **`linker-v1`** on `main` | 2026-10-10 → 10-14 | whole-board transformer; token = [is_text, ln w, ln h, w ≥ h] (geometry only); 2D RoPE on box centres in median-part units; score = dot product + a none column; per-part softmax loss; Hungarian decoding | CrossClass_data_v1 KiCad train (23,403 sides), no WACV | best `dv1_E1_XXL_150k_lr1e-4`: old42 89.9 % (cluster, bf16) / 89.7 % (fp32 rescore), **clean41 90.9 %** (379 wrong, precision 92.2 %); KiCad val 96.4 % |
| **v2** LightGlue-style matcher | branch **`linker-v2`** | 2026-10-14 → | v1 + matchability per part (σ: has a label) and per text (τ: names a part), P = σ·τ·row-softmax·column-softmax, balanced loss (`--head lg`); attention bias from pair geometry in each box's own size (`--geobias`); OCR-string tokens with OCR-like noise and 30 % dropout (`--tokens E2 --strnoise`) | same KiCad train | experiments A (`--head lg`), B (+ `--geobias`), C (+ strings): running |

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
| WACV-clean v1: wacv_clean_v1.json + review db export review_db_2026-10-13.json (2026-10-13) | `linkdata/CrossClass_data_v1/` | `linkdata/CrossClass_data_v1/` (wacv_clean_v1.json) |
| WACV photos + XML | `~/projects/GLIGEN/igligen/dataset/pcb_wacv_2019/` (assoc_lib `RAW`) | – |
| KiCad source boards (25 GB) | `kicad_scan/boards/` | – |
| vendor design files | `designfiles/` | – |
| v1 checkpoints | `ckpt/` (XL, XXL lr 3e-4 / 2e-4 / 1e-4) | `ckpt/` (all v1 runs) |
| v2 checkpoints | – | `ckpt/` (`v2_*`) |
| training logs | – | `logs/linkformer/`; wandb project `crossclass` (group `dv1` = v1, `v2` = v2) |
