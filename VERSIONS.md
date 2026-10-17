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
Note: the padded forward gives the same outputs, but the recipe itself differs from v1's (4 → 16 boards per step, lr 1e-4 → 2e-4,
150k → 37.5k steps, warmup 2000 → 500); its effect on accuracy was not checked before round 1.
Round 1 on data v2 (intern02 A800 GPUs 6/7, fast recipe, lr 2e-4, ~1.6 h each): **v2d2_G** (E1 + geobias) clean41 90.3 % (prec 91.7 %),
KiCad val 96.4 %; **v2d2_GS** (G + strings with noise) clean41 **91.6 %** (prec 91.0 %), KiCad val 97.7 %. Strings now help (+1.4,
19 images better / 9 worse: the data v2 fix works), but both are below v2_G (92.2 %): G lost 2.0 points (22 of 27 changed images
worse; losses spread over none-errors and wrong texts, not tied to the new prefixes). Data v2 and the recipe changed together, so the
cause is open (overnight run `v1d_G_b16_lr2e-4` = G, data v1, fast recipe).

**A800 optimization** (2026-10-14, tag `linker-v2-a800-optimize`; nsys on intern02 A800): the step was launch-bound (~5,500 kernels per
step, GPU busy ~65 %, 46 % of GPU time in small casts / elementwise ops, ~113 host-device syncs per step from per-board copies). Changes,
same math: one padded batch per step with the loss computed batched (`dense_batch`, `LinkFormer.train_loss`; per-board loop only for the
lg head), one non-blocking copy per field, no `.item()` sync per step, RoPE cos/sin and the attention bias computed once per forward,
bias made contiguous (a permuted bias silently sends SDPA to the slow math path), tokens padded to multiples of 16. Check against the
original code (16 augmented boards × 3): fp32 loss / gradients agree to ~1e-6; under bf16 both codes have the same error against fp32
(gradients ~1.5 %). Speed (A800, steps 50–199, same boards): G 0.133 → 0.094 s/step, GS 0.140 → 0.095 s/step (1.4–1.5×; a 600k-board
run ≈ 65 min instead of ≈ 100). Options added: `--profile A:B` (NVTX ranges + CUDA profiler range), `--compile` (torch.compile of the
training forward; ~0.085 s/step after compiling but recompiles cost ~2 min each, not used), `--fused_adam`.

**Overnight 2026-10-14/15** (intern02 A800 GPUs 6/7, data v2, 16 boards/step, lr 2e-4; clean41 acc / precision; per-image bootstrap CI):
- G's drop explained: `v1d_G_b16_lr2e-4` (data v1, 37.5k steps) 90.7 % vs v2_G (data v1, 150k steps × 4 boards) 92.2 % → the fast
  recipe's 4× fewer updates cause most of it (−1.6), data v2 ≈ −0.4. lr 1e-4 ≈ 2e-4 (GS 91.8 vs 91.6).
- 2×2 ablation at 100k updates (1.6M boards, ~2.8 h each): base (v1 model) 91.5 / 92.0, G 91.6 / 91.9, S 92.7 / 92.4, **GS 93.1 / 92.7
  (best; KiCad val 98.1)**. Strings: +1.2 [+0.1, +2.3] over base, +1.5 [+0.4, +2.5] over G. Geometry bias: +0.1 [−0.7, +1.0] alone,
  +0.4 [−0.4, +1.2] with strings — its +1.3 under the short recipe mostly disappears with longer training.
- Structure options (one change on GS): `--strrel` (string relations of text pairs in the pair bias) 37.5k +0.6, 100k 93.2 (+0.15,
  [−0.8, +1.1]); `--glayers` (own bias head per layer) 37.5k ±0, 40 % slower; `--local 8` (attention only within 8 part sizes) 100k
  92.5 (−0.6, [−1.5, +0.5]): far labels (> 8 sizes) 53.8 → 15.4 %, dense areas worse; no speed gain (mask on full attention).
- Lesson: early gains shrink with longer training; test changes at full length; differences < ~1 point need seeds.
- Queue tooling: `queue_runner.sh` / `gpu_guard.sh` on intern02; runs uploaded to wandb from their logs (`replay-<tag>`).

**2026-10-15 .. 10-17: text-image tokens, noise boxes, class head (still branch `linker-v2`, code not yet committed)**
(details: `memory/2026-10-17.md`, raw `snapshot/2026-10-17.md`; all 100k steps, 16 boards / step, lr 2e-4, data v2, intern02 A800)
- GS 150k = best oracle link with strings: clean41 93.6 %. **I** (`--tokens E1 --geobias --txtfeat linkdata/kicad_textfeat`): text
  tokens carry a PARSeq text-image feature (688-d, KiCad render crops with photo-like damage) instead of a string: clean41 92.9 %.
- **Noise boxes** `--noise 0.5|1.0` (`tools/kicad_noise.py`: vias, non-label silkscreen print, non-component pads, target none; features
  `text_features.py noise`): end-to-end link acc 52 -> 75 %, clean41 ±0. 200k overfits; `--rankrel`, `--glayers`, `--local`, board
  offset prior: no gain.
- **Class head** `--clshead --clsmode parallel` (mix / pool: no better). **B v1** = I + noise 0.5 + parallel head + `--detscore
  data/det_score_pool_v22test.json` (detector 9-class scores on part tokens, sampled per true class on KiCad) + `--fptype
  linkdata/kicad_fptype` (KiCad class target from the footprint name) + `--det_drop 0.3`: tag `v2d2_B1_det_fp_drop03_100k`, clean41 link
  93.7 % / class head 93.7 %; end to end (42 test, top-3) head 0.701, **(head + soft cascade C) / 2 = 0.724 mAP50** (detector 0.602;
  APr 0.485 -> 0.705). Evaluation: `tools/e2e_infer_b.py` + `tools/score_b.py`; ladder and metrics in `memory/2026-10-17.md`.

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
| text features, noise boxes, footprint types, detector 9-class scores (2026-10-15..17; also on intern02) | `linkdata/kicad_render`, `kicad_textfeat`, `kicad_noise`, `kicad_noisefeat`, `kicad_fptype`; `data/preds_wacv_1280_cls9.json`, `data/det_score_pool_v22test.json` | – |
| v2 checkpoints | `ckpt/` (v2_S, v2_G) | `ckpt/` (`v2_A/G/S`); intern02 `/dev_vepfs/xiyu/crossclass/ckpt/` (`v2d2_*`, overnight runs) |
| training logs | – | `logs/linkformer/`; wandb project `crossclass` (group `dv1` = v1, `v2` = v2) |
