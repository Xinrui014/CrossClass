# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).
Session logs: `snapshot/<date>.md` (raw), `memory/<date>.md` (distilled; latest 2026-10-14). Linker versions, git tags / branches and
where data and checkpoints live: `VERSIONS.md`.

## Done
- ① detector 0.541 / ② VLM two-panel 0.450 (small crop 0.400) / ③ OCR link 0.570 mAP50 on WACV 42 test — `CrossClass.md`, `results/`,
  page https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK (v4). Raw answers `answers/`, result boxes `preds/`.
- Linking sub-task (2026-10-10 →): problem setting, baselines, data, decisions in `LINKING.md`; prior work `results/literature_crossclass.md`;
  design / error page https://claude.ai/artifact/GgsGXi8FHqvZ1CS4LzTeiw (`page/linking.html`).
  Data **CrossClass_data_v1** (`linkdata/CrossClass_data_v1/`, card `datasets/CrossClass_data_v1.md`): 24,772 KiCad sides, 1.14 M links.
  **Best linker: `dv1_E1_XXL_150k_lr1e-4` = 89.9 % exact right label on the 42 test images** (precision 88.4 %, F1 0.891; KiCad val 96.4 %),
  vs XL lr3e-4 89.3 %, B8 (③) 85.7 %, B7 82.5 %; on WACV-clean v1 (41 test) 90.9 %. All runs: `results/linkformer/README.md`; wandb project `crossclass` (xinruiworkstation).
  Checkpoints: cluster `~/xrssd/crossclass/ckpt/`, L40 `ckpt/`. This is **v1** = git tag `linker-v1` (`VERSIONS.md`).
- WACV-clean v1 (2026-10-13): review 139 / 139, `tools/wacv_clean.py` → `linkdata/CrossClass_data_v1/wacv_clean_v1.json` (44 images, 41 test),
  rescored with `tools/wacv_clean_score.py` (LINKING.md §10). Error analysis pages: https://claude.ai/artifact/Ffx2RnK2NbxmsKCxnqrAj3 (errors),
  https://claude.ai/artifact/41WS71ciNk2ovCsLuhSueY (model explainer + v2 design).

## In progress
- **v2 on branch `linker-v2`** (`VERSIONS.md`). Cluster results (data v1, v1 recipe), clean41 vs v1 90.9 %: A `--head lg` 89.7 % (@sig),
  S `--tokens E2 --strnoise` 88.8 % (data gap: JP/CN/LD/… never parts in data v1), **G `--geobias` 92.2 %** (prec 92.8) = best so far.
- **CrossClass_data_v2** built (prefix-independent KiCad parts; `tools/kicad_links.py` v2 + `tools/make_data_v2.py`).
- Round 1 done (data v2, fast recipe, lr 2e-4): v2d2_G clean41 90.3 %, **v2d2_GS 91.6 %** (strings help now); both < v2_G 92.2 %.
- Training step optimized for the A800 (tag `linker-v2-a800-optimize`, 1.4–1.5× faster, same math; `VERSIONS.md`).
- **Overnight queue on intern02 GPUs 6/7** (owner OK 2026-10-15 00:30 SGT, until ~11:00–12:00 SGT): `/dev_vepfs/xiyu/crossclass/queue.txt`
  run by `queue_runner.sh 6|7` (log `logs/queue_log.txt`); `gpu_guard.sh` keeps a small holder on a GPU only while it has no job.
  Phase A = one change each against v2d2_GS: v1d_G (data v1, fast recipe: data or recipe behind G's drop?), lr 1e-4 / 3e-4, seed 1,
  `--label_drop 0.05`, `--none_w 0.5`, `--fake 0.15`, `--ign_messy`, `--size XL`, `--bps 32`. Choose winners on dev sets only (KiCad val +
  clean_dev3, against the seed-to-seed difference); clean41 is reported, not used to choose. Phase B = the best combination at 1.2M
  boards, 2 seeds (queued fallback: GS at 1.2M boards). Morning, no training: none bias and test-time augmentation on the checkpoints.

- Overnight results so far (clean41 acc / prec): `v1d_G_b16_lr2e-4` (G, data v1, fast recipe) 90.7 / 92.2 (KiCad val 96.4, clean_dev3
  91.0) vs v2_G (data v1, old recipe) 92.2 / 92.8 → the fast recipe (4× fewer updates) explains most of G's drop (−1.6), data v2 ≈ −0.4
  (v2d2_G 90.3). `v2d2_GS_b16_lr1e-4` 91.8 / 92.0 (KiCad val 97.7, clean_dev3 91.5) ≈ lr 2e-4 (91.6 / 91.0). wandb replays: `replay-<tag>`.
- **100k updates (04:20 SGT): `v2d2_GS_100k` clean41 93.1 % (prec 92.7, F1 0.929; KiCad val 98.1, clean_dev3 92.8) = new best;
  `v2d2_G_100k` 91.6 % (prec 91.9; KiCad val 96.2, clean_dev3 92.7).** Longer training: GS +1.5, G +1.3 over 37.5k; S on top of G +1.5.
- **2×2 ablation, data v2, 100k updates (06:55 SGT)** — clean41 acc / prec (KiCad val, clean_dev3): base 91.5 / 92.0 (96.4, 91.4);
  G 91.6 / 91.9 (96.2, 92.7); S 92.7 / 92.4 (98.0, 91.7); GS **93.1 / 92.7** (98.1, 92.8). Per-image bootstrap on clean41: S − base +1.2
  [+0.1, +2.3], GS − G +1.5 [+0.4, +2.5], GS − base +1.6 [+0.6, +2.5]; G − base +0.1 [−0.7, +1.0], GS − S +0.4 [−0.4, +1.2]. So strings
  are the clear gain; G's +1.3 on data v1 / old recipe mostly disappears with longer training (G may speed up learning more than it raises
  the end point; clean_dev3 still +1.2 with G). Seeds would settle G (owner declined seeds for now).
- Queue change 06:55: the 37.5k screens strrel / glayers (running since 06:50) finish, the rest move to 100k: strrel 100k, local 8 100k,
  glayers 100k, GS 150k, GS 100k + fake 0.15.
- 37.5k screens vs GS 37.5k (91.6 / 91.8 %): `--strrel` 92.3 % (prec 92.0; KiCad val 97.7, clean_dev3 89.9) → +0.6, kept (strrel 100k
  running 07:54 → ≈ 10:50); `--glayers` 91.6 % (prec 91.7; KiCad val 97.7, clean_dev3 90.7), 40 % slower → glayers 100k removed (08:15).
  local 8 100k running on GPU 7 (08:10 → ≈ 11:05); then GS 150k, GS 100k + fake 0.15.
- 100k structure runs vs GS 100k (93.1 / 92.7): `--strrel` 93.2 / 92.4 (KiCad val 98.1, clean_dev3 91.4), per-image CI [−0.8, +1.1] →
  no clear gain (the 37.5k +0.6 shrank); `--local 8` 92.5 / 91.9 (KiCad val 97.9, clean_dev3 89.5), CI [−1.5, +0.5] → local attention
  is worse, whole-board attention helps (dev3 −3.3). Pattern: early gains (G, strrel) shrink with longer training → test changes at full
  length; differences < ~1 point need seeds. Running from ~10:45 SGT: GS 150k (GPU 6, ≈ 15:00), GS 100k + fake 0.15 (GPU 7, ≈ 13:50).

## Owner ideas to discuss (2026-10-15 ~01:40 SGT, "record these")
- Other ways to represent the part ↔ label link than the 0/1 assignment table (now: score matrix + none column, per-part softmax, Hungarian).
  Candidates: optimal transport / Sinkhorn with a dustbin (SuperGlue); per-part offset to its label (regression, then nearest text);
  associative-embedding tags (pull / push); predict each part's designator string; edge classification on a graph.
- Put the whole board image into the model by cross-attention (tokens attend to image patch features). Needs training images (KiCad
  renders exist for only ~200 boards).
- Is whole-board self-attention needed, or only a local area? Test queued: `--local 8` (each token sees tokens within 8 median part
  sizes; KiCad link distances p99 = 22 units, WACV p99 = 8.9).
- Owner (01:55): I may change the rest of tonight's queue at any time from each run's result.
- Owner (01:50): tonight's aim = no idle GPUs and meaningful experiments (structure, not learning-rate tweaks); WACV-clean decisions
  tomorrow, not tonight; the report page UI was poor → ask what to change before the next page.

## Waiting on the owner
- Confirm provisional WACV-clean choices: M_Spartan6Redux_Top_52 kept as answered (text "Q1"; silkscreen says U5); ML365_Top #155 left out
  (its text 107 = #156's R180); 5 "看不清" parts left out; Claude's ML450 card fixes (Top2 C11/C12 swap, Top1 C22 box).

## Next (owner to choose)
- Paper direction (parked by owner until A/G/S results): CAD→photo transfer + structure-aware matching (silkscreen conventions) as the method novelty for CVPR; WACV-clean as benchmark.
- Linking (report on clean41): combine winners after round 1/2 · v2 D (+ part image patches; needs a decision: rendered KiCad images or WACV-only) · none bias −1.5 at
  inference (+1.4 acc, −1.5 precision) · label dropout 0.15 → 0.05 · ignore messy KiCad labels in training · end-to-end track (detector +
  OCR) with the new linker · XL lr1e-4 · test-time augmentation.
- CrossClass (older list): V1 (VLM told to ignore printed text) · capacitor-cue test · Qwen3-VL-8B · FPIC transfer · clean gated hybrid.
- VLM server is OFF. To rerun ②: `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` → `tools/qwen_class.py --tag <new> --view both` →
  `tools/qwen_class_score.py --tag <new>` → `tools/err_compare.py --qwen wacv<new>ALL42` → `tools/build_compare.py page` → republish.
