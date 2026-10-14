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
- **v2 on branch `linker-v2`** (`VERSIONS.md`): three one-change experiments on v1, same recipe as the v1 best (XXL, lr 1e-4, 150k steps):
  done on the cluster: A `--head lg` 89.7 % (@sig) < v1; S `--tokens E2 --strnoise` 88.8 % (prefix gap in data v1 → CrossClass_data_v2 built);
  **G `--geobias` 92.2 %** (prec 92.8) > v1 90.9 %. Round 1 on data v2 + fast recipe (--bps 16, token budget only, lr 2e-4) on intern02 GPUs 6/7: v2d2_G vs v2d2_GS (G + strings);
  auto-started by launch_round1.sh once the data copy finishes;
  sibling guard keeps one copy per run; queue limit 5 jobs per user. Compare on clean41 against v1 90.9 %. Logs `~/xrssd/crossclass/logs/
  linkformer/v2_*.log`, wandb group v2. Smoke tests: lg head trails v1 at small scale; strings promising.

## Waiting on the owner
- Confirm provisional WACV-clean choices: M_Spartan6Redux_Top_52 kept as answered (text "Q1"; silkscreen says U5); ML365_Top #155 left out
  (its text 107 = #156's R180); 5 "看不清" parts left out; Claude's ML450 card fixes (Top2 C11/C12 swap, Top1 C22 box).

## Next (owner to choose)
- Paper direction (parked by owner until A/G/S results): CAD→photo transfer + structure-aware matching (silkscreen conventions) as the method novelty for CVPR; WACV-clean as benchmark.
- Linking (report on clean41): v2 D (+ part image patches; needs a decision: rendered KiCad images or WACV-only) · none bias −1.5 at
  inference (+1.4 acc, −1.5 precision) · label dropout 0.15 → 0.05 · ignore messy KiCad labels in training · end-to-end track (detector +
  OCR) with the new linker · XL lr1e-4 · test-time augmentation.
- CrossClass (older list): V1 (VLM told to ignore printed text) · capacitor-cue test · Qwen3-VL-8B · FPIC transfer · clean gated hybrid.
- VLM server is OFF. To rerun ②: `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` → `tools/qwen_class.py --tag <new> --view both` →
  `tools/qwen_class_score.py --tag <new>` → `tools/err_compare.py --qwen wacv<new>ALL42` → `tools/build_compare.py page` → republish.
