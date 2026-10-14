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
