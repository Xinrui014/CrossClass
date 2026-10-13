# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).
Session logs: `snapshot/<date>.md` (raw), `memory/<date>.md` (distilled; latest 2026-10-13).

## Done
- ① detector 0.541 / ② VLM two-panel 0.450 (small crop 0.400) / ③ OCR link 0.570 mAP50 on WACV 42 test — `CrossClass.md`, `results/`,
  page https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK (v4). Raw answers `answers/`, result boxes `preds/`.
- Linking sub-task (2026-10-10 →): problem setting, baselines, data, decisions in `LINKING.md`; prior work `results/literature_crossclass.md`;
  design / error page https://claude.ai/artifact/GgsGXi8FHqvZ1CS4LzTeiw (`page/linking.html`).
  Data **CrossClass_data_v1** (`linkdata/CrossClass_data_v1/`, card `datasets/CrossClass_data_v1.md`): 24,772 KiCad sides, 1.14 M links.
  **Best linker: `dv1_E1_XXL_150k_lr1e-4` = 89.9 % exact right label on the 42 test images** (precision 88.4 %, F1 0.891; KiCad val 96.4 %),
  vs XL lr3e-4 89.3 %, B8 (③) 85.7 %, B7 82.5 %. All runs: `results/linkformer/README.md`; wandb project `crossclass` (xinruiworkstation).
  Checkpoints: cluster `~/xrssd/crossclass/ckpt/`, L40 `ckpt/` (lr2e-4 run).

## Waiting on the owner
- WACV-clean review: https://claude.ai/artifact/RMi6e4cS2zhayvWGjPX1Hs (212 candidates, db collection "decisions"; 0 decided on 2026-10-13).
  When done: read decisions (ArtifactData list), write `linkdata/CrossClass_data_v1/wacv_clean_v1.json` + change log, rescore B7 / B8 /
  transformers on both label sets. Target 98 % (ceiling on current labels 98.32 %).
- Uncommitted (ask "commit and push"): .gitignore, LINKING.md, results/linkformer/README.md + dv1_E1_XXL_150k*.json/_report.md,
  tools/cluster/linkformer_dv1_xxl.sbatch (sibling guard; cluster copy identical), page/review/review.html, tools/wacv_review_cands.py,
  results/linking/wacv_clean_candidates.json.

## Next (owner to choose)
- Linking: XL lr1e-4 (size effect at matched lr) · XXL lr5e-5 · test-time augmentation · error analysis of the 89.9 % model · text tokens on
  data_v1 · edge-gap attention bias · end-to-end track (detector + OCR) with the new linker.
- CrossClass (older list): V1 (VLM told to ignore printed text) · capacitor-cue test · Qwen3-VL-8B · FPIC transfer · clean gated hybrid.
- VLM server is OFF. To rerun ②: `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` → `tools/qwen_class.py --tag <new> --view both` →
  `tools/qwen_class_score.py --tag <new>` → `tools/err_compare.py --qwen wacv<new>ALL42` → `tools/build_compare.py page` → republish.
