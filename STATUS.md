# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).

## Done
- ① detector 0.541 / ② VLM two-panel 0.450 (small crop 0.400) / ③ OCR link 0.570 mAP50 on WACV 42 test — `CrossClass.md`, `results/`,
  page https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK (v4). Raw answers `answers/`, result boxes `preds/`.

- 2026-10-10 linking task set up (owner: current linker = a baseline; find a better text–part linking): problem setting, data, metrics,
  8 baselines + upper bound in `LINKING.md`; code `tools/link_baselines.py` (`~/venvs/gdino_fewshot`, CPU); results `results/linking/`.
  Oracle: rules 61–81 %, learned ③ linker 85.7 % right label; end to end only learned linkers beat the detector (0.568 mAP50);
  perfect linking on current OCR = 0.611, perfect reading + linking = 0.723. Waiting on owner decisions (LINKING.md §9).
  Literature check "has anyone done this": `results/literature_crossclass.md`.
- 2026-10-10 transformer-linker design page (token ablation E1–E4, 2D RoPE, KiCad K0–K3): https://claude.ai/artifact/GgsGXi8FHqvZ1CS4LzTeiw
  (source `page/linking.html`, figures `tools/link_figs.py`). Waiting on the owner's four choices at the end of the page.

- 2026-10-10 linking transformer: overfit OK (job 196808); full runs (job 197413, wandb crossclass/linkformer_full_197413): geometry-only
  transformer trained on 1,749 KiCad layout sides = 86.9 % exact on the 42 test images (B8 85.7 %, B7 82.5 %); details
  `results/linkformer/README.md`. Data: `linkdata/kicad_all.json` (110k links, `tools/kicad_select.py` + `tools/kicad_links.py`).
  Cluster clone `~/xrssd/crossclass`, checkpoints `~/xrssd/crossclass/ckpt/`.

## Next (owner to choose, see CrossClass.md "Not done")
- V1 (VLM told to ignore printed text) · capacitor-cue test · Qwen3-VL-8B · FPIC transfer · clean gated hybrid on dev only.
- VLM server is OFF. To rerun ②: `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` → `tools/qwen_class.py --tag <new> --view both` →
  `tools/qwen_class_score.py --tag <new>` → `tools/err_compare.py --qwen wacv<new>ALL42` → `tools/build_compare.py page` → republish.
