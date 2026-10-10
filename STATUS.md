# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).

## Done
- ① detector 0.541 / ② VLM two-panel 0.450 (small crop 0.400) / ③ OCR link 0.570 mAP50 on WACV 42 test — `CrossClass.md`, `results/`,
  page https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK (v4). Raw answers `answers/`, result boxes `preds/`.

## Next (owner to choose, see CrossClass.md "Not done")
- V1 (VLM told to ignore printed text) · capacitor-cue test · Qwen3-VL-8B · FPIC transfer · clean gated hybrid on dev only.
- VLM server is OFF. To rerun ②: `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` → `tools/qwen_class.py --tag <new> --view both` →
  `tools/qwen_class_score.py --tag <new>` → `tools/err_compare.py --qwen wacv<new>ALL42` → `tools/build_compare.py page` → republish.
