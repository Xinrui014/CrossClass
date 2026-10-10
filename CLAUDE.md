# Project: crossclass — who decides the class of a detected PCB component?

Read `CrossClass.md` first (task, setting, the three results, findings, next steps). Then `STATUS.md`.

## Locations (owner rule: one folder per project, same name on both machines)
- L40 (this folder, the real one): `~/projects/crossclass/`. Desktop: `~/claude-backend/crossclass/` = sshfs mount of it:
  `sshfs l40:projects/crossclass ~/claude-backend/crossclass -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o cache=yes,kernel_cache,compression=no,follow_symlinks`
- `data/` symlinks only (WACV images, gt9_*.json, the detector boxes `preds_wacv_1280.json`, the OCR-link result) → sources in
  `~/projects/gdino_fewshot/{data,err9}`, read-only. `crops/`, `crops_both/` → symlinks to `gdino_fewshot/qwen_label/`.
- `preds/` = the three result files (① detector, ② VLM two-panel [+ old small-crop], ③ OCR link). `answers/` = Qwen raw answers.
  `results/` = scores + README. `page/` = source of the artifact page https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK.
- `tools/`: `qwen_class.py` (VLM decides the class), `qwen_class_score.py`, `err_compare.py`, `build_compare.py`,
  `compare_template.html`, `serve_qwen38_fp8_1gpu.sh`; `analyze9.py` / `det_metrics.py` are copies (2026-10-10) of the shared scorers in
  `gdino_fewshot/err9`. All scripts take the project root from `$CROSSCLASS` (default `~/projects/crossclass`); `qwen_class_score.py`
  and `err_compare.py` chdir into `data/` because the scorers read `gt9_*.json` / `preds_*_1280.json` from the cwd.
- ③ (OCR + learned linking) is produced by `gdino_fewshot/tools/assoc/` — a dependency, not duplicated here.
- Python: `~/venvs/pcbid` for everything here; the VLM server uses the `dsv4` conda env (vLLM) via the serve script, one L40 GPU.

## Operating rules (owner)
- Report exactly three results: ① detector baseline, ② detector boxes + VLM class on every box, ③ detector boxes + OCR link.
- Rules/thresholds are fixed on the 3 dev images (ML450_Top1, RPI3B_Top, DuetWIFI_Top); report on the 42 test images only.
- Small parts must be shown to the VLM with their surroundings (two-panel crop) — owner feedback 2026-10-05.
- L40 GPUs are shared: check `nvidia-smi`, use a free GPU, stop the VLM server when done. Ask before deleting or moving data.
