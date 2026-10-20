# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).
Session logs: `snapshot/<date>.md` (raw), `memory/<date>.md` (distilled; **latest 2026-10-20**). Linker versions, flags, git tags /
branches, data and checkpoint locations: `VERSIONS.md`. Review base page: Pipeline Map https://claude.ai/artifact/GhKrhD6Fvv1kAwJk9yinSN
(source `page/pipeline/index.html`, v13; update in place, mark what changed).

## Done (details in the files named)
- ① detector 0.541 / ② VLM two-panel 0.450 / ③ OCR link 0.570 mAP50 on WACV 42 test — `CrossClass.md`, `results/`.
- Linking sub-task: `LINKING.md`; data CrossClass_data_v1 / v2; WACV-clean v1 (41 test = clean41); linker v1 (tag `linker-v1`, 90.9 %
  clean41) and v2 experiments (branch `linker-v2`) — `VERSIONS.md`, `results/linkformer/README.md`.
- 2026-10-15 .. 10-17 end-to-end track (`memory/2026-10-17.md`): noise boxes, top-3 output format, box score, soft cascade C,
  perfect-OCR ladder, error attribution, LLM per-board prefix -> class (blind), proposal B with detector scores + dropout.
  **Current best: `v2d2_B1_det_fp_drop03_100k`, (class head + C) / 2 = mAP50 0.724 (top-3, 42 test) vs detector 0.602;
  APr 0.485 -> 0.705; balanced acc 67 -> 83 %; clean41 link 93.7 % / class head 93.7 %.** Headline table: `memory/2026-10-17.md`.
- 2026-10-17 .. 10-20 (`memory/2026-10-20.md`): 21PCB second test set (28 photos with ICs, owner-checked GT
  `data/21pcb/final_v1/gt9_21pcb.json`); report tables one class per box (detector / C / C + LLM / B-head: WACV 0.541 / 0.608 /
  0.626 / 0.641, 21PCB 0.633 / 0.820 / 0.836 / 0.793; page https://claude.ai/artifact/CW5UbU18EEK9MeiiW4MBhf); C error analysis;
  B-head over-trusts the detector; causal AR link head tried (no gain).

## Live
- Nothing training. L40: GPU 0 not to be used; ask before L40 GPU jobs. intern02 GPUs 4-7 for CrossClass (0-3: do not use); guard
  holders stopped. VLM server off.
- L40 http server serving `page/c_errors/` (all 166 C errors on 21PCB) at 127.0.0.1:8766 (`ssh -L 8766:localhost:8766 l40`; 8765 is an
  older review server, not ours). Stop it when the owner no longer needs it.
- Checkpoints: intern02 runs of 10-14..17 in L40 `ckpt/intern02/` (deleted from intern02); AR runs `ckpt/v2d2_I_noise05_ar20k.pt`,
  `ckpt/v2d2_I_noise05_arjoint20k.pt`.
- Git: linker-v2 committed and pushed 2026-10-20 (4 commits after 9b954a0: 21PCB labelling pipeline, 21PCB evaluation,
  AR head + VLM test, notes and pages). Kept as a record but NOT used: tools/pcb21_pdf_silk.py (PDF silkscreen reading,
  unreliable) and tools/pcb21_page_data.py + page/label21/ (web labelling page, replaced by labelme). Ask the owner before
  every commit / push.

## Open / waiting on the owner
- GT fix round: owner edits ONLY `~/Downloads/pcb21_labelme/pcb21_final/` (= L40 `data/21pcb/labelme_final_v1/`, MD5.txt); check
  the 19 "possible GT error" cards in the C-error browser + Card20 T1/T2. Then housekeeper-proart uploads -> rebuild GT
  (pcb21_build_final.py from that folder; text boxes from owner_final2/pcb21_batch1) -> rerun results_two_sets.py + browser.
- Confirm the batch-1 text-box strings (owner drew some without a string).
- VLM joint decision after linking (`tools/pcb21_vlm_class.py`): paused for discussion; 532 / 2,799 21PCB answers kept.
- Older items: (a) B (head + C) + LLM table; (b) seed 1 of drop03; (c) KiCad renders with 3D models; (d) B v2 fixed-offset crops;
  B-head retrain with confidently-wrong simulated detector scores, or a per-box detector weight from link / reading confidence.
- 98 % link target: messy labels; worst boards. WACV-clean provisional choices to confirm (see LINKING.md §10).
- (parked) SAM-style zero-shot finding. Owner ideas (2026-10-15): Sinkhorn with dustbin, offset regression, embedding tags,
  designator prediction, graph edges; whole-board image cross-attention.

## Next (owner to choose)
- Linker v3, structured decoding (owner idea 2026-10-19): (a) causal AR head TRIED 2026-10-20, no gain (memory/2026-10-20.md);
  (b) iterative refinement
  (Mask-Predict / MaskGIT style: predict all, keep confident, re-mask the rest, 3-5 rounds; train by revealing random GT links) =
  recommended first; (c) discrete diffusion over the assignment (DIFUSCO-like), heaviest. Expected gain: dense-row / conflict
  link errors (21PCB C: <= 30 of 166; larger on WACV). Also a paper novelty for the "structure-aware matching" story.
- Paper direction: CAD -> photo transfer (KiCad-only training reaches the numbers above) + structure / design knowledge (noise boxes,
  footprint-type targets, LLM world knowledge for board naming) + WACV-clean benchmark; long-tail metrics as the headline.
- To rerun ② (VLM): `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` -> `tools/qwen_class.py --tag <new> --view both` ->
  `tools/qwen_class_score.py --tag <new>` -> `tools/err_compare.py --qwen wacv<new>ALL42` -> `tools/build_compare.py page` -> republish.
