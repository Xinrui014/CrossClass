# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).
Session logs: `snapshot/<date>.md` (raw), `memory/<date>.md` (distilled; **latest 2026-10-17**). Linker versions, flags, git tags /
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

## Live
- Nothing running. intern02 GPUs 4-7 free for CrossClass (6 / 7 hold gpu_guard holders); GPUs 0-3: owner unsure, other users on them —
  do not use. L40: GPU 0 not to be used. VLM server off.
- Checkpoints of the 2026-10-15..17 runs are only on intern02 `/dev_vepfs/xiyu/crossclass/ckpt/` (move to the L40 when the link is fast).
- Code, notes and result JSONs committed on `linker-v2` as 6833335 (2026-10-17, owner OK); not pushed yet (ask before pushing).

## Open / waiting on the owner
- Next experiments to propose: (a) B (head + C) combined with the LLM per-board prefix mapping; (b) seed 1 of the drop03 run;
  (c) KiCad renders WITH 3D models -> part-crop features with exact labels (feasibility check: 3D library on the L40? render 3 boards);
  (d) proposal B v2: text candidates = OCR boxes + fixed-offset crops around each part.
- Metric grouping for the paper: frequent / common / rare by the detector's training frequency (v22): R C J / IC D / L SW Q Y (owner OK'd
  "report plain mAP plus metrics with clear gains").
- (pending) 21PCB as a third test set: high-quality photos, has a BOM, no labels (owner may hand-label).
- (parked) SAM-style zero-shot component finding, label-guided.
- 98 % link target: 12 % "messy" labels cap accuracy (75 % on them); look at the worst boards (DigitalDiscovery 54 / 191 wrong).
- Confirm provisional WACV-clean choices: M_Spartan6Redux_Top_52 kept as answered (text "Q1"; silkscreen says U5); ML365_Top #155 left
  out (its text 107 = #156's R180); 5 "看不清" parts left out; Claude's ML450 card fixes (Top2 C11/C12 swap, Top1 C22 box).
- Owner ideas still open (2026-10-15): other link representations (Sinkhorn with dustbin, offset regression, embedding tags, designator
  prediction, graph edges); whole-board image cross-attention.

## Next (owner to choose)
- Paper direction: CAD -> photo transfer (KiCad-only training reaches the numbers above) + structure / design knowledge (noise boxes,
  footprint-type targets, LLM world knowledge for board naming) + WACV-clean benchmark; long-tail metrics as the headline.
- To rerun ② (VLM): `GPUS=<free> tools/serve_qwen38_fp8_1gpu.sh` -> `tools/qwen_class.py --tag <new> --view both` ->
  `tools/qwen_class_score.py --tag <new>` -> `tools/err_compare.py --qwen wacv<new>ALL42` -> `tools/build_compare.py page` -> republish.
