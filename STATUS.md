# crossclass — Status

**Created:** 2026-10-09 (owner: "same as gdino_fewshot, a project folder from L40 sshfs to deskbox"; work done 2026-10-05 in gdino_fewshot).
Session logs: `snapshot/<date>.md` (raw), `memory/<date>.md` (distilled; **latest 2026-10-21**). Linker versions, flags, git tags /
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
- 2026-10-20 GT fix round: owner's check uploaded by housekeeper-proart to `data/21pcb/owner_final3/pcb21_final/` (28 json,
  MD5_laptop.txt; 4 files changed: Card10_T, Card13_T B3 -> IC, Card14_T U3 -> transistor, Card19_T R202 removed). final_v1
  OVERWRITTEN (owner: no v2) with `PCB21_SRC=owner_final3/pcb21_final` final_check + build_final; 3,273 components, B3/U3
  designators cleared. New 21PCB mAP50 detector / C / C + LLM / B-head 0.636 / 0.827 / 0.843 / 0.795 (WACV unchanged);
  C errors 164. Report page still shows the old numbers. Owner's next baseline for edits = owner_final3/pcb21_final.
  Owner decisions: WACV-clean drops ML365_Top (to apply: rebuild clean set, rescore -> clean40); Card4 / Card8 file names stay.

## Live
- Nothing training. L40: GPU 0 not to be used; ask before L40 GPU jobs. intern02 GPUs 4-7 for CrossClass (0-3: do not use); guard
  holders stopped. VLM server off.
- 2026-10-21 link accuracy on 21PCB (oracle, `tools/pcb21_link_eval.py`): v2d2_I_noise05_100k 90.8 % (batch 1 94.1, batch 2 85.7;
  nearest-text 62.2); C errors re-classified with the link GT (164: right label + misread 49 / prefix 30 / naming 18, wrong text 31,
  linker none 12, OCR missed 14, no label 10). Report page updated (v6). Details memory/2026-10-21.md.
- 2026-10-21 class GT SYNCED with the link GT (owner OK): Card14_T re-labelled (UG-855 p.24: orange designators are not printed
  on the board -> none; bottom side has no silkscreen) and uploaded to `linktool_edits_v2/` (merge prefers v2); class GT boxes of
  final_v1 replaced by the link-GT parts (`tools/pcb21_sync_class_gt.py`; previous file `final_v1/gt9_21pcb_prelink.json`):
  3,273 -> 3,278 boxes. New 21PCB mAP50 detector / C / C + LLM / B-head 0.633 / 0.810 / 0.826 / 0.780 (APr 0.394 / 0.701 / 0.729 /
  0.651); drop mostly from 2 new oscillator GT boxes (Card20_B capacitor -> oscillator, Card2_T added) that every method misses.
  WACV unchanged. C errors 164. Card14_T p93 (C22 by rule) lost its text to p142 and stays unchecked = no text.
- 2026-10-21 link GT v1 DONE: owner labelled all 28 photos offline (all 3,278 parts checked); upload
  `data/21pcb/linktool_edits/` (28 json md5-checked) -> `tools/pcb21_link_merge.py` -> `data/21pcb/link_gt_v1.json` (native px;
  3,199 parts linked, 79 without printed text; 6,781 texts, 3,199 linked) + `results/pcb21/link_gt_v1_stats.json`.
  Pre-links kept: match 1,789/1,799, rule 724/979, near 116/206, none 46/278. Owner edits: parts +16/-11, 6 boxes, 2 classes
  (NOT yet in the class GT final_v1); texts +127/-304, 115 boxes, 131 strings. Next: owner decisions (see reply).
- Link labelling (owner 2026-10-20: strings alone are pointless; label part -> text links directly, offline on the laptop
  during a flight). Pre-links `tools/pcb21_link_prep.py` (no LinkFormer): match 1,801 / rule 982 / near 212 / none 278 of 3,273
  parts (28 photos; batch-2 texts = raw OCR boxes). Tool `tools/pcb21_linktool.py` (stdlib server + canvas page; Space accepts the
  view, click part + text = link, 0 = no text, resize / move / new boxes, E string, 1-9 class, Ctrl+Z; busy port -> next).
  Bundle `data/21pcb/linktool/` -> `linktool.zip` (md5 909b9e8a...), on the laptop (proart: unzipped, offline test
  passed; CachyOS, Python 3.14.7, Chrome 154) at `~/Downloads/pcb21_labelme/pcb21_link/`. Source fix after the copy: stem
  validation on /save (no effect on the laptop copy; the page never sends odd stems).
  Later fixes the same afternoon (owner at the airport, copied by scp; proart was offline): S strings on/off, H hide boxes,
  touchpad (two-finger scroll = pan, pinch = zoom), edge / corner resize of the box and its linked text, selection cleared after
  each link + parts first on click + Shift+click for a text inside a part. Last md5 fb89c98e03ccde7fa73358998a091948 (the owner's
  laptop copy may be any of these versions). View rotation was asked for, then dropped (boarding); not implemented. Edits come back as `edits/<stem>.json` -> upload to `data/21pcb/linktool_edits/` -> merge.
  The string review page (port 8767, 239 answers used by the pre-links) is stopped and superseded.
- L40 http server serving `page/c_errors/` (all 164 C errors on 21PCB, GT of 2026-10-20) at 127.0.0.1:8766 (`ssh -L 8766:localhost:8766 l40`; 8765 is an
  older review server, not ours). Stop it when the owner no longer needs it.
- Checkpoints: intern02 runs of 10-14..17 in L40 `ckpt/intern02/` (deleted from intern02); AR runs `ckpt/v2d2_I_noise05_ar20k.pt`,
  `ckpt/v2d2_I_noise05_arjoint20k.pt`.
- Git: linker-v2 committed and pushed 2026-10-20 (4 commits after 9b954a0: 21PCB labelling pipeline, 21PCB evaluation,
  AR head + VLM test, notes and pages). Kept as a record but NOT used: tools/pcb21_pdf_silk.py (PDF silkscreen reading,
  unreliable) and tools/pcb21_page_data.py + page/label21/ (web labelling page, replaced by labelme). Ask the owner before
  every commit / push.

## Open / waiting on the owner
- Link GT on 21PCB (owner 2026-10-20: next step). Batch 1 (12 design photos): 1,925 parts with design designators, 1,587 linked
  by string; 3,037 text boxes = 1,854 with string (213 name no part, 70 repeated), 951 empty, 232 "?TP..". Batch 2 (16 photos):
  1,317 parts, 771 rule designators (unchecked), no text boxes (3,924 raw OCR boxes).
  (superseded the same day by the link tool, see Live) Step 1 (owner 2026-10-20: strings first, links later): `tools/pcb21_text_ocr.py` (PP-OCRv6 rec on each given box, CPU, 2 min)
  -> `data/21pcb/text_check_b1/` (+ .zip, `text_check_b1_ocr.json`): text 1,942 / text_other 608 / text_check 487 (NEW|OLD when the
  reading differs). On the laptop (proart, 12/12 MD5 OK): `~/Downloads/pcb21_labelme/pcb21_text_b1/`; owner checks in
  the review page (port 8767, see Live), then merge the answers into the GT text boxes. Links / batch 2: later.
  Boxes covering two labels: owner types both strings with a space; list `data/21pcb/text_review_split.json` (split later).
- VLM joint decision after linking (`tools/pcb21_vlm_class.py`): paused for discussion; 532 / 2,799 21PCB answers kept.
- Older items: (a) B (head + C) + LLM table; (b) seed 1 of drop03; (c) KiCad renders with 3D models; (d) B v2 fixed-offset crops;
  B-head retrain with confidently-wrong simulated detector scores, or a per-box detector weight from link / reading confidence.
- 98 % link target: messy labels; worst boards. WACV-clean: ML365_Top to be dropped (owner 2026-10-20), see LINKING.md §10.
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
