# CrossClass — who decides the class of a detected PCB component?

Boxes come from our detector (PCB-Dense v22); three sources give the class: ① the detector itself, ② a VLM (Qwen3.8-27B) looking at a
two-panel crop, ③ the printed designator found by OCR and linked to the part. WACV 42 test images, 5,585 parts: mAP50 0.541 / 0.450 /
0.570; wrong class 420 / 1,045 / 371. Spec, numbers and next steps: `CrossClass.md`; state: `STATUS.md`; layout and rules: `CLAUDE.md`.

Not in the repo: `data/` (WACV images and GT, symlinked on the L40), `crops*/` and `page/img/` (crops of WACV photos), `logs/`.
