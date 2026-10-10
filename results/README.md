# Qwen3.8-27B decides the class itself (not reading the label) — WACV, 2026-10-05, owner request ("try v2 first")

Question: give the VLM the detected part + its surroundings and the 9 classes, let it decide the class (JSON), replace the detector's
class, compare mAP50. Variant v2 = appearance + context, printed text allowed as evidence (v1 = appearance only, not run yet).
Setup: same 5,573 WACV detection crops as the reader test (`results/qwen_label/`, 3x the part, ≥ 160 px native, magenta outline),
Qwen3.8-27B **FP8** on one L40 (bf16 needs two free GPUs; `tools/qwen_label/serve_qwen38_fp8_1gpu.sh`, --max-num-seqs 64 needed),
thinking off, temperature 0, 6.6 q/s (14 min). Scripts `tools/qwen_label/qwen_class.py`, `qwen_class_score.py` (runs in err9/).
Override policy chosen on the 3 dev images (ML450_Top1, RPI3B_Top, DuetWIFI_Top), reported on the 42 test images.
Caveat: policies Hd_nonRC / Hd_look were added after the first test table (the mass error C → R was visible there); selection
itself is on dev, but it is not a clean held-out result.

| 42 test images | P | R | F1 | macro-F1 | mAP50 | mAP50-95 | dev mAP50 |
|---|---|---|---|---|---|---|---|
| detector alone | 0.814 | 0.661 | 0.729 | 0.604 | 0.541 | 0.359 | 0.611 |
| Qwen class, confidence high (H) | 0.678 | 0.537 | 0.599 | 0.533 | 0.416 | 0.284 | 0.443 |
| H and detector score < 0.7 (Hd) | 0.807 | 0.640 | 0.713 | 0.628 | 0.558 | 0.376 | 0.589 |
| Hd, new class may not be R/C (Hd_nonRC, **chosen**) | 0.820 | 0.659 | 0.730 | 0.632 | **0.569** | 0.381 | 0.609 |
| Hd, new class ∈ {L, D, SW, Q, Y} (Hd_look) | 0.816 | 0.659 | 0.729 | 0.625 | 0.566 | 0.378 | 0.609 |
| for reference: Qwen READS the designator (results/qwen_label) | 0.840 | 0.673 | 0.748 | 0.661 | 0.590 | 0.393 | – |
| for reference: OCR-linking first, else Qwen reader (combo) | 0.848 | 0.679 | 0.754 | 0.677 | 0.607 | 0.405 | – |
| ceiling: all class errors fixed | – | – | 0.804 | 0.836 | 0.817 | 0.546 | – |

Class accuracy on detections sitting on a GT part (IoU ≥ 0.5, all 45 images, 4,871 parts): detector 88.0 %, **Qwen 70.6 %**;
Qwen fixes 343 of the detector's 586 wrong classes but breaks 1,191 right ones. Confidence is uninformative: "high" on 95 % of
answers (73 % right), "medium" 23 % right. Evidence it claims: both 3,817 / appearance 913 / label 141.
Per class (Qwen right): resistor 89 %, switch 92 %, IC 91 %, connector 67 %, oscillator 67 %, inductor 60 %, capacitor 60 %,
transistor 53 %, diode 33 %. Mass errors: capacitor → resistor 628, capacitor → inductor 207, diode → resistor 75, diode → inductor 50.

Reading: letting the VLM judge the class from the picture is clearly worse than letting it read the designator (0.569 vs 0.590
mAP50 with the best policy; raw accuracy 71 % vs the detector's 88 %). It is only useful as a guarded second opinion for the
look-alike classes; as a classifier of small passives it is bad (0402 C vs R at crop scale). Not run: v1 appearance-only.

## Per-part comparison page (2026-10-05): https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK ("Who Gets the Class Right")
`tools/qwen_label/err_compare.py` → `compare.json` (here), 148 sample crops; page built by `build_compare.py` + `compare_template.html`.
v2 of the page (owner: "Qwen judging the class itself is the BASELINE, compare with OCR + link"): 42 test images, 5,585 GT parts;
per method the class of the best-IoU detection (≥ 0.5) or "missed". Methods: detector (reference) · BASELINE = Qwen's class on
EVERY box (policy ALL) · OURS = OCR + learned linking p ≥ 0.5 (`preds_wacvocrp05_test`, 0.570) · OURS final = combo (0.607).
Missed = 1,520 (27 %: R 656, C 494, connectors 288), identical for all. Wrong class: detector 420 · **Qwen baseline 1,201** · OCR+link
371 · combo 294. Baseline's mass errors: C→R 540, C→L 178, D→R 61, connectors 85 wrong; it beats the detector only on inductor
(51→30), switch (27→7), transistor (40→20), IC (38→29), oscillator (14→12). Remaining errors of ours (combo): R→C 64, L→C 24, D→C 23,
Q→IC 21 (unprinted labels / neighbour's label). 151 sample crops, 16–19 per true class, captions with all four verdicts.

### v3 of the page (owner, 2026-10-05): report exactly three results
① detector baseline (F1 0.729 / macro 0.604 / mAP50 0.541, wrong class 420) · ② detector boxes + VLM decides the class on every box
(0.574 / 0.512 / 0.400, wrong 1,201) · ③ detector boxes + OCR linking only, p ≥ 0.5, no Qwen fallback (0.739 / 0.634 / 0.570, wrong 371).
Missed 1,520 identical. Page sections: metrics table, per-class right/wrong/missed, per-class error pairs + 151 sample crops with the
three verdicts. Qwen FP8 server on L40 stopped after this (GPU 0 free).

### v4 (owner 2026-10-05: "the crop is too small for small parts") — two-panel VLM input, tag qc3
`qwen_class.py --view both`: one image, left = 3x close-up (≥ 160 px native), right = surroundings (max(8x, 400 px) native), part
outlined in both; 4.4 q/s (22 min). Qwen's class right 73.4 % (was 70.6 %). ② on every box: F1 0.604 / macro 0.563 / mAP50 **0.450**
(was 0.400), wrong class 1,045 (was 1,201): C→R 463, C→L 205, D→R 55, connector→switch 35. Beats the detector now on R (104→73),
L (51→20), switch (27→10), Q (40→15), Y (14→9); IC level; loses on C (52→678), D (69→114), connectors (25→89). Guarded best 0.588
(Hd_nonRC; dev picked Hd_look 0.580). Page v4 samples = the two-panel images the VLM saw. Server stopped afterwards.
