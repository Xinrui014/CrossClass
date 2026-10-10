# CrossClass — who should decide the class of a detected PCB component?

Owner-defined task (2026-10-05). Boxes come from our detector; the question is only **which source gives the class**:
the detector itself, a VLM looking at the part, or printed designators found by OCR and linked to the part.
Everything below is on the same boxes, so misses are identical across the three and only *wrong class* differs.

## Setting
- Data: WACV PCB test set, 45 images; 3 dev images (ML450_Top1, RPI3B_Top, DuetWIFI_Top) fix every rule, **42 test images** report.
  5,585 annotated parts, 9 classes (Resistor, Capacitor, Inductor, Connector, Diode, Switch, Transistor, IC, Oscillator).
- Boxes: PCB-Dense v22 YOLOv8l-P2 at 1280 px, conf ≥ 0.25 → 5,573 detections (`err9/preds_wacv_1280.json`).
- Metrics: class-aware P / R / F1 (IoU 0.5), macro-F1 over 9 classes, mAP50, mAP50-95 (`err9/det_metrics.py`).
  Per-part view: each GT part gets the class of its best-IoU detection (≥ 0.5) under each method, or "missed".

## The three results
| 42 test images | P | R | F1 | macro-F1 | mAP50 | mAP50-95 | wrong class | missed |
|---|---|---|---|---|---|---|---|---|
| ① detector baseline | 0.814 | 0.661 | 0.729 | 0.604 | 0.541 | 0.359 | 420 | 1,520 |
| ② detector boxes + VLM decides the class (Qwen3.8-27B, two-panel crop) | 0.683 | 0.541 | 0.604 | 0.563 | **0.450** | 0.305 | **1,045** | 1,520 |
| (② with the first, small crop: 3× the part, ≥ 160 px native) | 0.649 | 0.514 | 0.574 | 0.512 | 0.400 | 0.274 | 1,201 | 1,520 |
| ③ detector boxes + OCR linking (PP-OCRv6 + learned linker, p ≥ 0.5; no VLM fallback) | 0.825 | 0.669 | 0.739 | 0.634 | **0.570** | 0.380 | **371** | 1,520 |

Per true class, wrong-class counts (① → ② → ③):
Resistor 104 → 73 → 96 · Capacitor 52 → **678** → 47 · Inductor 51 → 20 → 46 · Connector 25 → 89 → 20 · Diode 69 → 114 → 60 ·
Switch 27 → 10 → 19 · Transistor 40 → 15 → 35 · IC 38 → 37 → 34 · Oscillator 14 → 9 → 14.
②'s mass errors: capacitor → resistor 463, capacitor → inductor 205, diode → resistor 55, connector → switch 35.
③'s remaining errors: resistor → capacitor 80, diode → capacitor 28, inductor → capacitor 27, transistor → IC 26.

## How each method works
- **② VLM**: Qwen3.8-27B (FP8, vLLM, one L40 GPU, thinking off, temperature 0, 4.4 queries/s). Input = one image with two panels:
  left the 3× close-up (≥ 160 px native), right the surroundings (max(8× part, 400 px) native, ≤ 1600), the part outlined in magenta
  in both. Prompt: judge the type from package, size, colour, pins and surroundings; printed text may be used; choose one of the
  9 classes or "unsure"; JSON {class, confidence, evidence}. Its answer is used for every box (no gate). Qwen says "high" on 95 %
  of answers, so its confidence cannot gate it.
- **③ OCR linking**: PP-OCRv6 (native-resolution tiles, 0°/90°) finds designator texts; a gradient-boosted pair scorer (24 geometry /
  compatibility features, trained on WACV's own text–part links, 3 board folds) + per-board Hungarian assignment links a text to
  each part; class = prefix table (R→resistor, C→capacitor, L/FB→inductor, D/LED→diode, Q→transistor, U/IC→IC, SW/S→switch,
  J/P/CN→connector, Y/X→oscillator); parts without a linked text keep the detector's class.

## Findings
1. Reading the printed designator beats looking at the part: ③ 0.570 > ① 0.541 > ② 0.450. ③ is at least as good as the detector in
   every class.
2. The VLM's failure is one class: 0402 capacitors read as resistors (463) or inductors (205). Capacitors are 43 % of parts, so this
   one confusion sinks the whole number. With surroundings visible, the VLM is *better* than the detector on resistors, inductors,
   switches, transistors and oscillators, level on ICs, worse on capacitors, diodes and pin headers.
3. Context matters for a VLM: the two-panel input raised its class accuracy 70.6 % → 73.4 % and mAP50 0.400 → 0.450.
4. Misses (27 % of parts: 656 resistors, 494 capacitors, 288 connectors) are untouched by any class method; the ceiling with all class
   errors fixed is mAP50 0.817, so class is a smaller lever than localisation.
5. A gated hybrid (VLM only when the detector is unsure and never into R/C) reaches 0.588, slightly above ③ — but the gate was designed
   after seeing the first test table, so it is not a clean result (see results/qwen_class/README.md caveat).

## Files
- Page: https://claude.ai/artifact/T1MkAEUMAfBxDsgipdhyHK ("Who Gets the Class Right", v4: three results, per-class counts,
  error pairs with counts, 149 sample two-panel crops captioned with the three verdicts).
- Numbers: `results/qwen_class/` (README.md, score_qc2 / qc3, det_metrics_qc2 / qc3, answers_qc2 / qc3.jsonl, compare.json,
  page_metrics.json); OCR linking: `results/assoc/README.md` §B; detector: `results/err9_crossdomain.md`.
- Code: `tools/qwen_label/qwen_class.py` (VLM decider, `--view both`), `qwen_class_score.py`, `err_compare.py`, `build_compare.py`,
  `compare_template.html`, `serve_qwen38_fp8_1gpu.sh` (needs `--max-num-seqs 64` on one GPU); linking `tools/assoc/`.
- L40: `~/projects/gdino_fewshot/qwen_label/{crops,crops_both}/` (5,573 crops each), `err9/preds_wacvqc3ALL42_1280.json` (②),
  `err9/preds_wacvocrp05_test_1280.json` (③), `err9/compare_page4/`.

## Not done / possible next steps (owner to choose)
- V1: VLM told to ignore printed text (isolates "eyes" from "reading"). Not run.
- Why capacitors → resistors: colour/marking cue test (e.g. tell the VLM "beige/brown body without a printed value = capacitor").
- A second VLM (Qwen3-VL-8B) or a larger one, same protocol.
- FPIC transfer of ② and ③ (③ already: 0.511 mAP50, `results/assoc/README.md` §C).
- Clean re-selection of the gated hybrid on dev only.
