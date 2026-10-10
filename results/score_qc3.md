# VLM class decider on WACV — tag qc3

Answers: 5573; class changes per policy: {'H': 2080, 'Hd': 1038, 'HM': 2114, 'HMd': 1069, 'ALL': 2114, 'Hd_nonRC': 411, 'Hd_look': 335}; policy chosen on the 3 dev images: **Hd_look** (dev mAP50)

| 42 test images | precision | recall | F1 | macro-F1 | mAP50 | mAP50-95 | dev mAP50 |
|---|---|---|---|---|---|---|---|
| detector alone | 0.814 | 0.661 | 0.729 | 0.604 | 0.541 | 0.359 | 0.611 |
| policy H | 0.685 | 0.543 | 0.606 | 0.567 | 0.451 | 0.305 | 0.434 |
| policy Hd | 0.827 | 0.655 | 0.731 | 0.643 | 0.579 | 0.390 | 0.590 |
| policy HM | 0.683 | 0.541 | 0.604 | 0.563 | 0.450 | 0.305 | 0.433 |
| policy HMd | 0.825 | 0.654 | 0.730 | 0.639 | 0.579 | 0.390 | 0.588 |
| policy ALL | 0.683 | 0.541 | 0.604 | 0.563 | 0.450 | 0.305 | 0.433 |
| policy Hd_nonRC | 0.824 | 0.661 | 0.734 | 0.640 | 0.588 | 0.394 | 0.610 |
| policy Hd_look (chosen) | 0.818 | 0.659 | 0.730 | 0.632 | 0.580 | 0.388 | 0.612 |

## Class accuracy on detections that sit on a GT part (IoU ≥ 0.5, all 45 images)

- detections on a part: 4871; detector class right 4285 (88.0 %); Qwen class right 3575 (73.4 %); Qwen unsure 0
- detector wrong 586: Qwen right on 408; detector right but Qwen wrong: 1118
- Qwen accuracy by its confidence: high 3573/4856 (74 %), medium 2/15 (13 %)
- evidence Qwen says it used: both 4376, appearance 490, label 5

## Confusion of Qwen's class (rows = true class, columns = Qwen), on-part detections

| true \ Qwen | Resistor | Capacitor | Inductor | Connector | Diode | Switch | Transistor | IC | Oscillator | right |
|---|---|---|---|---|---|---|---|---|---|---|
| Resistor | 1319 | 42 | 31 | 3 | 8 | 2 | 0 | 15 | 0 | 93 % |
| Capacitor | 554 | 1359 | 245 | 0 | 2 | 0 | 4 | 3 | 2 | 63 % |
| Inductor | 23 | 4 | 73 | 1 | 0 | 0 | 0 | 1 | 1 | 71 % |
| Connector | 9 | 2 | 19 | 212 | 13 | 48 | 9 | 5 | 10 | 65 % |
| Diode | 65 | 16 | 43 | 1 | 115 | 5 | 9 | 7 | 2 | 44 % |
| Switch | 0 | 0 | 1 | 7 | 0 | 109 | 0 | 3 | 1 | 90 % |
| Transistor | 6 | 1 | 6 | 0 | 0 | 0 | 46 | 11 | 0 | 66 % |
| IC | 15 | 1 | 14 | 2 | 1 | 2 | 8 | 316 | 0 | 88 % |
| Oscillator | 0 | 0 | 9 | 1 | 0 | 0 | 0 | 3 | 26 | 67 % |
