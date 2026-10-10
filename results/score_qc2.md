# VLM class decider on WACV — tag qc2

Answers: 5573; class changes per policy: {'H': 1828, 'Hd': 917, 'HM': 2146, 'HMd': 1134, 'ALL': 2149, 'Hd_nonRC': 379, 'Hd_look': 288}; policy chosen on the 3 dev images: **Hd_nonRC** (dev mAP50)

| 42 test images | precision | recall | F1 | macro-F1 | mAP50 | mAP50-95 | dev mAP50 |
|---|---|---|---|---|---|---|---|
| detector alone | 0.814 | 0.661 | 0.729 | 0.604 | 0.541 | 0.359 | 0.611 |
| policy H | 0.678 | 0.537 | 0.599 | 0.533 | 0.416 | 0.284 | 0.443 |
| policy Hd | 0.807 | 0.640 | 0.713 | 0.628 | 0.558 | 0.376 | 0.589 |
| policy HM | 0.649 | 0.514 | 0.574 | 0.512 | 0.400 | 0.274 | 0.428 |
| policy HMd | 0.794 | 0.629 | 0.702 | 0.609 | 0.546 | 0.369 | 0.578 |
| policy ALL | 0.649 | 0.514 | 0.574 | 0.512 | 0.400 | 0.274 | 0.428 |
| policy Hd_nonRC (chosen) | 0.820 | 0.659 | 0.730 | 0.632 | 0.569 | 0.381 | 0.609 |
| policy Hd_look | 0.816 | 0.659 | 0.729 | 0.625 | 0.566 | 0.378 | 0.609 |

## Class accuracy on detections that sit on a GT part (IoU ≥ 0.5, all 45 images)

- detections on a part: 4871; detector class right 4285 (88.0 %); Qwen class right 3437 (70.6 %); Qwen unsure 0
- detector wrong 586: Qwen right on 343; detector right but Qwen wrong: 1191
- Qwen accuracy by its confidence: high 3383/4637 (73 %), medium 54/234 (23 %)
- evidence Qwen says it used: both 3817, appearance 913, label 141

## Confusion of Qwen's class (rows = true class, columns = Qwen), on-part detections

| true \ Qwen | Resistor | Capacitor | Inductor | Connector | Diode | Switch | Transistor | IC | Oscillator | right |
|---|---|---|---|---|---|---|---|---|---|---|
| Resistor | 1261 | 63 | 25 | 1 | 27 | 2 | 8 | 32 | 1 | 89 % |
| Capacitor | 628 | 1309 | 207 | 0 | 3 | 0 | 2 | 13 | 7 | 60 % |
| Inductor | 21 | 10 | 62 | 0 | 2 | 0 | 6 | 1 | 1 | 60 % |
| Connector | 4 | 8 | 17 | 218 | 19 | 28 | 19 | 7 | 7 | 67 % |
| Diode | 75 | 13 | 50 | 1 | 87 | 3 | 20 | 13 | 1 | 33 % |
| Switch | 0 | 0 | 1 | 5 | 0 | 111 | 0 | 3 | 1 | 92 % |
| Transistor | 5 | 0 | 5 | 0 | 1 | 1 | 37 | 20 | 1 | 53 % |
| IC | 4 | 1 | 6 | 3 | 0 | 1 | 15 | 326 | 3 | 91 % |
| Oscillator | 2 | 1 | 5 | 0 | 0 | 0 | 0 | 5 | 26 | 67 % |
