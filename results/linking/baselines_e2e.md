  B1 nearest text (centre): dev-chosen parameter 0.5
  B2 nearest text (edge gap): dev-chosen parameter 0.5
  B3 text picks its nearest part: dev-chosen parameter 0.5
  B4 one-to-one, centre distance: dev-chosen parameter 0.5
  B5 one-to-one, edge gap: dev-chosen parameter 0.5
  B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style): dev-chosen parameter (0, 1, 1.5)

### End-to-end track — detector boxes + PP-OCRv6 texts; link scores over detections on a GT part (42 test images)

| method | right label (linkable parts) | right prefix class | link precision | link F1 | links on parts with no printed label |
|---|---|---|---|---|---|
| B1 nearest text (centre) | 9.2 % | 10.2 % | 29.2 % | 0.139 | 265 |
| B2 nearest text (edge gap) | 62.1 % | 73.2 % | 40.9 % | 0.493 | 1398 |
| B3 text picks its nearest part | 54.3 % | 60.4 % | 51.7 % | 0.530 | 728 |
| B4 one-to-one, centre distance | 8.7 % | 9.8 % | 29.1 % | 0.134 | 253 |
| B5 one-to-one, edge gap | 64.6 % | 71.6 % | 50.1 % | 0.564 | 1002 |
| B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style) | 73.1 % | 81.7 % | 52.7 % | 0.613 | 1117 |
| B7 learned pair scorer, geometry only + one-to-one | 52.3 % | 54.0 % | 75.3 % | 0.617 | 305 |
| B8 learned pair scorer + detector class + one-to-one (current ③ linker) | 64.8 % | 68.4 % | 81.9 % | 0.723 | 254 |
| U1 perfect linker on the OCR texts (upper bound) | 100.0 % | 98.1 % | 100.0 % | 1.000 | 0 |

Class changes on the 42 test images (detections on a GT part): fixed = wrong -> right, broken = right -> wrong

| method | fixed | broken | net |
|---|---|---|---|
| B1 nearest text (centre) | 30 | 153 | -123 |
| B2 nearest text (edge gap) | 193 | 589 | -396 |
| B3 text picks its nearest part | 152 | 312 | -160 |
| B4 one-to-one, centre distance | 28 | 149 | -121 |
| B5 one-to-one, edge gap | 173 | 407 | -234 |
| B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style) | 104 | 106 | -2 |
| B7 learned pair scorer, geometry only + one-to-one | 121 | 75 | +46 |
| B8 learned pair scorer + detector class + one-to-one (current ③ linker) | 70 | 22 | +48 |
| U1 perfect linker on the OCR texts (upper bound) | 225 | 27 | +198 |


Class from the linked prefix (unlinked detections keep the detector class), 42 test images:

set                                        precision recall   F1    macroF1  mAP50  mAP50-95
detector                                      0.814  0.661  0.729   0.604  0.541   0.359
B1                                            0.787  0.637  0.704   0.566  0.479   0.313
B2                                            0.729  0.581  0.647   0.522  0.396   0.258
B3                                            0.779  0.628  0.695   0.578  0.469   0.306
B4                                            0.787  0.638  0.705   0.567  0.483   0.316
B5                                            0.763  0.615  0.681   0.552  0.436   0.283
B6                                            0.814  0.659  0.728   0.615  0.535   0.355
B7                                            0.824  0.665  0.736   0.630  0.559   0.371
B8                                            0.824  0.668  0.738   0.634  0.568   0.380
U1                                            0.855  0.685  0.761   0.678  0.611   0.408

