# Transformer linker, full runs: 42 WACV test images, oracle track, exact pairs

| method | training data | right label | link precision | link F1 | KiCad val right label |
|---|---|---|---|---|---|
| B7 trees, geometry | WACV folds | 82.5 % | 79.0 % | 0.807 |  |
| B8 trees + detector class (③) | WACV folds | 85.7 % | 82.7 % | 0.842 |  |
| transformer E2, KiCad only (zero-shot) | all 1,749 KiCad sides | 80.3 % | 84.7 % | 0.825 | 95.8 % |
| transformer E2, KiCad only (zero-shot) | 1,000 KiCad projects | 77.9 % | 83.3 % | 0.805 | 95.3 % |
| transformer E2, KiCad only (zero-shot) | 300 KiCad projects | 66.3 % | 80.3 % | 0.726 | 91.7 % |
| transformer E2, WACV only | WACV folds | 61.2 % | 81.4 % | 0.698 |  |
| transformer E2, KiCad pre-train + WACV fine-tune | KiCad + WACV folds | 83.1 % | 86.9 % | 0.849 |  |
| transformer E1, KiCad only (zero-shot) | all 1,749 KiCad sides | 86.9 % | 85.0 % | 0.860 | 94.6 % |
| transformer E1, KiCad pre-train + WACV fine-tune | KiCad + WACV folds | 85.2 % | 86.4 % | 0.858 |  |
