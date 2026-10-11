# dv1_E1_XL_150k: 42 WACV test images, oracle track, exact pairs

| method | training data | right label | link precision | link F1 | KiCad val right label |
|---|---|---|---|---|---|
| B7 trees, geometry | WACV folds | 82.5 % | 79.0 % | 0.807 |  |
| B8 trees + detector class (③) | WACV folds | 85.7 % | 82.7 % | 0.842 |  |
| dv1_E1_XL_150k, KiCad only (zero-shot) | CrossClass_data_v1 train | 89.3 % | 87.8 % | 0.885 | 96.2 % |
| dv1_E1_XL_150k + WACV fine-tune | data_v1 + WACV folds | 87.8 % | 88.0 % | 0.879 |  |
| dv1_E1_XL_150k + WACV/KiCad 1:1 fine-tune | data_v1 + WACV folds | 88.3 % | 88.1 % | 0.882 |  |
