# dv1_E1_XXL_150k_lr2e-4: 42 WACV test images, oracle track, exact pairs

| method | training data | right label | link precision | link F1 | KiCad val right label |
|---|---|---|---|---|---|
| B7 trees, geometry | WACV folds | 82.5 % | 79.0 % | 0.807 |  |
| B8 trees + detector class (③) | WACV folds | 85.7 % | 82.7 % | 0.842 |  |
| dv1_E1_XXL_150k_lr2e-4, KiCad only (zero-shot) | CrossClass_data_v1 train | 88.8 % | 87.2 % | 0.880 | 96.0 % |
