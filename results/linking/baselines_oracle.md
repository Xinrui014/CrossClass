oracle, 42 test images: parts 5583, designator texts 5753, linkable parts 4591
  B1 nearest text (centre): dev-chosen parameter 2
  B2 nearest text (edge gap): dev-chosen parameter 0.5
  B3 text picks its nearest part: dev-chosen parameter 1
  B4 one-to-one, centre distance: dev-chosen parameter 3
  B5 one-to-one, edge gap: dev-chosen parameter 0.5
  B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style): dev-chosen parameter (0.5, 0.5, 1.5)

### Oracle track — annotated parts + annotated designator texts (42 test images)

| method | right label (linkable parts) | right prefix class | link precision | link F1 | links on parts with no printed label |
|---|---|---|---|---|---|
| B1 nearest text (centre) | 67.2 % | 80.9 % | 60.6 % | 0.638 | 655 |
| B2 nearest text (edge gap) | 64.8 % | 78.6 % | 62.4 % | 0.636 | 514 |
| B3 text picks its nearest part | 61.2 % | 67.7 % | 75.2 % | 0.675 | 259 |
| B4 one-to-one, centre distance | 79.1 % | 84.6 % | 73.7 % | 0.763 | 524 |
| B5 one-to-one, edge gap | 72.2 % | 77.8 % | 76.9 % | 0.745 | 330 |
| B6 one-to-one, gap + orientation + prefix/detector-class (P&ID STA-style) | 81.0 % | 87.1 % | 78.2 % | 0.796 | 443 |
| B7 learned pair scorer, geometry only + one-to-one | 82.5 % | 88.2 % | 79.0 % | 0.807 | 436 |
| B8 learned pair scorer + detector class + one-to-one (current ③ linker) | 85.7 % | 91.7 % | 82.7 % | 0.842 | 374 |
