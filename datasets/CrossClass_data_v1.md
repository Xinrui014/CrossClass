# CrossClass_data_v1 — part ↔ designator link data (frozen 2026-10-11)

Location: L40 `~/projects/crossclass/linkdata/CrossClass_data_v1/` (git-ignored; copy on the cluster under `~/xrssd/crossclass/linkdata/`).
Task format (LINKING.md §10): per board side, parts = boxes (+ designator, class for analysis only), texts = boxes + designator-like
strings; a link = a part and a text with the same string. Units: KiCad = mm, WACV = pixels (models normalise by median part size).

| file | content |
|---|---|
| `kicad.json` | 24,772 KiCad board sides (19,412 top, 5,360 bottom, mirrored in x), 21,578 layouts, 11,239 repo groups, 1,400,683 parts, 1,231,322 texts, 1,142,279 links. Per side: id, board (project), group (lower-case repo name), side, density (parts / cm² of the part extent), n_parts, n_links, smd_share, version |
| `split.json` | fixed validation groups: md5(group) % 20 == 0 → val 577 groups, 1,369 sides, 63,137 links; train 10,662 groups, 23,403 sides, 1,079,142 links (WACV-like density 2.1–11.2: 4,420 sides, 235,623 links) |
| `wacv.json`, `folds.json` | the 45 annotated WACV images (6,399 parts, 6,569 texts, 5,282 links) and the 3 board folds; 3 dev images ML450_Top1, RPI3B_Top, DuetWIFI_Top; report on the other 42 |
| `stats.json`, `kicad_build_stats.json` | counts above; build status (21,578 ok, 3,609 no side with ≥ 10 links, 456 unreadable by KiCad 9, 108 excluded, 4 crashed / hung) |

How it was built (code in `tools/`):
1. `kicad_scan.py` (pcbprior env): every .kicad_pcb of every open-schematics row (394,104 files, 60,298 projects) → 40,900 unique layouts
   (place_hash) → 25,753 with ≥ 10 visible silkscreen designators on a side, written once per layout.
2. `kicad_links.py --run 24 --list kicad_scan/selected.json --in kicad_scan/boards` (system python3 + KiCad 9.0.7 pcbnew): part box = pads +
   Fab shapes; only footprints whose prefix maps to the 9 classes are parts (TP / H / FID … stay as texts naming no part); a label ≥ 50 %
   inside any footprint body is dropped (hidden under a component); sides with ≥ 10 links kept; workers capped at 24 GB and 5 min / board.
3. Excluded projects that might recreate WACV boards: OLIMEX/*, Duet3D/*, BehrensG/ARTY_Z7, boobalakrishnan08/Arty100T,
   antmicro/arty-expansion-board, Arduino-Mega recreations / shields (5 projects), gggmmm/pcb-spartan-3, Fabeltranm/SPARTAN6…,
   jonthomasson/SpartanMini, tcfh2003/Virtex-6…, viktor-nikolov/MicroZed….
Class ids: 0 resistor, 1 capacitor, 2 inductor, 3 connector, 4 diode, 5 switch, 6 transistor, 7 IC, 8 oscillator
(counts 435,616 / 392,056 / 23,838 / 140,373 / 181,850 / 47,473 / 43,947 / 125,687 / 9,843).
Known limits: mostly sparse hobby boards (density median 0.94 vs WACV ~5.9); KiCad boxes are exact pads + Fab outlines, WACV boxes are
hand-drawn on photos; labels hidden by other means (faded, covered by stickers) are not modelled.
