# Linking — which printed label names which part? (problem setting, draft 2026-10-10)

Owner idea (2026-10-10): the current linker (③) is a fairly arbitrary first attempt; treat it as a baseline and look for a better way to
link printed texts to parts. This file fixes the problem setting, the data, the evaluation and the baselines before any new method.

## 1. Problem in one sentence
On a PCB photo, most parts have a designator printed next to them (R12, C5, U3). Given the parts and the printed texts on one board,
decide for every part which text names it, or that none does. The part's class then follows from the prefix (R → resistor, C → capacitor).

## 2. Formal statement
- Board image $I$. Parts $P=\{p_1,\dots,p_n\}$, $p_i=(b_i,\hat c_i)$: box $b_i$, optionally the detector's class $\hat c_i$ and score.
- Texts $T=\{t_1,\dots,t_m\}$, $t_j=(b_j, s_j)$: box and string, kept only if designator-like (`^[A-Z]{1,4}\d{1,4}[A-Z]?$`).
- Output: a link map $\pi: P \to T\cup\{\varnothing\}$, one-to-one on $T$ (a text names at most one part).
- Ground truth: $\pi^*(p_i)=t_j \iff \mathrm{des}(p_i)=s_j$; $\pi^*(p_i)=\varnothing$ when the part's designator is not printed (or not annotated).
- Class from a link: $c_i=\mathrm{prefix}(s_{\pi(p_i)})$ if linked (fixed prefix table), else $\hat c_i$.
- One-to-one is an assumption. Where it breaks in WACV: 540 texts (8 %) repeat a string on the same image, 359 parts (6 %) share a designator
  with another part (multi-part arrays, group labels, run-on labels such as "C1C2C3" noted by FPIC).

## 3. Two tracks
| track | parts | texts | what it measures |
|---|---|---|---|
| **oracle** (main track for linking research) | annotated part boxes | annotated designator texts (box + string) | linking alone |
| **end to end** (what ③ is) | PCB-Dense v22 detections, conf ≥ 0.25 | PP-OCRv6 strings (native tiles, 0°/90°) | linking under real finding / reading errors → class metrics |

## 4. Metrics
- Linking (both tracks), over **linkable parts** (designator printed and annotated; end to end: also read by OCR somewhere on the board):
  right label % (exact string) · right prefix class % (a neighbour with the same prefix still gives the right class) ·
  link precision (right links / all links made; a link on a part with no printed label counts as wrong) · link F1.
- End to end, class from the link: class-aware P / R / F1, macro-F1, mAP50, mAP50-95 (`tools/det_metrics.py`), plus fixed / broken
  counts (detections on a GT part whose class goes wrong → right / right → wrong).

## 5. Protocol
WACV 45 images (28 boards). 3 dev images (ML450_Top1, RPI3B_Top, DuetWIFI_Top) fix every rule threshold or weight; 42 test images report.
Learned linkers: 3 folds split by board (`gdino_fewshot/wacv_in/folds.json`), so every image is scored by a model that never saw its board.
Rule thresholds are chosen on dev by link F1 (oracle) or by net class fixes (end to end).

## 6. Data — what each open dataset gives for linking
| dataset | what it has for linking | use |
|---|---|---|
| **WACV** (Kuo et al., WACV 2019) | every part named with its designator ("resistor R20"), every silkscreen text annotated with its content ("text R20") → explicit links by string equality. 45 images: 6,399 parts, 6,569 designator texts, 5,282 linkable parts (83 %), 1,222 texts that name no annotated part (test points, unannotated parts) | **main benchmark**, both tracks |
| **FPIC** (Jessurun et al., ACM JETC 2023; 228 annotated images, 97 boards; shared data account, read-only) | SMD polygons with a "Designator" field (the paper shows letter prefixes: C, L, R, U); 23k text boxes; the paper says "silkscreen-to-component association" is recorded but does not document the format. The L40 copy is only the YOLO export (class id = prefix) + our own OCR | today: cross-dataset test of the end-to-end class result (③ already: 0.446 → 0.511 mAP50). To become a linking test: inspect the original CSVs (needs the data-account password) |
| **KiCad open-schematics** (open-source board designs) | each footprint carries its Reference text with exact position → perfect links on thousands of boards, real designer placement styles. 200 WACV-like boards already rendered with exact label boxes (`gdino_fewshot/kicad_render/`) | training source for a linker (geometry from the layout file alone, or renders); never a test set |
| FICS-PCB, PCB-DSLR, Roboflow PCB sets | no silkscreen designator annotations | not usable for linking |

## 7. Baselines (`tools/link_baselines.py`, results in `results/linking/`)
| id | method | idea / source |
|---|---|---|
| B1 | nearest text (centre distance) | the obvious rule |
| B2 | nearest text (edge gap) | the obvious rule, box-to-box |
| B3 | each text picks its nearest part | "a label belongs to the part closest to it" (designer convention) |
| B4 | one-to-one matching, centre distance (Hungarian, "none" allowed) | global assignment |
| B5 | one-to-one matching, edge gap | global assignment |
| B6 | one-to-one, cost = gap + orientation mismatch + prefix-vs-detector-class clash | P&ID symbol–tag association (Romaric, Mbakop, Marinai, DocEng 2026) transferred to PCB |
| B7 | learned pair scorer (24 → 20 geometry features, gradient boosting) + one-to-one | ours, without the detector class |
| B8 | B7 + detector-class / prefix-fit features | **current ③ linker** |
| U1 | perfect linker on the texts OCR read | upper bound for linking alone |

### Oracle track (42 test images; 5,583 parts, 5,753 texts, 4,591 linkable)
| method | right label | right prefix class | link precision | link F1 |
|---|---|---|---|---|
| B1 nearest text (centre) | 67.2 % | 80.9 % | 60.6 % | 0.638 |
| B2 nearest text (edge gap) | 64.8 % | 78.6 % | 62.4 % | 0.636 |
| B3 text picks its nearest part | 61.2 % | 67.7 % | 75.2 % | 0.675 |
| B4 one-to-one, centre | 79.1 % | 84.6 % | 73.7 % | 0.763 |
| B5 one-to-one, edge gap | 72.2 % | 77.8 % | 76.9 % | 0.745 |
| B6 P&ID-style one-to-one | 81.0 % | 87.1 % | 78.2 % | 0.796 |
| B7 learned, geometry only | 82.5 % | 88.2 % | 79.0 % | 0.807 |
| B8 learned + detector class (③) | **85.7 %** | **91.7 %** | **82.7 %** | **0.842** |

### End-to-end track (42 test images; class from the link, unlinked detections keep the detector class)
| method | fixed | broken | net | F1 | macro-F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| detector alone | – | – | – | 0.729 | 0.604 | 0.541 | 0.359 |
| B1 nearest text (centre) | 30 | 153 | −123 | 0.704 | 0.566 | 0.479 | 0.313 |
| B2 nearest text (edge gap) | 193 | 589 | −396 | 0.647 | 0.522 | 0.396 | 0.258 |
| B3 text picks its nearest part | 152 | 312 | −160 | 0.695 | 0.578 | 0.469 | 0.306 |
| B4 one-to-one, centre | 28 | 149 | −121 | 0.705 | 0.567 | 0.483 | 0.316 |
| B5 one-to-one, edge gap | 173 | 407 | −234 | 0.681 | 0.552 | 0.436 | 0.283 |
| B6 P&ID-style one-to-one | 104 | 106 | −2 | 0.728 | 0.615 | 0.535 | 0.355 |
| B7 learned, geometry only (p ≥ 0.5) | 121 | 75 | +46 | 0.736 | 0.630 | 0.559 | 0.371 |
| B8 learned + detector class (③, p ≥ 0.5) | 70 | 22 | +48 | 0.738 | 0.634 | 0.568 | 0.380 |
| U1 perfect linker on OCR texts | 225 | 27 | +198 | 0.761 | 0.678 | **0.611** | 0.408 |
| perfect reading + perfect linking (from `gdino_fewshot/results/assoc` §B) | | | | 0.782 | 0.759 | **0.723** | 0.485 |
B8 here is 0.568 vs 0.570 in CrossClass.md: same method, retrained after the 2026-10-04 designator-parser fix.

## 8. What the baselines say
1. With clean inputs, simple global matching already gets most of the way: one-to-one by distance 79 %, P&ID-style 81 %, our learned
   linker 86 %. The learned linker is a modest step over a published rule, not a different league.
2. End to end, every rule baseline makes the class worse than the detector: rules link every part to something, and when OCR missed the
   part's own label they hand it a neighbour's label. Only the learned linkers, which can say "not sure" (p ≥ 0.5), gain. So in real use
   the key property is knowing when not to link, not only picking the nearest label.
3. Head room. Linking alone (perfect linker, same OCR): 0.568 → 0.611 mAP50 (+0.043). Reading + linking together: up to 0.723 (+0.155).
   A better linker is worth having, but most of the end-to-end room needs reading and linking improved together.
4. Using the detector's class in the linker trades fixes for safety: B7 fixes 121 / breaks 75, B8 fixes 70 / breaks 22 (same as §J in
   `gdino_fewshot/results/assoc/README.md`).

## 9. Decisions for the owner
1. Is one-to-one OK as the main assumption (multi-part and run-on labels handled later)?
2. May the linker use the detector's class? Suggestion: main setting = class-agnostic (boxes + texts only; B7 is its baseline), class-aware
   reported as a variant.
3. Primary linking metric: right label (exact) or right prefix class (what CrossClass needs)?
4. FPIC: OK to read the original annotation CSVs on the data account (password needed) to see whether text–part links exist?
5. Not run yet: a VLM linker baseline (Qwen marks which label belongs to the outlined part) — all four L40 GPUs were in use by other jobs (none from this project)
   on 2026-10-10.

## 10. Owner decisions and error analysis (2026-10-10, later)
- **One-to-one fixed (owner):** a printed label names at most one part and a part has at most one label; parts may take "none",
  texts may stay unused (1,222 texts = 19 % name no annotated part: R/C = skipped parts, P/TP/PP = test points, A/IO/GPIO = pin names;
  1,117 parts = 17 % have no annotated label). The 359 "shared designator" parts are annotation conventions (one part drawn as two boxes,
  e.g. Arty J13, ML365 P7; sub-circuits that restart numbering, e.g. ML450 System Monitor C90/C96) — consistent with one-to-one.
  Annotation noise (e.g. the ML450_Top2 "capacitor unknown" next to a printed C12x) is ignored (owner).
- **Where B7 / B8 go wrong** (`tools/link_errors.py` → `results/linking/errors.json`, oracle, 42 test, 4,591 linkable):
  conflict (own label given to another part) 537 / 445 · other label while own free 148 / 101 · label outside the 12 candidates 93 / 93 ·
  none 24 / 17 → total 802 / 656. B8: 337 of 656 wrong links keep the right prefix; true label rank for errors: nearest 105, 2nd–3rd 259,
  4th–12th 199, >12th 93. 7 boards (Arty, DigitalDiscovery, ML450, ML365, Virtex6, ZTurn, Zybo) = 74 % of errors, 52 % of parts.
  Patterns: competing neighbours, label tables (DigitalDiscovery grids), pin names in brackets "(N14)", stacked labels, annotation typos
  (Arty part "C195" vs printed "C95", ZTurn "C96" vs "C98") where B8 is probably right.
- B7 internals: candidates = 12 nearest texts within 8 part sizes (~49k pairs / fold, ~3.4k true); 20 formula features from the two boxes
  + string length (no CNN, no pixels); sklearn HistGradientBoosting 400 trees; Hungarian with none at p = 0.005.
- Page (English, v2): https://claude.ai/artifact/GgsGXi8FHqvZ1CS4LzTeiw (`page/linking.html`).
- **Metric decision (owner, 2026-10-10):** a link counts as right only if it is the exact label (a same-prefix neighbour, e.g. C45 for
  C44, is wrong). Primary linking metrics = exact pair precision / recall / F1 and exact right-label %; "right prefix class" dropped.
- **Problem definition fixed (owner, 2026-10-10):** input = two object sets per board (parts: boxes; texts: boxes + strings; oracle =
  annotated, end-to-end = detector + OCR); output = format A: a score for every (part, text) and every (part, none), decoded by Hungarian
  into a one-to-one list of exact pairs. Image crops per object = later extension; whole-image input out of scope.
- **Model (owner, 2026-10-10):** transformer encoder trained from scratch (no LLM; an LLM only as a possible baseline). Owner: start
  large and first check that it can overfit a tiny set before any full training. Code `tools/linkformer.py` (tokens = parts + texts;
  E1 content = type, log w, log h, orientation; E2 + prefix and digits; positions in median-part-size units via 2D RoPE only; head =
  q·k score matrix + none column; loss = per-part softmax with all same-string texts as targets; Hungarian decoding), data
  `tools/linkdata.py` → `linkdata/{wacv,kicad199}.json` (git-ignored). Sizes L = 384 × 8 (14.7 M), XL = 512 × 12 (38.8 M).
  Overfit runs O1 (3 dev boards) and O2 (fold-0 training boards, 30 images) × {L, XL} × {E1, E2} on L40 CPU (all GPUs busy),
  logs `logs/linkformer/`, results `results/linkformer/`.
- **Overfit test done (cluster job 196808):** both L and XL fit the training boards to the one-to-one ceiling (3 boards 98.0–98.6 % vs
  98.7 %; 30 boards up to 98.4 % = ceiling). Held-out fold-0 test: E2 69.5 % (L) / 68.2 % (XL), E1 38–42 %, vs B7 91.4 % / B8 93.3 % on the
  same boards → memorises 30 boards, does not generalise; strings +27–31 points; XL ≯ L. `results/linkformer/README.md`.
- **KiCad link set built (2026-10-10, CPU data prep):** `tools/kicad_select.py` (pcbprior env; 2,745 WACV-like candidates from
  boards.parquet → 1,498 with ≥ 70 % visible top references and ≥ 60 % SMD pads → `kicad_boards/`, 1.8 GB, git-ignored) and
  `tools/kicad_links.py` (system python3 + pcbnew 9.0.7; part box = pads + Fab shapes; classless footprints such as TP / H / FID are not
  parts; labels ≥ 50 % inside a footprint body dropped as hidden; bottom sides mirrored; side kept with ≥ 10 links; OLIMEX/*, Duet3D/*,
  BehrensG/ARTY_Z7 excluded for possible overlap with WACV OlimexRed / DuetWIFI / Arty) → `linkdata/kicad_all.json`:
  1,412 projects, 1,842 sides (1,351 top, 491 bottom), 126,611 parts, 116,628 texts, **110,335 links** (median 45 per side; WACV 81 per
  image). Earlier estimate of ~250k links was too high: the visible-label check halves the boards and KiCad sides are smaller.
  Example drawing `results/linking/kicad_side_example.jpg`.
- **Full runs done (job 197413, `results/linkformer/README.md`):** transformer E1 trained on KiCad layouts only = **86.9 %** exact right
  label on the 42 test images (precision 85.0 %, F1 0.860) vs B8 85.7 % / B7 82.5 %; dev3 ranks configs the same way. KiCad + WACV
  fine-tune E1 85.2 %; E2 (strings) 80.3 % zero-shot / 83.1 % fine-tuned; WACV-only 61.2 %. Data curve still rising. Below 90 % target.
- **Full open-schematics scan (2026-10-11, owner: use much more data):** `tools/kicad_scan.py` parsed every .kicad_pcb of every row
  (394,104 board files, 60,298 projects) → 40,900 unique layouts (place_hash), 25,753 with ≥ 10 visible silkscreen designators on a side
  (13,484 projects, 12,756 repo names), ~1.6 M visible designators. Density median 0.9 parts/cm² (WACV ~5.9); 5,200 layouts in WACV's
  2.1–11.2 range. Earlier 1,412-project set came from render-oriented filters (density, R+C, IC, SMD, 3D models) and one board per project.
  Stage 2 rerun on all 25,755 saved layouts with crash-safe workers and per-side metadata (group = repo name, density, SMD share,
  version); EXCLUDE list widened to 15 project groups that might recreate WACV boards → `linkdata/kicad_full.json`.
- **Full KiCad link set built (`linkdata/kicad_full.json`, 187 MB, git-ignored):** 25,755 layouts in → 21,578 ok, 3,609 with no side
  ≥ 10 links, 456 unreadable by KiCad 9, 108 excluded, 4 crashed/hung (skipped). **24,772 sides (19,412 top, 5,360 bottom), 11,239 repo
  groups, 1,400,683 parts, 1,231,322 texts, 1,142,279 links** (10.4× kicad_all). Links per side median 30 (WACV 81 per image).
  Density of the part extent: < 1 parts/cm² 13,018 sides (517k links) · 1–2.1: 7,038 (374k) · 2.1–5.9: 4,255 (231k) · 5.9–11.2: 397
  (18.6k) · > 11.2: 64 (2.5k) → WACV-like 2.1–11.2: 4,652 sides, 249k links. SMD ≥ 60 %: 14,531 sides (783k links). Group-hash split
  (md5(repo name) % 20 == 0): 577 val groups, 1,369 sides, 63,137 links. Incident: two boards hung KiCad, one worker reached ~200 GB RSS
  on the shared L40 → killed; workers now capped at 24 GB address space and 5 min per board.
- **dv1_E1_XL_150k done (job 198314, `results/linkformer/README.md`):** XL geometry-only transformer on CrossClass_data_v1 = **89.3 %**
  exact on the 42 test images (precision 87.8 %, F1 0.885) with no WACV training; WACV fine-tune 87.8 %, 1:1 mix 88.3 % (dev3 would pick
  mix). Validation curves flat from ~80k of 150k steps → converged. 0.7 below the 90 % target.
- **dv1_E1_XXL_150k (job 199198):** 86.6 % on the 42 test images (XL 89.3 %); KiCad val 95.1 % (XL 96.2 %), higher training loss → with
  the XL recipe the larger model trains worse; size is not the lever under this recipe.
- **XXL lr sweep:** lr 3e-4 / 2e-4 / 1e-4 → 86.6 / 88.8 / **89.9 %** (42 test; F1 0.861 / 0.880 / **0.891**; KiCad val 95.1 / 96.0 / 96.4).
  The XXL drop was the learning rate; tuned XXL beats XL (89.3 %) by 0.6. New best: dv1_E1_XXL_150k_lr1e-4. XL at lower lr untested.
- **WACV-clean via design files (2026-10-13):** `tools/design_check.py` aligns a board's design (Allegro centroid, Eagle .brd, IPC-D-356
  or a vector assembly PDF via `tools/pdf_to_centroid.py`) to the photo and reads each part's true designator; `tools/design_check_all.sh`
  reruns all 13 checked images (results `results/linking/design_check/`, git-ignored). "DuetWIFI" in WACV is really a Smoothieboard.
  Review page (owner) keeps every change in an append-only history; labels may be fixed (misread text / part designator), boxes
  merged (same component), texts marked "not a designator".
- **Owner decision 2026-10-13: drop ML365_Bottom** (test image, 665 parts, 43 review candidates) — no design file exists (only the
  2004 user guide UG066 with schematics) and the dense board is too hard to clean by eye. The cleaned test set (wacv_clean_v1) leaves
  it out; numbers on the old 42-image set stay reported for comparison. ML365_Top: same board, no design file — open.
- **WACV-clean v1 built (2026-10-13, after the owner finished all 139 review candidates):** `tools/wacv_clean.py` →
  `linkdata/CrossClass_data_v1/wacv_clean_v1.json` (44 images, 41 test; review db export `review_db_2026-10-13.json`, both git-ignored).
  Truth = an explicit `link` (text index) per part instead of string equality; authority wacv string match < design file (only where its
  designator is printed in the photo — later-revision renames keep the printed link) < Claude's answers < the owner's answers; parts the
  owner could not read, and one open conflict (ML365_Top #155), are marked `ign` and left out (7 parts). Changes: 53 boxes merged into
  their part, 6 deleted, 9 drawn, 2 redrawn, 7 text boxes added, 10 text strings fixed, 23 links moved by design files, 3 fixes on the
  ML450 clock card that is in both Top1 and Top2 (Top2 had C11/C12 swapped; Top1 lacked the C22 box). Build log
  `results/linking/wacv_clean_v1_build.json`.
- **Rescored (`tools/wacv_clean_score.py`, `results/linking/wacv_clean_v1_scores.{json,md}`), exact right label on linkable parts:**
  | set | linkable | ceiling | B7 | B8 | XL lr3e-4 | XXL lr3e-4 | XXL lr2e-4 | XXL lr1e-4 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | old42 (published labels) | 4,591 | 98.32 % | 82.5 % | 85.7 % | 89.4 % | 86.5 % | 88.8 % | 89.7 % (473 wrong) |
  | old41 (without ML365_Bottom) | 4,179 | 98.16 % | 81.8 % | 85.2 % | 89.0 % | 86.2 % | 88.2 % | 89.3 % (449 wrong) |
  | clean41 | 4,185 | 100 % | 84.0 % | 86.9 % | 90.5 % | 87.8 % | 90.0 % | **90.9 % (379 wrong)** |
  Cleaning moves every linker up 1.2–1.7 points on the same 41 images and keeps the ranking; best precision 88.3 → 92.2 %. With the
  ceiling at 100 %, 98 % means ≤ 84 wrong of 4,185 — the remaining ~295-part gap is the linker, not the labels. Transformer numbers are
  fp32 on the L40 and differ by 4–8 parts from the bf16 numbers logged on the cluster (e.g. XXL lr1e-4 473 vs 465 wrong on old42).
