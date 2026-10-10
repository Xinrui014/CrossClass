# Has anyone done CrossClass before? — web search (2026-10-10)

Question: does any work decide the class of a detected PCB part (a) with a VLM looking at the part, or (b) by reading the printed
designator and linking it to the part — and compare these with the detector's own class?
Earlier search (2026-10-03, text-aware detection): `~/projects/gdino_fewshot/results/literature_text_aware.md`.

## Verdict
Nobody found does the CrossClass comparison on PCBs. Each piece exists somewhere, mostly outside PCB photos.
The new part is the question and the same-box comparison; the linker technique itself (OCR + pair score + Hungarian) is already
published for engineering drawings (P&ID), so it must be cited, not claimed.

## ② VLM decides the class
| work | what it does | how close |
|---|---|---|
| Intelligent Disassembly System for PCB Components, MLLM + multi-agent (Processes 14(2):227, 2026) | YOLOv13 boxes + an MLLM describes parts to plan disassembly; 30 electricity-meter boards, recognition 90–100 % | same shape (detector box → MLLM), but large parts, no per-class score, no 0402 passives |
| PCB-Bench (ICLR 2026, github digailab/PCB-Bench) | benchmark of (M)LLMs on PCB placement/routing; "component identification" is one image subtask | design images, not per-box class on photos |
| PCBA Standard-to-Real Grand Challenge (ACM MM 2026); Task-Aware GRPO, arXiv 2609.21276 | VQA on PCBA images: defect existence, attribute, component / pin counting, standards knowledge; 83.24 overall (2nd) | VLM on PCBA, but questions, not class per detected box |
| Prompt-induced hallucinations in LVLMs, arXiv 2604.21911 | Grounding DINO proposes, Qwen2-VL-7B verifies each crop (crop + image caption as context); confident detections skip the VLM | general version of our gated hybrid |
| MLLMs Know Where to Look (Zhang et al., ViCrop); LookWise arXiv 2603.00171 | VLM accuracy drops causally with small subject size → zoom crops help; forced cropping can mislead on simple cases | explains why the two-panel crop (close-up + surroundings) helped: 0.400 → 0.450 |

## ③ Read the designator and link it to the part
| work | what it does | how close |
|---|---|---|
| US patent 12,639,668 "Circuit board analysis and bill of materials generation" (DARPA HR00112190029) | finds components; OCR on silkscreen text lying outside component boxes = designators; matches them to parts (by distance per search snippets); looks up MPN for a BOM | closest PCB prior art for ③; rule-based as far as visible; no detection score reported. Full text is a scanned PDF, not read in full |
| FPIC (Jessurun et al., ACM JETC 2023, arXiv 2202.08414) | dataset with component↔silkscreen links; says designator text is critical for type, but linking is designer-specific, many-to-one, hard to automate | dataset + motivation, no linker |
| PCB marking framework (arXiv 2307.13105, UF) | proposes designator prefix → component family (R, C, U) as a classification cue | idea / plan stage, no results |
| Kuo et al., WACV 2019 | PCB component detection; silkscreen text listed as future work | – |
| Romaric, Mbakop, Marinai, "Towards Automated P&ID Digitization: Graph-Based OCR Consolidation and Global Symbol-Tag Association", DocEng 2026 (code github dimitri009/STA) | tiled OCR + RF-DETR symbol detector + symbol–tag link as min-cost bipartite matching (geometric / semantic / spatial cues), Hungarian | near-identical structure to our ③ — on drawings, not photos; must cite |
| Retail price-tag ↔ product patents (US 12,333,488; US 12,367,457; US 10,878,372) | detect tags and products, associate them geometrically or by text | same association problem, different domain |
| Mafla et al., WACV 2020 / 2021 | fine-grained classification using scene text linked to image regions (graph) | general template for "class from nearby text" |

## Not found anywhere
- The three sources of class compared on the same detector boxes, with misses held fixed, on PCB photos.
- A learned designator↔part linker on PCB photos evaluated by detection mAP.
- A measured VLM failure mode on tiny passives (0402 capacitor → resistor / inductor).

## Couldn't confirm this time
- "Vision-Based Reconstruction of Electrical Schematics from PCB Photographs" (Electronics 15(14):3125, 2026) from the 10-03 list
  did not come up again; recheck on mdpi.com before citing.
- The patent's exact linking rule (scanned PDF; no OCR tool on the desktop).
