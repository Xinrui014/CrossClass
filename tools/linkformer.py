#!/usr/bin/env python3
"""Whole-board transformer linker (LINKING.md §10: input = two object sets, output = score matrix + none column -> Hungarian).
Tokens: one per part and one per text. Content: type, log width / height, orientation (E1); + prefix and digits of the string (E2).
Position (box centre, in units of the board's median part size) enters only through 2D rotary embeddings (RoPE) in self-attention.
--pos abs / none (2026-10-23, RoPE control): no RoPE; abs = SuperGlue-style KeypointEncoder MLP (2-32-64-128-256-d) of the box centre,
normalised as (centre - board centre) / (0.7 max(board w, h)) over the union of the board's boxes, added once to every input token.
--pos rope4 (2026-10-23, owner's trunk principle): RoPE over the 4 box corners (x1, y1, x2, y2) instead of the centre; --nosize: the
token geometry input keeps only the token type (no size), so all geometry is in the rope4 rotation.
Head: s_ij = <W_q h_i, W_k h_j> / sqrt(d) for part i, text j; one 'none' score per part. Loss: per-part softmax over texts + none,
target = every text whose string equals the part's designator (or none; WACV-clean boards give an explicit 'link' instead, and
parts marked 'ign' are left out of loss and scores). Decoding: Hungarian on -log p, own none column per part.
Training options: --aug (90-degree rotations + mirror, box jitter, dropped labels, fake labels), --dropout, --accum (boards per
optimizer step), cosine schedule, --save / --init checkpoints (fine-tuning), --workers (background board preparation),
multi-GPU data parallel when launched with torchrun (--accum boards per step are split across the ranks).
v2 options (2026-10-14, LightGlue-style matcher): --head lg = similarity S_ij plus a matchability score per part (sigma_i: has a printed
label) and per text (tau_j: names some part); P_ij = sigma_i tau_j softmax_row(S)_ij softmax_col(S)_ij; loss = -log P on true pairs
+ balanced -log(1 - sigma) / -log(1 - tau) on parts / texts without a partner (LightGlue, ICCV 2023). --geobias = attention bias from
each token pair's box geometry measured in the query box's own size (edge gap, offsets, size ratios; Relation Networks). --strnoise =
OCR-like corruption and 30 % dropout of text strings during training (use with --tokens E2). --batched = the boards of a step in one
padded forward (identical math, ~3x faster); --bps N = N boards per step, sorted by size and cut into padded groups (sizes within
--size_ratio, <= --tok_budget padded tokens), gradients accumulated so every board weighs 1 / N; evaluation then also runs in groups.
Results JSON keeps per-image counts so any image subset
(e.g. the 42 test images) can be scored afterwards (tools/linkformer_report.py). --wandb <project> logs loss, lr, eval and final scores.
Run identity (2026-10-22, tools/run_meta.py): ckpts and the results JSON carry 'meta' (code md5 + git, argv, host, times, full md5 of
every input read, kept / dropped boards); ckpts also tag and step. A run is refused at start when results/linkformer/<tag>.json or the
--save ckpt (or its _step ckpts) exists, unless --overwrite.
Usage: linkformer.py --train dev3 --eval dev3 --size L --tokens E1 --steps 1000 [--device cuda] [--tag name]
  board sets: dev3, test42, all45, fold<k>train, fold<k>test, kicad199, kicadtrain, kicadval, kicad300, kicad1000;
  with --data CrossClass_data_v1: ktrain, kval, kvalsub (~25 % of kval), kdense (train sides with WACV density 2.1-11.2),
  clean_dev3 / clean_test41 / clean_all44 (WACV-clean v1, tools/wacv_clean.py)"""
import argparse, json, math, os, re, shutil, sys, time, hashlib, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); H = Path.home()
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
SIZES = {"S": (128, 4), "M": (256, 6), "L": (384, 8), "XL": (512, 12), "XXL": (768, 16), "3XL": (1024, 24)}
PFX = re.compile(r"^([A-Z]+)(\d+)([A-Z]?)$")
FAKE_PFX = ["TP", "R", "C", "U", "J", "D", "Q", "L", "FB", "SW", "Y", "P"]
STRNOISE = (0.0, 0.0)                                 # (p string dropped, p string corrupted) per text in augment(); set by --strnoise
AUGP = (0.15, 0.05)                                   # (p label dropped, fake labels per text) in augment(); set by --label_drop / --fake
NONE_W = 1.0                                          # loss weight of parts whose target is none (batched dot head); set by --none_w
IGN_MESSY = False                                     # training: no loss on parts whose label is printed nearer another part; --ign_messy
IMG_DIM = 0                                           # > 0: tokens carry an image feature ("img" in each part / text dict); set by --imgfeat
ZLOSS = 0.0                                           # --zloss: weight of the z-loss (logsumexp^2 of the link and class logits)
PART_TERMS = ("link", "cls", "tcls", "link_lab", "link_none", "zloss", "lse")   # model._parts / the log lines
NVTX = False                                          # --profile: NVTX ranges (nv() below) for Nsight Systems; no effect on the math
def nv(name):
    from contextlib import nullcontext
    return torch.cuda.nvtx.range(name) if NVTX else nullcontext()
PIMG_DIM = 0                                          # > 0: part tokens carry a part image feature ("pimg" in each part dict; detector RoI); --pimgfeat
TXT_DIM = 0                                           # > 0: text tokens carry a text-image (PARSeq) feature ("tf" in each text dict); --txtfeat
AUG_ROT = True                                        # augment(): random 90-degree rotation and mirror; off with --aug_norot
DISTRACT, PREFIX_DROP, CLS_W, TCLS_W = 0.0, 0.0, 1.0, 0.5   # --distract, --prefix_drop, --cls_w, --tcls_w
DET_DIM, DET_POOL, DET_DROP = 0, None, 0.0                           # --detscore: part tokens carry detector class scores (9); KiCad: sampled from DET_POOL[class]
AR_ONLY = False                                       # --arhead --freeze_backbone: the loss is the causal head's only
NOISE = 0.0                                           # augment(): noise boxes (vias, print, non-component pads; tools/kicad_noise.py) per label; --noise
NUMS = re.compile(r"^(\d{1,4})([A-Z]?)$")             # a designator whose letters OCR lost ("C12" -> "12")
PFX_CLS = {"R": 0, "RN": 0, "RV": 0, "VR": 0, "RP": 0, "RA": 0, "RM": 0, "TH": 0, "C": 1, "L": 2, "FB": 2,
           "F": 0, "PF": 0, "FU": 0, "NTC": 0, "PTC": 0, "RT": 0, "MOV": 0, "VDR": 0,   # taxonomy_final 2026-10-25: fuse / polyfuse, thermistor, varistor -> R
           "J": 3, "P": 3, "JP": 3, "CN": 3, "CON": 3, "USB": 3, "D": 4, "LED": 4, "LD": 4, "DS": 4, "CR": 4, "RGB": 4,
           "SW": 5, "S": 5, "BTN": 5, "Q": 6, "T": 2, "TR": 2, "U": 7, "IC": 7, "M": 7, "Y": 8, "X": 8, "XL": 8}   # T / TR: transformer -> L; M: module / motor / MOSFET, kept 7 (open); = kicad_links.PFX
IGNORE_PFX_CLS = {"FL", "FIL", "K", "RL", "PS", "BT", "BAT", "BZ", "LS", "SP", "HS", "AE", "ANT", "MK", "MIC"}   # taxonomy_final: no class (EMI filter, relay, power module, battery, buzzer / speaker, heatsink, antenna, microphone); not in PFX
def distractor(rng):
    """A non-designator silkscreen string of the kind OCR also returns: a part value, a net / pin name, a bare number, a word."""
    r = rng.random()
    if r < 0.4: return rng.choice(["1", "2.2", "4.7", "10", "22", "47", "100", "220", "470", "1K", "4K7", "10K", "100K"]) + rng.choice(["", "R", "K", "NF", "UF", "PF", "UH", "M", "V"])
    if r < 0.7: return rng.choice(["GND", "VCC", "VDD", "3V3", "5V", "12V", "VIN", "SDA", "SCL", "TX", "RX", "RST", "EN", "PWR", "+", "-", "IN", "OUT", "A", "B"])
    if r < 0.9: return str(rng.randint(1, 64))
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(rng.randint(2, 6)))
CONFUSE = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6"}

def corrupt(s, rng):
    """One OCR-like error: last character cut off, a look-alike swap, a character lost or a stray one added."""
    if not s: return s
    r = rng.random()
    if r < 0.35 and len(s) >= 3: return s[:-1]
    if r < 0.7:
        idx = [k for k, c in enumerate(s) if c in CONFUSE]
        if idx: k = rng.choice(idx); return s[:k] + CONFUSE[s[k]] + s[k+1:]
    if r < 0.85 and len(s) >= 2: k = rng.randrange(len(s)); return s[:k] + s[k+1:]
    k = rng.randrange(len(s)+1); return s[:k] + rng.choice("0123456789ABCDEFGHJKLMNPRSTUVWXYZ") + s[k:]

def load_sets(data=""):
    if data:   # a frozen dataset folder, e.g. linkdata/CrossClass_data_v1 (kicad.json + split.json + wacv.json + folds.json)
        D = ROOT/"linkdata"/data; wacv = {b["id"]: b for b in json.load(open(D/"wacv.json"))}; folds = json.load(open(D/"folds.json"))
        kic = json.load(open(D/"kicad.json")); val = set(json.load(open(D/"split.json"))["val_groups"])
        sets = {"dev3": [wacv[s] for s in DEV], "all45": list(wacv.values()), "test42": [b for s, b in wacv.items() if s not in DEV]}
        for k, v in folds.items():
            sets[f"fold{k}test"] = [b for b in wacv.values() if b["board"] in v["test_boards"]]
            sets[f"fold{k}train"] = [b for b in wacv.values() if b["board"] not in v["test_boards"]]
        sets["kval"] = [b for b in kic if b["group"] in val]; sets["ktrain"] = [b for b in kic if b["group"] not in val]
        sets["kvalsub"] = [b for b in sets["kval"] if int(hashlib.md5(b["id"].encode()).hexdigest(), 16) % 4 == 0]   # ~25 % for periodic checks
        sets["kdense"] = [b for b in sets["ktrain"] if 2.1 <= b["density"] <= 11.2]
        if (D/"wacv_clean_v1.json").exists():          # WACV-clean v1: explicit links, ML365_Bottom dropped
            cl = {b["id"]: b for b in json.load(open(D/"wacv_clean_v1.json"))}
            sets["clean_dev3"] = [cl[s] for s in DEV]; sets["clean_all44"] = list(cl.values()); sets["clean_test41"] = [b for s, b in cl.items() if s not in DEV]
            for k, v in folds.items():                 # WACV-clean board folds (fine-tuning tests): train on two folds, test on the third
                sets[f"cfold{k}train"] = [b for b in cl.values() if b["board"] not in v["test_boards"]]
                sets[f"cfold{k}test"] = [b for b in cl.values() if b["board"] in v["test_boards"]]
        return sets
    wacv = {b["id"]: b for b in json.load(open(ROOT/"linkdata/wacv.json"))}
    folds = json.load(open(ROOT/"linkdata/folds.json"))   # copy of gdino_fewshot/wacv_in/folds.json (written by linkdata.py)
    sets = {"dev3": [wacv[s] for s in DEV], "all45": list(wacv.values()), "test42": [b for s, b in wacv.items() if s not in DEV]}
    if (ROOT/"linkdata/kicad199.json").exists(): sets["kicad199"] = json.load(open(ROOT/"linkdata/kicad199.json"))
    for k, v in folds.items():
        sets[f"fold{k}test"] = [b for b in wacv.values() if b["board"] in v["test_boards"]]
        sets[f"fold{k}train"] = [b for b in wacv.values() if b["board"] not in v["test_boards"]]
    if (ROOT/"linkdata/kicad_all.json").exists():
        kic = json.load(open(ROOT/"linkdata/kicad_all.json"))
        h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)
        val = {b["board"] for b in kic if h(b["board"]) % 20 == 0}                   # ~5 % of projects
        sets["kicadval"] = [b for b in kic if b["board"] in val]; sets["kicadtrain"] = [b for b in kic if b["board"] not in val]
        projs = sorted({b["board"] for b in sets["kicadtrain"]}, key=h)
        for n in (300, 1000): keep = set(projs[:n]); sets[f"kicad{n}"] = [b for b in sets["kicadtrain"] if b["board"] in keep]
    return sets

class Vocab:
    def __init__(self, prefixes): self.prefixes = sorted(prefixes); self.idx = {p: i+2 for i, p in enumerate(self.prefixes)}   # 0 = part, 1 = unknown
    @classmethod
    def from_boards(cls, boards): return cls({PFX.match(t["text"]).group(1) for b in boards for t in b["texts"] if PFX.match(t["text"])})
    def __call__(self, s):
        m = PFX.match(s)
        if m: return self.idx.get(m.group(1), 1)
        if s and NUMS.match(s): return self.idx.get("<NUM>", 1)                # digits only (letters lost)
        return self.idx.get("<TXT>", 1) if s else 1                            # other text / empty

def attach_pimg(boards, path):
    """Part image features from an npz with one (n parts, D) array per board id, in the boards' parts order (tools/part_roi.py photo);
    boards without a matching array keep no image (the model then uses its no-image vector). Returns the number of boards matched."""
    Z = np.load(path); k = 0
    for b in boards:
        if b["id"] in Z.files and len(Z[b["id"]]) == len(b["parts"]):
            for pp, v in zip(b["parts"], Z[b["id"]]): pp["pimg"] = v
            k += 1
    return k

def det_sample(c, rng):
    """A real detector class-score vector (9) of a val9 part of true class c (any class when c is unknown / other)."""
    pool = DET_POOL.get(c) if 0 <= c <= 8 and len(DET_POOL.get(c, [])) else DET_POOL["all"]
    return pool[rng.randrange(len(pool))]

def augment(b, rng, p_drop=0.15, p_fake=0.05, jitter=0.1):
    """A randomly transformed copy of a board: rotation by k*90 degrees and mirror, box-edge jitter (fraction of the box size),
    dropped labels (their parts become 'none'), fake designator-like labels that name no part."""
    k, mir = (rng.randrange(4), rng.random() < 0.5) if AUG_ROT else (0, False)
    def tf(box):
        x0, y0, x1, y1 = box
        if mir: x0, x1 = -x1, -x0
        for _ in range(k): x0, y0, x1, y1 = y0, -x1, y1, -x0          # rotate 90 degrees
        w, h = x1-x0, y1-y0
        return [x0+rng.uniform(-jitter, jitter)*w, y0+rng.uniform(-jitter, jitter)*h, x1+rng.uniform(-jitter, jitter)*w, y1+rng.uniform(-jitter, jitter)*h]
    parts = [dict(p, box=tf(p["box"])) for p in b["parts"]]
    if DET_POOL is not None and b.get("source") == "kicad":   # simulated detector output: a real val9 score vector of a part of the same class
        for p in parts: p["ds"] = det_sample(p.get("cls", -1), rng) if rng.random() >= DET_DROP else np.zeros(9, np.float32)   # --det_drop: no detector info
    keep, texts = [], []
    for j, t in enumerate(b["texts"]):                 # same random-number order as before
        if rng.random() >= p_drop: keep.append(j); texts.append(dict(t, box=tf(t["box"])))
    if any(p.get("link") is not None for p in parts):  # explicit links (WACV-clean): follow the kept texts; a dropped label -> none
        new = {o: k for k, o in enumerate(keep)}
        parts = [dict(p, link=new.get(p["link"])) if p.get("link") is not None else p for p in parts]
    if STRNOISE[0] or STRNOISE[1]:
        for t in texts:
            u = rng.random()
            if u < STRNOISE[0]: t["ocr"] = ""
            elif u < STRNOISE[0] + STRNOISE[1]: t["ocr"] = corrupt(t["text"], rng)
    if PREFIX_DROP:
        for t in texts:
            mm = PFX.match(t.get("ocr", t["text"]))
            if mm and rng.random() < PREFIX_DROP: t["ocr"] = mm.group(2) + mm.group(3)
    if texts and parts:
        used = {t["text"] for t in b["texts"]} | {p["des"] for p in b["parts"]}
        allb = np.array([o["box"] for o in parts+texts]); lo, hi = allb[:, :2].min(0), allb[:, 2:].max(0)
        for _ in range(int(p_fake*len(texts)) + (rng.random() < (p_fake*len(texts)) % 1)):
            t0 = rng.choice(texts)["box"]; w, h = t0[2]-t0[0], t0[3]-t0[1]; cx, cy = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
            s = None
            for _ in range(20):
                s = f"{rng.choice(FAKE_PFX)}{rng.randint(1, 999)}"
                if s not in used: break
            texts.append(dict(box=[cx-w/2, cy-h/2, cx+w/2, cy+h/2], text=s, angle=None)); used.add(s)
        nd = (int(DISTRACT*len(texts)) + (rng.random() < (DISTRACT*len(texts)) % 1)) if DISTRACT else 0
        for _ in range(nd):                            # non-designator texts (values, net names, numbers): name no part
            t0 = rng.choice(texts)["box"]; h = max(min(t0[2]-t0[0], t0[3]-t0[1]), 1e-3); s = distractor(rng); L = 0.6*h*max(len(s), 1)
            if rng.random() < 0.5: p0 = rng.choice(parts)["box"]; cx, cy = rng.uniform(p0[0]-h, p0[2]+h), rng.uniform(p0[1]-h, p0[3]+h)
            else: cx, cy = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
            w, hh = (h, L) if rng.random() < 0.3 else (L, h)
            texts.append(dict(box=[cx-w/2, cy-hh/2, cx+w/2, cy+hh/2], text=s, angle=None, distract=True))
    if NOISE and b.get("noise"):                       # real non-label boxes of the side (no string, PARSeq feature of the render crop): name no part
        nz = b["noise"]; k = min(len(nz), int(NOISE*len(b["texts"])) + (rng.random() < (NOISE*len(b["texts"])) % 1))
        texts += [dict(t, box=tf(t["box"])) for t in rng.sample(nz, k)]
    return dict(b, parts=parts, texts=texts)

def encode(b, vocab, as_numpy=False):
    """Board -> tensors (or numpy arrays, for DataLoader workers). Units = median part size (longer side)."""
    P = np.array([p["box"] for p in b["parts"]], float).reshape(-1, 4); T = np.array([t["box"] for t in b["texts"]], float).reshape(-1, 4)
    unit = float(np.median(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]))) if len(P) else 1.0
    B = np.concatenate([P, T]) / max(unit, 1e-6); w = np.maximum(B[:, 2]-B[:, 0], 1e-3); h = np.maximum(B[:, 3]-B[:, 1], 1e-3)
    n, m = len(P), len(T); is_text = np.r_[np.zeros(n), np.ones(m)]
    feats = np.stack([is_text, np.log(w), np.log(h), (w >= h).astype(float)], 1)
    pos = np.stack([(B[:, 0]+B[:, 2])/2, (B[:, 1]+B[:, 3])/2], 1); pos -= pos.mean(0)
    ocr = [t.get("ocr", t["text"]) for t in b["texts"]]          # what the token sees (noisy in training with --strnoise)
    pref = np.r_[np.zeros(n, int), [vocab(s) for s in ocr]]
    dig = np.full((n+m, 4), 10, int)   # right-aligned digits, 10 = pad
    for j, s in enumerate(ocr):
        mm = PFX.match(s) or (NUMS.match(s) if "<NUM>" in vocab.idx else None)
        if mm:
            ds = mm.group(2 if mm.re is PFX else 1)[-4:]; dig[n+j, 4-len(ds):] = [int(c) for c in ds]
    strs = [t["text"] for t in b["texts"]]
    tgt = []   # per part: list of target text indices, [] = none
    for p in b["parts"]: tgt.append(([p["link"]] if p["link"] is not None else []) if "link" in p else [j for j, s in enumerate(strs) if p["des"] and s == p["des"]])
    ign = [bool(p.get("ign")) for p in b["parts"]]
    tmask = np.zeros((n, m+1), bool)                 # target cells per part row (all same-string texts, or the none column)
    for i, t in enumerate(tgt): tmask[i, t if t else [m]] = True
    tmask[np.array(ign, bool)] = True                # unknown label: every cell is a target -> no loss
    extra = {"pcls": np.array([p.get("cls", -1) for p in b["parts"]], np.int64)}   # part class (9 = other, -1 = unknown)
    tcl = []                                         # class each text names: its prefix's class 0-8; 9 = not a designator; -1 = unknown / other
    for t in b["texts"]:
        mm = PFX.match(t["text"]); tcl.append(9 if t.get("distract") or t.get("noise") else (PFX_CLS.get(mm.group(1), -1) if mm else -1))
    extra["tcls"] = np.array(tcl, np.int64)
    if IMG_DIM:                                      # image feature per token; zeros where a box has none (fake labels, boards without features)
        img = np.zeros((n+m, IMG_DIM), np.float32)
        for k, o in enumerate(b["parts"] + b["texts"]):
            if "img" in o: img[k] = o["img"]
        extra["img"] = img
    if TXT_DIM:                                      # PARSeq feature per text token (zeros for parts)
        tf = np.zeros((n+m, TXT_DIM), np.float32)
        for k, t in enumerate(b["texts"]):
            if "tf" in t: tf[n+k] = t["tf"]
        extra["tf"] = tf
    if DET_DIM:                                      # detector class scores per part token (zeros for texts / unmatched parts)
        ds = np.zeros((n+m, DET_DIM), np.float32)
        for k, pp in enumerate(b["parts"]):
            if "ds" in pp: ds[k] = pp["ds"]
        extra["ds"] = ds
    if PIMG_DIM:                                     # part image feature + has-image flag (pok 0: texts, parts without image)
        pimg = np.zeros((n+m, PIMG_DIM), np.float32); pok = np.zeros((n+m, 1), np.float32)
        for k, pp in enumerate(b["parts"]):
            if pp.get("pimg") is not None: pimg[k] = pp["pimg"]; pok[k] = 1.0
        extra["pimg"], extra["pok"] = pimg, pok
    if as_numpy:
        return dict(id=b["id"], n=n, m=m, feats=feats.astype(np.float32), pos=pos.astype(np.float32), pref=pref.astype(np.int64), dig=dig.astype(np.int64),
                    box=B.astype(np.float32), tmask=tmask, tgt=tgt, ign=ign, strs=strs, des=[p["des"] for p in b["parts"]], **extra)
    f = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(id=b["id"], n=n, m=m, feats=f(feats), pos=f(pos), pref=f(pref, torch.long), dig=f(dig, torch.long), box=f(B), tmask=torch.from_numpy(tmask),
                tgt=tgt, ign=ign, strs=strs, des=[p["des"] for p in b["parts"]], **{k: f(v, torch.long if v.dtype == np.int64 else torch.float32) for k, v in extra.items()})

ARCH = ("size", "tokens", "head", "geobias", "strrel", "glayers", "local", "imgdim", "clshead", "txtdim", "clsmode", "rankrel", "detdim", "arhead", "pos", "nosize", "pimgdim", "clshead_tf", "qknorm")
ARCH_LATE = {"pos": "rope", "nosize": False, "pimgdim": 0, "clshead_tf": 2, "qknorm": False}   # added 2026-10-23 after meta ckpts existed (absmlp runs: pos only; rope4 runs: no pimgdim)
OLD_WARNED = set()                                    # old ckpts (no meta) already warned about

def load_model(path, dev):
    """A saved checkpoint -> (model in eval mode, vocab, its training args). A ckpt with meta (tools/run_meta.py) must carry every
    architecture key; an old ckpt (no meta) keeps the defaults below for the keys it lacks, with one warning per ckpt."""
    ck = torch.load(path, map_location="cpu"); A = ck["args"]; vocab = Vocab(ck["vocab"]); miss = [k for k in ARCH if k not in A]
    if "meta" in ck:                                   # keys added after meta ckpts existed may be missing (code that knew a key always saves it)
        assert not [k for k in miss if k not in ARCH_LATE], f"{path}: ckpt with meta lacks architecture keys {miss}"
        if miss and str(path) not in OLD_WARNED:
            OLD_WARNED.add(str(path)); print(f"WARNING: {path}: keys newer than its code, old defaults: {[(k, ARCH_LATE[k]) for k in miss]}", file=sys.stderr, flush=True)
    elif str(path) not in OLD_WARNED:
        OLD_WARNED.add(str(path)); print(f"WARNING: {path} is an old ckpt (no meta); defaults for missing keys: {miss or 'none missing'}", file=sys.stderr, flush=True)
    d, L = SIZES[A["size"]]
    m = LinkFormer(d, L, len(vocab.idx)+2, A["tokens"] == "E2", 0.0, A.get("head", "dot"), A.get("geobias", False), A.get("strrel", False), A.get("glayers", False), A.get("local", 0.0), A.get("imgdim", 0), A.get("clshead", False), A.get("txtdim", 0), A.get("clsmode", "parallel"), A.get("rankrel", False), A.get("detdim", 0), A.get("arhead", False), pos=A.get("pos", "rope"), nosize=A.get("nosize", False), pimgdim=A.get("pimgdim", 0), clshead_tf=A.get("clshead_tf", 2), qknorm=A.get("qknorm", False))
    m.load_state_dict(ck["model"]); return m.to(dev).eval(), vocab, A

def messy(b, margin=0.5):
    """Per part: True when its label (first text with the part's designator) is printed >= margin median part sizes nearer, by edge gap,
    to another part than to its own (the 'misplaced' rule of tools/label_placement_check.py)."""
    P = np.array([p["box"] for p in b["parts"]], float).reshape(-1, 4); T = np.array([t["box"] for t in b["texts"]], float).reshape(-1, 4)
    if len(P) < 2 or not len(T): return [False]*len(P)
    unit = max(float(np.median(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]))), 1e-6)
    gx = np.maximum(0, np.maximum(T[:, None, 0]-P[None, :, 2], P[None, :, 0]-T[:, None, 2]))
    gy = np.maximum(0, np.maximum(T[:, None, 1]-P[None, :, 3], P[None, :, 1]-T[:, None, 3])); G = np.hypot(gx, gy)/unit
    first = {}
    for j, t in enumerate(b["texts"]): first.setdefault(t["text"], j)
    out = []
    for i, p in enumerate(b["parts"]):
        j = first.get(p["des"]) if p["des"] else None
        if j is None: out.append(False); continue
        g = G[j].copy(); own = g[i]; g[i] = np.inf; out.append(bool(own - g.min() >= margin))
    return out

def disk_ok(path, min_gb, tag):
    """--min_free_gb: False (and a DISK ALERT line) when the file system of path has less free space than min_gb GB; 0 = always True."""
    if not min_gb: return True
    free = shutil.disk_usage(Path(path).parent).free / 2**30
    if free >= min_gb: return True
    print(f"[{tag}] DISK ALERT: {free:.1f} GB free < --min_free_gb {min_gb:g}: {path} NOT saved", flush=True); return False

def from_numpy(e): return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in e.items()}

class BoardStream(torch.utils.data.IterableDataset):
    """Endless stream of (augmented) encoded training boards; each DataLoader worker has its own RNG. Yields numpy arrays (no shared memory)."""
    def __init__(self, trb, mixb, mix_p, vocab, aug, seed): super().__init__(); self.trb, self.mixb, self.mix_p, self.vocab, self.aug, self.seed = trb, mixb, mix_p, vocab, aug, seed
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(self.seed*1000 + (wi.id if wi else 0))
        while True:
            mix = bool(self.mixb) and rng.random() < self.mix_p; b = rng.choice(self.mixb if mix else self.trb)
            if IGN_MESSY and not mix: b = dict(b, parts=[dict(p, ign=True) if m else p for p, m in zip(b["parts"], messy(b))])
            e = encode(augment(b, rng, *AUGP) if (self.aug or mix) else b, self.vocab, as_numpy=True)
            if e["n"] and e["m"]: yield e

def dense_batch(es, dev, targets=False, pin=False):
    """Training boards (encode(..., as_numpy=True) dicts) -> one padded batch on dev, one non-blocking host->device copy per field:
    tokens padded to the largest board, valid-token mask, part counts n; with targets, tm (B, max parts, N+1) = target cells per part
    (text tokens at their token index, none at column N; padded part rows all True = no loss, like 'ign' rows)."""
    B = len(es); Ns = [e["n"]+e["m"] for e in es]; r16 = lambda k: -(-k // 16) * 16
    N = r16(max(Ns)); P = r16(max(e["n"] for e in es))     # multiples of 16: aligned kernels, one torch.compile graph (padding is masked)
    feats = np.zeros((B, N, 4), np.float32); pos = np.zeros((B, N, 2), np.float32); box = np.zeros((B, N, 4), np.float32)
    pref = np.zeros((B, N), np.int64); dig = np.full((B, N, 4), 10, np.int64); valid = np.zeros((B, N), bool)
    for b, (e, k) in enumerate(zip(es, Ns)):
        feats[b, :k], pos[b, :k], box[b, :k], pref[b, :k], dig[b, :k], valid[b, :k] = e["feats"], e["pos"], e["box"], e["pref"], e["dig"], True
    out = dict(feats=feats, pos=pos, box=box, pref=pref, dig=dig, valid=valid, n=np.array([e["n"] for e in es], np.int64))
    if targets and "pcls" in es[0]:                  # class targets per part row; other (9) / unknown / padded rows = -1 (no loss)
        pc = np.full((B, P), -1, np.int64)
        for b, e in enumerate(es): c = e["pcls"].copy(); c[(c < 0) | (c > 8)] = -1; pc[b, :e["n"]] = c
        out["pc"] = pc
        tc = np.full((B, N), -1, np.int64)                # class named by each text token (mix / pool class heads)
        for b, e in enumerate(es): tc[b, e["n"]:e["n"]+e["m"]] = e["tcls"]
        out["tc"] = tc
    for key in ("img", "tf", "ds", "pimg", "pok"):
        if key in es[0]:
            arr = np.zeros((B, N, es[0][key].shape[1]), np.float32)
            for b, (e, k) in enumerate(zip(es, Ns)): arr[b, :k] = e[key]
            out[key] = arr
    if targets:
        tm = np.ones((B, P, N+1), bool)
        for b, e in enumerate(es):
            n, m = e["n"], e["m"]; tm[b, :n] = False; tm[b, :n, n:n+m] = e["tmask"][:, :m]; tm[b, :n, N] = e["tmask"][:, m]
        out["tm"] = tm
        if NONE_W != 1.0:                                 # rows whose only target is the none column (not 'ign') get weight NONE_W
            w = np.ones((B, P), np.float32)
            for b, e in enumerate(es):
                t = e["tmask"]; w[b, :e["n"]][t[:, -1] & ~t[:, :-1].any(1) & ~np.array(e["ign"], bool)] = NONE_W
            out["w"] = w
    if pin: return {k: torch.from_numpy(v).pin_memory().to(dev, non_blocking=True) for k, v in out.items()}   # --prefetch: truly async copies
    return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in out.items()}

def rope_cs(pos, freqs):
    """cos / sin of the RoPE angles per coordinate, (..., 1, N, q): computed once per forward, shared by q, k and all layers.
    pos (..., N, K): K = 2 (centre x, y; --pos rope) or 4 (corners x1, y1, x2, y2; --pos rope4)."""
    return [(torch.cos(ang), torch.sin(ang)) for ang in ((pos[..., a:a+1] * freqs).unsqueeze(-3) for a in range(pos.shape[-1]))]

def rope_apply(x, cs):
    """Same rotation as rope_2d, with precomputed cos / sin: the head dim is cut into len(cs) equal blocks, block a rotated by
    coordinate a (2 blocks = rope_2d; 4 blocks of 16 = rope4, like FLUX EmbedND with axes_dim [16, 16, 16, 16]). v is not rotated."""
    w = x.shape[-1] // len(cs); out = []
    for a, (c, s) in enumerate(cs):
        blk = x[..., a*w:(a+1)*w]; x1, x2 = blk[..., 0::2], blk[..., 1::2]; out += [torch.stack([x1*c - x2*s, x1*s + x2*c], -1).flatten(-2)]
    return torch.cat(out, -1)

def rope_2d(x, pos, freqs):
    """x: (heads, N, hd) with pos (N, 2), or (B, heads, N, hd) with pos (B, N, 2); first half of hd rotated by x-position, second
    half by y-position, one frequency per pair."""
    hd = x.shape[-1]; q = hd // 4
    out = []
    for a, half in enumerate((x[..., :hd//2], x[..., hd//2:])):
        ang = (pos[..., a:a+1] * freqs[:q]).unsqueeze(-3)          # (1, N, q) or (B, 1, N, q)
        c, s = torch.cos(ang), torch.sin(ang); x1, x2 = half[..., 0::2], half[..., 1::2]
        out += [torch.stack([x1*c - x2*s, x1*s + x2*c], -1).flatten(-2)]
    return torch.cat(out, -1)

class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.0, qknorm=False):
        super().__init__(); self.h = heads; self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.p = dropout; self.drop = nn.Dropout(dropout)
        self.qn = nn.LayerNorm(d // heads) if qknorm else None; self.kn = nn.LayerNorm(d // heads) if qknorm else None   # --qknorm: per head, before RoPE
        self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d); self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, pos, freqs, bias=None, cs=None):
        N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(N, 3, self.h, d//self.h).permute(1, 2, 0, 3)
        if self.qn is not None: q, k = self.qn(q), self.kn(k)
        if cs is not None: q, k = rope_apply(q, cs), rope_apply(k, cs)               # --pos rope4 (precomputed tables)
        elif freqs is not None: q, k = rope_2d(q, pos, freqs), rope_2d(k, pos, freqs)   # freqs None: no RoPE (--pos abs / none)
        am = bias[None].to(q.dtype) if bias is not None else None                  # (1, heads, N, N), added to the attention scores
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=am, dropout_p=self.p if self.training else 0.0)[0].permute(1, 0, 2).reshape(N, d)
        x = x + self.drop(self.o(a)); return x + self.drop(self.mlp(self.n2(x)))
    def forward_b(self, x, cs, am):
        """Padded batch: x (B, N, d), cs = rope_cs(pos), am (B, heads or 1, 1 or N, N) additive mask / bias (-inf on padded keys), already
        in the attention dtype."""
        B, N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d//self.h).permute(2, 0, 3, 1, 4)
        if self.qn is not None: q, k = self.qn(q), self.kn(k)
        if cs is not None: q, k = rope_apply(q, cs), rope_apply(k, cs)          # cs None: no RoPE (--pos abs / none)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.p if self.training else 0.0).transpose(1, 2).reshape(B, N, d)
        x = x + self.drop(self.o(a)); return x + self.drop(self.mlp(self.n2(x)))

def pair_geometry(box, is_text):
    """(N, N, 10) features of every token pair (query a, key b), boxes in median-part units; offsets and gaps are measured in the
    query box's own width / height (Relation Networks style), plus the plain gap in units and the two token types."""
    x0, y0, x1, y1 = box.unbind(-1); w = (x1-x0).clamp_min(1e-3); h = (y1-y0).clamp_min(1e-3); cx, cy = (x0+x1)/2, (y0+y1)/2
    A = lambda t: t[..., :, None]; K = lambda t: t[..., None, :]                   # query (row) / key (column) broadcast; any leading batch dims
    dx = K(cx)-A(cx); dy = K(cy)-A(cy)
    gx = torch.maximum(K(x0)-A(x1), A(x0)-K(x1)).clamp_min(0); gy = torch.maximum(K(y0)-A(y1), A(y0)-K(y1)).clamp_min(0)
    wa, ha = A(w), A(h); g = torch.hypot(gx, gy); sh = dx.shape
    return torch.stack([torch.sign(dx)*torch.log1p(dx.abs()/wa), torch.sign(dy)*torch.log1p(dy.abs()/ha), torch.log1p(gx/wa), torch.log1p(gy/ha),
                        torch.log1p(g/torch.maximum(wa, ha)), torch.log(K(w)/wa), torch.log(K(h)/ha), torch.log1p(g),
                        A(is_text).expand(sh), K(is_text).expand(sh)], -1)

def pair_strings(pref, dig, is_text):
    """(..., N, N, 3) string relations of token pairs (query a, key b) when both are texts with a known prefix: same prefix; signed log of the
    designator number difference (b - a) when both have digits; consecutive numbers (|difference| = 1). Zero for every other pair."""
    known = (pref > 1) & (is_text > 0.5); has = (dig < 10).any(-1)
    num = ((dig < 10) * dig.clamp(max=9) * torch.tensor([1000, 100, 10, 1], device=dig.device)).sum(-1).float()   # right-aligned digits, 10 = pad
    A = lambda t: t[..., :, None]; K = lambda t: t[..., None, :]
    same = A(known) & K(known) & (A(pref) == K(pref)); both = same & A(has) & K(has); dn = K(num) - A(num)
    return torch.stack([same.float(), torch.where(both, torch.sign(dn)*torch.log1p(dn.abs()), torch.zeros_like(dn)), (both & (dn.abs() == 1)).float()], -1)

def pair_rank(box, is_text, valid):
    """(..., N, N, 3) nearest-neighbour relations between a part and a text (query a, key b of the other type): log(1 + rank of b among a's
    other-type tokens by edge gap), the same seen from b, and 'mutual nearest' (both ranks 0). Zero for same-type pairs and padding."""
    x0, y0, x1, y1 = box.unbind(-1); cx, cy = (x0+x1)/2, (y0+y1)/2
    A = lambda t: t[..., :, None]; K = lambda t: t[..., None, :]
    gx = torch.maximum(K(x0)-A(x1), A(x0)-K(x1)).clamp_min(0); gy = torch.maximum(K(y0)-A(y1), A(y0)-K(y1)).clamp_min(0)
    dist = torch.hypot(gx, gy) + 0.01*torch.hypot(K(cx)-A(cx), K(cy)-A(cy))    # edge gap; the centre distance breaks ties
    ok = (A(is_text) != K(is_text)) & A(valid) & K(valid)
    r = dist.float().masked_fill(~ok, float("inf")).argsort(-1).argsort(-1).float(); rt = r.transpose(-1, -2)
    return torch.stack([torch.log1p(r), torch.log1p(rt), ((r == 0) & (rt == 0)).float()], -1) * ok[..., None]

class ARHead(nn.Module):
    """Causal (autoregressive) link head (2026-10-19, owner idea: decide the links one part at a time, so one-to-one is part of the
    model, not a Hungarian post-step). Parts are ordered by the base head's confidence (max row probability, high first). Step k sees,
    through a causal transformer over the ordered parts, the decisions of steps < k (token = part + embedding of the chosen text or
    none, shifted by one); texts taken at steps < k are masked (-inf). Logits = base logits + learned correction (zero-initialised,
    so an untrained head = greedy one-to-one decoding of the base head in confidence order)."""
    def __init__(self, d, heads, layers=2):
        super().__init__(); self.d = d
        self.dz = nn.Linear(d, d); self.none_z = nn.Parameter(torch.zeros(d)); self.start = nn.Parameter(torch.zeros(d))
        self.dec = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.nz = nn.Linear(d, 1)
        for m in (self.q, self.nz): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def seq_logits(self, hp, xt, base, chosen, protect=None):
        """hp (P, d) part tokens in decoding order, xt (M, d) text tokens, base (P, M+1) base logits in that order, chosen (P,) the
        decision of each step (0..M-1 = text, M = none; teacher forcing in training, the running greedy choices at inference).
        -> (P, M+1) step logits, texts taken at earlier steps masked."""
        P, M = base.shape[0], base.shape[1] - 1
        z = torch.where((chosen < M)[:, None], xt[chosen.clamp(max=max(M-1, 0))] if M else self.none_z.expand(P, -1), self.none_z.expand(P, -1))
        u = hp + torch.cat([self.start[None], self.dz(z[:-1])], 0)
        cm = torch.triu(torch.full((P, P), float("-inf"), device=hp.device), 1)
        o = self.dec(u, mask=cm)
        L = base.float() + torch.cat([self.q(o) @ self.k(xt).T / math.sqrt(self.d), self.nz(o)], 1).float()
        if M:
            oh = F.one_hot(chosen.clamp(max=M), M+1)[:, :M].float()                       # (P, M) chosen text per step
            taken = (torch.cumsum(oh, 0) - oh) > 0                                         # taken at steps < k
            if protect is not None: taken = taken & ~protect[:, :M]                        # training: never mask a part's own target
            L = torch.cat([L[:, :M].masked_fill(taken, float("-inf")), L[:, M:]], 1)
        return L
    @torch.no_grad()
    def greedy(self, hp, xt, base):
        """Inference: decide step by step in confidence order; -> (P, M+1) step logits in the ORIGINAL part order."""
        P, M = base.shape[0], base.shape[1] - 1
        order = torch.argsort(torch.softmax(base.float(), 1).max(1).values, descending=True)
        chosen = torch.full((P,), M, dtype=torch.long, device=hp.device); rows = torch.empty(P, M+1, device=hp.device)
        for s in range(P):
            L = self.seq_logits(hp[order[:s+1]], xt, base[order[:s+1]], chosen[:s+1])[s]
            rows[s] = L.clamp_min(-1e4); chosen[s] = int(L.argmax())
        out = torch.empty_like(rows); out[order] = rows; return out

class LinkFormer(nn.Module):
    def __init__(self, d, layers, n_prefix, use_string, dropout=0.0, head="dot", geobias=False, strrel=False, glayers=False, local=0.0, imgdim=0, clshead=False, txtdim=0, clsmode="parallel", rankrel=False, detdim=0, arhead=False, pos="rope", nosize=False, pimgdim=0, clshead_tf=2, qknorm=False):
        super().__init__(); self.use_string = use_string
        assert pos in ("rope", "abs", "none", "rope4"), f"--pos {pos}"; self.pos = pos; self.nosize = bool(nosize)
        if nosize: self.register_buffer("fmask", torch.tensor([1.0, 0.0, 0.0, 0.0]), persistent=False)   # token geometry = type only
        if pos == "rope4": self.register_buffer("freqs4", 2*math.pi/torch.logspace(math.log10(0.5), math.log10(512), 8), persistent=False)
        hd = 64; heads = d // hd; self.head = head; self.geobias = geobias; self.rankrel = rankrel
        self.strrel, self.glayers, self.heads, self.local, self.imgdim, self.clshead = strrel, glayers, heads, local, imgdim, clshead
        self.clsmode = clsmode if clshead else ""
        if clshead and clsmode == "parallel": self.cls = nn.Linear(d, 9)         # end-to-end class of each part (9 WACV classes)
        if clshead and clsmode in ("mix", "pool"): self.tcls = nn.Linear(d, 10)   # class each text names (+ 9 = not a designator)
        if clshead and clsmode == "mix": self.fcls = nn.Linear(d, 9)            # fallback class of a part without a text
        if clshead and clsmode == "pool": self.cmlp = nn.Sequential(nn.Linear(2*d+1, d), nn.GELU(), nn.Linear(d, 9))
        self.txtdim = txtdim; self.detdim = detdim
        if detdim: self.dsp = nn.Sequential(nn.Linear(detdim, d), nn.GELU(), nn.Linear(d, d))   # detector class scores -> part token
        if txtdim: self.tfp = nn.Sequential(nn.Linear(txtdim, d), nn.GELU(), nn.Linear(d, d))   # text-image feature -> token
        if imgdim: self.img = nn.Linear(imgdim, d); nn.init.zeros_(self.img.weight); nn.init.zeros_(self.img.bias)   # starts as the model without images
        self.inp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.pref = nn.Embedding(n_prefix, d); self.dig = nn.ModuleList([nn.Embedding(11, d) for _ in range(4)])
        self.blocks = nn.ModuleList([Block(d, heads, dropout, qknorm) for _ in range(layers)]); self.nf = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d)
        if head == "lg": self.ms = nn.Linear(d, 1); self.mt = nn.Linear(d, 1)          # matchability of parts / texts
        else: self.none = nn.Linear(d, 1)
        nf = 10 + 3*strrel + 3*rankrel                                           # pair features: geometry (+ string relations, + nearest-neighbour ranks)
        if geobias and glayers: self.gb = nn.Sequential(nn.Linear(nf, 32), nn.GELU()); self.gbl = nn.Linear(32, heads*layers)   # own bias per layer
        elif geobias: self.gb = nn.Sequential(nn.Linear(nf, 32), nn.GELU(), nn.Linear(32, heads))
        lam = torch.logspace(math.log10(0.5), math.log10(512), hd // 4)   # wavelengths from half a part to 512 parts
        self.register_buffer("freqs", 2*math.pi/lam)
        self.arhead = arhead
        if arhead: self.ar = ARHead(d, heads)                              # causal link head on top of the matrix head
        if pos == "abs":                                                   # created last: the other weights get the same init as with rope
            ch = [2, 32, 64, 128, 256]                                     # SuperGlue KeypointEncoder; LayerNorm instead of BatchNorm (padding)
            self.kenc = nn.Sequential(*[m for a, b in zip(ch, ch[1:]) for m in (nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU())], nn.Linear(ch[-1], d))
            nn.init.zeros_(self.kenc[-1].bias)
        self.board_log = False                                             # --gradlog: keep the per-board losses of each training batch
        self.pimgdim = pimgdim; self.img_drop = 0.0                        # img_drop: set by the trainer (--img_drop), training only
        if pimgdim:                                                        # part image feature (detector RoI): created last, as kenc
            self.pimp = nn.Sequential(nn.Linear(pimgdim, d), nn.GELU(), nn.Linear(d, d))
            self.noimg = nn.Parameter(torch.randn(d) * 0.02)               # learned "no image" vector (DiT / MAE style)
        assert clsmode in ("parallel", "mix", "pool", "sel") and not (clshead and clsmode == "sel" and head != "dot"), (clsmode, head)
        self.clshead_tf = int(clshead_tf); self.cls_link = "gt"; self.cls_frac = 0.0   # cls_link / cls_frac: set by the trainer
        self.cls_detach_text = False                                       # --cls_detach_text: set by the trainer
        if clshead and clsmode == "sel":                                   # class head on the selected link (owner design 2026-10-23), created last
            self.cin = nn.Linear(2*d + 1, d); self.enone = nn.Parameter(torch.randn(d) * 0.02); self.cout = nn.Linear(d, 9)
            if self.clshead_tf:                                            # K transformer blocks over the board's class tokens
                self.cblocks = nn.ModuleList([Block(d, heads, 0.1, qknorm) for _ in range(self.clshead_tf)]); self.cnf = nn.LayerNorm(d)
    def cls_sel(self, x, box, pos, npart, jt):
        """--clsmode sel: c_i = W_in [h_i ; h_j(i) or e_none ; none_i]; K Blocks over the board's class tokens with the trunk's position
        encoding on the part boxes (K = 0: GELU only) -> LayerNorm -> W_out, 9 logits. x (B, N, D) final trunk states; box, pos (B, N, .);
        npart (B,) parts per board; jt (B, P) token index of the selected text, -1 = none (an index: no gradient through the selection)."""
        B, P = jt.shape; hp = x[:, :P]; none = jt < 0
        ht = torch.gather(x, 1, jt.clamp_min(0)[..., None].expand(-1, -1, x.shape[2]))
        if self.cls_detach_text: ht = ht.detach()                          # no class-loss gradient into the trunk through the text token
        ht = torch.where(none[..., None], self.enone.to(ht.dtype).expand_as(ht), ht)
        c = self.cin(torch.cat([hp, ht, none[..., None].to(hp.dtype)], -1))
        if not self.clshead_tf: return self.cout(F.gelu(c)).float()
        pv = torch.arange(P, device=x.device)[None] < npart[:, None]; bx, ps = box[:, :P], pos[:, :P]
        if self.pos == "abs": c = c + self.abs_pos(bx, pv)
        am = torch.zeros(B, 1, 1, P, device=x.device).masked_fill(~pv[:, None, None, :], float("-inf")).to(c.dtype)
        cs = self.rope_tab(bx, ps)
        for blk in self.cblocks: c = blk.forward_b(c, cs, am)
        return self.cout(self.cnf(c)).float()
    def part_img(self, d):
        """--pimgfeat: part tokens get (1 - d_i) MLP(f_i) + d_i e_noimg; d_i = 1 without an image (pok 0), else Bernoulli(img_drop) in
        training only; text tokens get nothing. (B, N, d) addend."""
        part = 1 - d["feats"][..., :1]; ok = d["pok"] if "pok" in d else torch.zeros_like(part)
        if self.training and self.img_drop > 0: ok = ok * (torch.rand_like(ok) >= self.img_drop).to(ok.dtype)
        f = self.pimp(d["pimg"]) if "pimg" in d else 0.0
        return part * (ok * f + (1 - ok) * self.noimg)
    def tok(self, feats):
        """Token geometry input: (is_text, log w, log h, w >= h); --nosize keeps only is_text (the size enters through rope4 only)."""
        return feats * self.fmask if self.nosize else feats
    def rope_tab(self, box, pos):
        """cos / sin tables of the RoPE in use: rope = box centres; rope4 = corners (x1, y1, x2, y2) in the same median-part units and
        frame as the centres (board mean of the centres subtracted); None = no RoPE (--pos abs / none)."""
        if self.pos == "rope": return rope_cs(pos, self.freqs)
        if self.pos == "rope4":
            m = (box[..., :2] + box[..., 2:]) / 2 - pos                    # board mean of the centres (0 on padded tokens)
            return rope_cs(box - torch.cat([m, m], -1), self.freqs4)
        return None
    def abs_pos(self, box, valid):
        """--pos abs: KeypointEncoder of the box centres, (c - board centre) / (0.7 max(board w, h)), board = union of the valid boxes."""
        v = valid[..., None]; big = 1e9
        lo = torch.where(v, box[..., :2], big).amin(-2, keepdim=True); hi = torch.where(v, box[..., 2:], -big).amax(-2, keepdim=True)
        s = 0.7 * (hi - lo).amax(-1, keepdim=True).clamp_min(1e-6)
        return self.kenc(((box[..., :2] + box[..., 2:]) / 2 - (lo + hi) / 2) / s)
    def forward(self, e):
        if isinstance(e, list): return self.forward_batch(e)
        if "tm" in e: return self.train_loss(e)
        if self.strrel or self.rankrel or self.glayers or self.local or self.imgdim or self.clshead or self.txtdim or self.detdim or self.pimgdim: return self.forward_batch([e])[0]   # dense() only
        x = self.inp(self.tok(e["feats"]))
        if self.use_string: x = x + self.pref(e["pref"]) + sum(emb(e["dig"][:, k]) for k, emb in enumerate(self.dig))
        if self.pos == "abs": x = x + self.abs_pos(e["box"], torch.ones(x.shape[0], dtype=torch.bool, device=x.device))
        bias = self.gb(pair_geometry(e["box"], e["feats"][:, 0])).permute(2, 0, 1) if self.geobias else None   # (heads, N, N)
        cs = self.rope_tab(e["box"], e["pos"]) if self.pos == "rope4" else None
        for blk in self.blocks: x = blk(x, e["pos"], self.freqs if self.pos == "rope" else None, bias, cs=cs)
        return self.head_out(self.nf(x), e["n"])
    def forward_batch(self, es, with_cls=False, gt_links=None):
        """Several boards in one padded batch (same math as one forward per board; padded tokens are masked out as keys).
        with_cls (class head): also return each board's (n parts, 9) class logits. --clsmode sel: the head gets the Hungarian link of
        each part (decode()); gt_links (one {part: text index or None} per board) also gives self._cls_gt, the head with the true link."""
        Ns = [e["feats"].shape[0] for e in es]; N = max(Ns); B = len(es)
        if self.strrel or self.rankrel or self.glayers or self.local or self.imgdim or self.clshead or self.txtdim or self.detdim or self.pimgdim:   # through dense(): pad as dense_batch does
            pad = lambda t, v=0: F.pad(t, (0, 0)*(t.dim()-1) + (0, N-t.shape[0]), value=v)
            d = dict(feats=torch.stack([pad(e["feats"]) for e in es]), pos=torch.stack([pad(e["pos"]) for e in es]), box=torch.stack([pad(e["box"]) for e in es]),
                     pref=torch.stack([pad(e["pref"]) for e in es]), dig=torch.stack([pad(e["dig"], 10) for e in es]),
                     valid=torch.stack([torch.arange(N, device=es[0]["feats"].device) < n for n in Ns]))
            for key in ("img", "tf", "ds", "pimg", "pok"):
                if key in es[0]: d[key] = torch.stack([pad(e[key]) for e in es])
            x = self.dense(d); outs = [self.head_out(x[b, :n], e["n"]) for b, (e, n) in enumerate(zip(es, Ns))]
            if self.arhead and not self.training:                               # causal head: step logits (one-to-one by construction)
                outs = [self.ar.greedy(x[b, :e["n"]].float(), x[b, e["n"]:n].float(), o.float()) if e["n"] and n > e["n"] else o
                        for b, (e, n, o) in enumerate(zip(es, Ns, outs))]
            if not with_cls: return outs
            if self.clsmode == "sel":                                            # Hungarian link -> selected text -> class head
                P = max(e["n"] for e in es); npart = torch.tensor([e["n"] for e in es], device=x.device)
                pick = lambda Ls: torch.tensor([[(e["n"] + Lb[i]) if Lb.get(i) is not None else -1 for i in range(P)] for e, Lb in zip(es, Ls)], device=x.device)
                self._links = [decode(o) for o in outs]; c = self.cls_sel(x, d["box"], d["pos"], npart, pick(self._links))
                if gt_links is not None: self._cls_gt = self.cls_sel(x, d["box"], d["pos"], npart, pick(gt_links))
                return outs, [c[b, :e["n"]] for b, e in enumerate(es)]
            if self.clsmode == "parallel": return outs, [self.cls(x[b, :e["n"]]).float() for b, e in enumerate(es)]
            cls = []
            for b, (e, o) in enumerate(zip(es, outs)):                           # link probs over the board's texts + none
                A = torch.softmax(o.float(), 1); xb = x[b, :Ns[b]]
                cls.append(self.class_through_link(torch.cat([A.new_zeros(e["n"], e["n"]), A], 1), xb[:e["n"]], xb))
            return outs, cls
        pad = lambda t, v=0: F.pad(t, (0, 0)*(t.dim()-1) + (0, N-t.shape[0]), value=v)
        valid = torch.zeros(B, N, dtype=torch.bool, device=es[0]["feats"].device)
        for b, n in enumerate(Ns): valid[b, :n] = True
        x = self.inp(self.tok(torch.stack([pad(e["feats"]) for e in es]))); pos = torch.stack([pad(e["pos"]) for e in es])
        if self.use_string:
            dig = torch.stack([pad(e["dig"], 10) for e in es])
            x = x + self.pref(torch.stack([pad(e["pref"]) for e in es])) + sum(emb(dig[..., k]) for k, emb in enumerate(self.dig))
        if self.pos == "abs": x = x + self.abs_pos(torch.stack([pad(e["box"]) for e in es]), valid)
        am = torch.zeros(B, 1, 1, N, device=x.device).masked_fill(~valid[:, None, None, :], float("-inf"))
        if self.geobias:
            am = am + torch.stack([F.pad(self.gb(pair_geometry(e["box"], e["feats"][:, 0])).permute(2, 0, 1), (0, N-n, 0, N-n)) for e, n in zip(es, Ns)])
        am = am.to(torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype).contiguous()
        cs = self.rope_tab(torch.stack([pad(e["box"]) for e in es]), pos)
        for blk in self.blocks: x = blk.forward_b(x, cs, am)
        x = self.nf(x)
        return [self.head_out(x[b, :n], e["n"]) for b, (e, n) in enumerate(zip(es, Ns))]
    def dense(self, d):
        """Padded batch from dense_batch -> final token features (B, N, d). Same math as forward_batch, without per-board loops.
        The bias must be contiguous: a permuted (strided) attn_mask makes SDPA fall back from the memory-efficient kernel to the slow math path."""
        B, N = d["valid"].shape; x = self.inp(self.tok(d["feats"]))
        if self.use_string: x = x + self.pref(d["pref"]) + sum(emb(d["dig"][..., k]) for k, emb in enumerate(self.dig))
        if self.imgdim and "img" in d: x = x + self.img(d["img"])                  # image feature of each box (zero-initialised adapter)
        if self.txtdim and "tf" in d: x = x + self.tfp(d["tf"]) * d["feats"][..., :1]   # text tokens only
        if self.detdim and "ds" in d: x = x + self.dsp(d["ds"]) * (1 - d["feats"][..., :1])   # part tokens only
        if self.pos == "abs": x = x + self.abs_pos(d["box"], d["valid"])           # absolute position, once at the input
        if self.pimgdim: x = x + self.part_img(d)                                  # part image feature or the no-image vector
        am = torch.zeros(B, 1, 1, N, device=x.device).masked_fill(~d["valid"][:, None, None, :], float("-inf"))
        if self.local:   # local attention: a real token sees only tokens whose centre is within `local` median part sizes (padded rows: all)
            far = (torch.cdist(d["pos"], d["pos"]) > self.local) & d["valid"][:, :, None]
            am = am.masked_fill(far[:, None], float("-inf"))
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        cs = self.rope_tab(d["box"], d["pos"])
        if self.geobias:
            pf = pair_geometry(d["box"], d["feats"][..., 0])
            if self.strrel: pf = torch.cat([pf, pair_strings(d["pref"], d["dig"], d["feats"][..., 0])], -1)
            if self.rankrel: pf = torch.cat([pf, pair_rank(d["box"], d["feats"][..., 0], d["valid"])], -1)
            if self.glayers:                                                     # shared hidden layer, one output head per layer
                H = self.gb(pf); h = self.heads
                for l, blk in enumerate(self.blocks):
                    bl = F.linear(H, self.gbl.weight[l*h:(l+1)*h], self.gbl.bias[l*h:(l+1)*h]).permute(0, 3, 1, 2)
                    x = blk.forward_b(x, cs, (am + bl).to(dt).contiguous())
                return self.nf(x)
            am = am + self.gb(pf).permute(0, 3, 1, 2)
        am = am.to(dt).contiguous()
        for blk in self.blocks: x = blk.forward_b(x, cs, am)
        return self.nf(x)
    def train_loss(self, d):
        """Sum over the boards of loss_fn (mean over each board's parts), in one padded batch: part rows x (text tokens + none)."""
        with nv("trunk"): x = self.dense(d)
        B, N, D = x.shape; P = d["tm"].shape[1]; hp = x[:, :P]; n = d["n"][:, None]
        L = torch.cat([self.wq(hp) @ self.wk(x).transpose(1, 2) / math.sqrt(D), self.none(hp)], 2).float()      # (B, P, N+1)
        col = torch.cat([(torch.arange(N, device=x.device)[None] >= n) & d["valid"], d["valid"].new_ones(B, 1)], 1)   # text tokens + none
        L = L.masked_fill(~col[:, None, :], float("-inf"))
        per = torch.logsumexp(L, 2) - torch.logsumexp(L.masked_fill(~d["tm"], float("-inf")), 2)                # padded part rows: 0
        if "w" in d: per = per * d["w"]
        pb = per.sum(1) / d["n"]; lk = pb.sum(); loss = lk; lc_ = lt_ = lk.new_zeros(()); cb = None   # pb / cb: per-board link / class loss
        rowv = torch.arange(P, device=x.device)[None] < n; nonet = d["tm"][..., N] & rowv                      # part rows; rows whose target is none
        lk_none = ((per * nonet).sum(1) / d["n"]).sum(); lk_lab = lk - lk_none                                  # link loss on none / labelled parts
        lse = torch.logsumexp(L, 2).masked_fill(~rowv, 0.0); lse_b = (lse.sum(1) / d["n"]).sum()               # link-logit logsumexp (log)
        zl = ((lse**2).sum(1) / d["n"]).sum()                                                                    # z-loss term (link logits)
        if self.arhead:                                                          # causal head, teacher forcing in base-confidence order
            la = lk.new_zeros(()); Nv = d["valid"].sum(1)
            for b in range(B):
                n_, nb = int(d["n"][b]), int(Nv[b])
                if n_ == 0 or nb <= n_: continue
                base = torch.cat([L[b, :n_, n_:nb], L[b, :n_, N:N+1]], 1); tm = torch.cat([d["tm"][b, :n_, n_:nb], d["tm"][b, :n_, N:N+1]], 1); M = nb - n_
                order = torch.argsort(torch.softmax(base.detach(), 1).max(1).values, descending=True)
                bo, to = base[order], tm[order]
                ch = torch.where(to[:, :M].any(1), bo[:, :M].detach().masked_fill(~to[:, :M], float("-inf")).argmax(1), torch.full_like(order, M))
                La = self.ar.seq_logits(hp[b, order].float(), x[b, n_:nb].float(), bo, ch, protect=to)
                la = la + (torch.logsumexp(La, 1) - torch.logsumexp(La.masked_fill(~to, float("-inf")), 1)).mean()
            loss = (la if AR_ONLY else loss + la); lt_ = la                     # logged in the 3rd loss slot
            self._parts = torch.stack([lk, lc_, lt_]).detach(); return loss
        if self.clshead and "pc" in d:                                           # + class loss per part (mean over the board's parts)
            if self.clsmode == "sel":                                            # selected link: gt = first target text (none if the none column
                tm = d["tm"]; gt = torch.where(tm[..., -1], -1, tm[..., :-1].int().argmax(2))   # is a target); pred = argmax of the link logits; -1 = the none column (not N: --compile keeps N dynamic)
                pr = L.detach().argmax(2); pr = torch.where(pr == N, -1, pr)
                use_pred = self.cls_link == "pred" or (self.cls_link == "mix" and self.cls_frac >= 0.5)
                jt = torch.where(torch.rand(pr.shape, device=pr.device) < 0.5, pr, gt) if self.cls_link == "mix" and use_pred else (pr if use_pred else gt)
                lc = self.cls_sel(x, d["box"], d["pos"], d["n"], jt)
            else: lc = self.cls(hp).float() if self.clsmode == "parallel" else self.class_through_link(torch.softmax(L, 2), hp, x)
            ce = F.cross_entropy(lc.transpose(1, 2), d["pc"], ignore_index=-1, reduction="none")
            cb = ce.sum(1) / d["n"]; lc_ = cb.sum(); loss = loss + CLS_W * lc_
            zl = zl + ((torch.logsumexp(lc, -1).masked_fill(~rowv, 0.0)**2).sum(1) / d["n"]).sum()             # z-loss term (class logits)
            if self.clsmode in ("mix", "pool") and "tc" in d:                    # + class named by each text (mean over the board's texts)
                ct = F.cross_entropy(self.tcls(x).float().transpose(1, 2), d["tc"], ignore_index=-1, reduction="none")
                nt = (d["tc"] >= 0).sum(1).clamp_min(1); lt_ = (ct.sum(1) / nt).sum(); loss = loss + TCLS_W * lt_
        if ZLOSS: loss = loss + ZLOSS * zl                                       # --zloss
        self._parts = torch.stack([lk, lc_, lt_, lk_lab, lk_none, zl, lse_b]).detach()   # unweighted terms for the log (PART_TERMS)
        if self.board_log: self._board_loss = torch.stack([pb, cb if cb is not None else torch.zeros_like(pb)], 1).detach()   # --gradlog
        return loss
    def class_through_link(self, A, hp, x):
        """mix: p_i(c) = sum_j a_ij q_j(c) + r_i(c) (a_i,none + sum_j a_ij q_j(not a designator)) -> log p;
        pool: logits = MLP([h_i ; sum_j a_ij h_j ; a_i,none]). A: link probabilities over tokens (non-text columns 0) + none."""
        At, an = A[..., :-1], A[..., -1:]
        if self.clsmode == "mix":
            q = torch.softmax(self.tcls(x).float(), -1); r = torch.softmax(self.fcls(hp).float(), -1)
            return torch.log((At @ q[..., :9] + r * (an + At @ q[..., 9:])).clamp_min(1e-9))
        return self.cmlp(torch.cat([hp, (At.to(x.dtype) @ x), an.to(x.dtype)], -1)).float()
    def head_out(self, x, n):
        hp, ht = x[:n], x[n:]
        S = self.wq(hp) @ self.wk(ht).T / math.sqrt(x.shape[1])
        if self.head == "lg": return S, self.ms(hp).squeeze(1), self.mt(ht).squeeze(1)
        return torch.cat([S, self.none(hp)], 1)                     # (n, m+1), last column = none

def loss_fn(logits, tmask):
    """Per-part softmax over texts + none; -log of the probability mass on the target cells (same math as the old per-part loop)."""
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~tmask, float("-inf")), 1)).mean()

def lg_logp(out):
    """log P_ij = log sigma_i + log tau_j + log softmax_row(S)_ij + log softmax_col(S)_ij; and log(1 - sigma_i)."""
    S, ls, lt = (o.float() for o in out)
    return F.logsigmoid(ls)[:, None] + F.logsigmoid(lt)[None, :] + F.log_softmax(S, 1) + F.log_softmax(S, 0), F.logsigmoid(-ls)

def loss_lg(out, tmask, ign):
    """-log P on true pairs (several same-string texts: their sum), plus -log(1 - sigma) on parts without a label and -log(1 - tau)
    on texts that name no part, each half-weighted and averaged over its own count (balanced, as LightGlue). 'ign' parts are skipped."""
    lp, lnone = lg_logp(out); m = lp.shape[1]; tm = tmask[:, :m]; ign = torch.as_tensor(ign, dtype=torch.bool, device=lp.device)
    has = tm.any(1) & ~ign; nolab = ~tm.any(1) & ~ign; named = (tm & ~ign[:, None]).any(0)
    mean = lambda t: t.mean() if t.numel() else lp.new_zeros(())
    pos = -torch.logsumexp(lp[has].masked_fill(~tm[has], float("-inf")), 1)      # rows selected first: no all -inf row in the graph
    return mean(pos) + 0.5*mean(-lnone[nolab]) + 0.5*mean(-F.logsigmoid(-out[2].float())[~named])

@torch.no_grad()
def decode(logits, mode="hung"):
    """Hungarian on -log p; each part has its own none column. lg head: cost -log P_ij, none cost -log(1 - sigma_i) ('hung'),
    or 'sig': parts with sigma_i >= 0.5 must take a text when one is free, the others take none."""
    if isinstance(logits, tuple):
        lp, lnone = lg_logp(logits); lp, lnone = lp.cpu().numpy(), lnone.cpu().numpy(); n, m = lp.shape
        C = np.full((n, m+n), 1e6); C[:, :m] = -lp
        C[np.arange(n), m+np.arange(n)] = -lnone if mode == "hung" else np.where(lnone < math.log(0.5), 1e3, 0.0)
        r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}
    lp = F.log_softmax(logits.float(), 1).cpu().numpy(); n, m1 = lp.shape; m = m1-1
    C = np.full((n, m+n), 1e6); C[:, :m] = -lp[:, :m]; C[np.arange(n), m+np.arange(n)] = -lp[:, m]
    r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}

def count(e, L):
    """Right = the chosen text is a target text (same string as the part's designator, or the explicit link); 'ign' parts skipped."""
    c = dict(linkable=0, right=0, links=0)
    for i, t in enumerate(e["tgt"]):
        if e.get("ign") and e["ign"][i]: continue
        j = L.get(i); c["links"] += j is not None
        if t: c["linkable"] += 1; c["right"] += j in t
    return c

def class_count(e, L, head):
    """Parts with a known class (0-8): class right from the class head, and from the prefix rule (class of the linked text's prefix;
    no link or an unknown prefix = no class)."""
    c = dict(cls_n=0, cls_head=0, cls_rule=0)
    for i, y in enumerate(e["pcls"].tolist()):
        if not 0 <= y <= 8: continue
        c["cls_n"] += 1; c["cls_head"] += int(head[i]) == y; j = L.get(i)
        m = PFX.match(e["strs"][j]) if j is not None else None
        c["cls_rule"] += bool(m) and PFX_CLS.get(m.group(1), -1) == y
    return c

def summarize(per):
    c = {k: sum(v[k] for v in per.values()) for k in ("linkable", "right", "links")}
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    out = dict(acc=round(acc, 4), prec=round(prec, 4), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), **c)
    for t in ("link", "cls"):                                                        # eval losses (mean per part), when evaluate() gave them
        nn_ = sum(v.get(f"loss_{t}_n", 0) for v in per.values())
        if nn_: out[f"loss_{t}"] = round(sum(v.get(f"loss_{t}_sum", 0.0) for v in per.values()) / nn_, 4)
    if any("cls_n" in v for v in per.values()):
        n = sum(v.get("cls_n", 0) for v in per.values())
        out.update(cls_head=round(sum(v.get("cls_head", 0) for v in per.values())/max(n, 1), 4), cls_rule=round(sum(v.get("cls_rule", 0) for v in per.values())/max(n, 1), 4), cls_n=n)
        if any("cls_gt" in v for v in per.values()): out["cls_gt"] = round(sum(v.get("cls_gt", 0) for v in per.values())/max(out["cls_n"], 1), 4)   # sel: true link
    return out

def to_dev(e, dev): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in e.items()}

AMP = False
def size_groups(es, ratio=2.0, budget=16384, cap=None):
    """Boards sorted by token count, cut into groups whose sizes differ <= ratio x and whose padded size (boards x largest) <= budget
    (and <= cap boards); each group is one padded forward."""
    out, cur = [], []
    for e in sorted(es, key=lambda e: e["n"]+e["m"]):
        N = e["n"]+e["m"]
        if cur and (N > ratio*(cur[0]["n"]+cur[0]["m"]) or (len(cur)+1)*N > budget or (cap and len(cur) >= cap)): out.append(cur); cur = []
        cur.append(e)
    return out + ([cur] if cur else [])

BATCH_EVAL = False                                    # set by --bps / --batched: evaluate size groups in padded forwards (same results)
def evaluate(model, encs, dev, mode="hung"):
    model.eval(); per = {}
    if BATCH_EVAL:
        for g in size_groups([e for e in encs if e["n"] and e["m"]], cap=64):
            ch = getattr(model, "clshead", False); sel = getattr(model, "clsmode", "") == "sel"   # sel: also the head with the true link
            gl = [{i: (t[0] if t and not (e.get("ign") and e["ign"][i]) else None) for i, t in enumerate(e["tgt"])} for e in g] if sel else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
                outs = model.forward_batch([to_dev(e, dev) for e in g], with_cls=True, gt_links=gl) if ch else model([to_dev(e, dev) for e in g])
            if ch: outs, cls = outs
            for k, (e, o) in enumerate(zip(g, outs)):
                L = decode(o, mode); per[e["id"]] = count(e, L)
                if o.dim() == 2 and o.shape[1] == e["m"] + 1:                      # dot head: eval link loss (as loss_fn) and class loss per part
                    tm_ = e["tmask"].to(o.device).bool(); ok_ = tm_.any(1) & torch.tensor([not (e.get("ign") and e["ign"][i]) for i in range(e["n"])], device=o.device)
                    ll = (torch.logsumexp(o.float(), 1) - torch.logsumexp(o.float().masked_fill(~tm_, float("-inf")), 1))[ok_]
                    per[e["id"]].update(loss_link_sum=float(ll.sum()), loss_link_n=int(ok_.sum()))
                    if ch:
                        y_ = e["pcls"].to(o.device); v_ = (y_ >= 0) & (y_ <= 8)
                        if v_.any(): per[e["id"]].update(loss_cls_sum=float(F.cross_entropy(cls[k][:e["n"]].float()[v_], y_[v_], reduction="sum")), loss_cls_n=int(v_.sum()))
                if ch: per[e["id"]].update(class_count(e, L, cls[k].argmax(1).cpu().numpy()))
                if sel: per[e["id"]]["cls_gt"] = class_count(e, L, model._cls_gt[k, :e["n"]].argmax(1).cpu().numpy())["cls_head"]
        for e in encs:
            if not (e["n"] and e["m"]): per[e["id"]] = count(e, {})
        model.train(); return summarize(per), per
    for e in encs:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
            per[e["id"]] = count(e, decode(model(to_dev(e, dev)), mode) if e["n"] and e["m"] else {})
    model.train(); return summarize(per), per

CFG_SECTIONS = {   # only for reading the written config files; the keys stay flat
    "run": ("tag", "seed", "device", "threads", "workers", "prefetch", "save", "init", "ckpt_every", "overwrite", "wandb", "wandb_group", "profile",
            "timing", "gradlog", "gradlog_spike", "gradlog_window", "roll_every", "roll_keep", "roll_from", "roll_optim", "stop_at", "min_free_gb"),
    "data": ("data", "train", "eval", "final_eval", "mix", "mix_p", "txtfeat", "txtfeat_wacv", "noise", "noise_dir", "imgfeat", "pimgfeat",
             "pimgfeat_wacv", "txtfeat_wacv_raw", "pimgfeat_wacv_raw", "fptype", "detscore"),
    "model": ("size", "tokens", "head", "pos", "nosize", "geobias", "rankrel", "strrel", "glayers", "local", "strtype", "arhead", "clshead",
              "clsmode", "clshead_tf", "qknorm"),
    "train": ("steps", "lr", "warmup", "cosine", "accum", "bps", "batched", "size_ratio", "tok_budget", "dropout", "wd", "amp", "compile",
              "fused_adam", "beta2", "zloss", "zclip", "zclip_alpha", "zclip_warmup", "freeze_backbone", "img_lr_mult", "cls_w", "tcls_w", "cls_link", "cls_detach_text", "none_w", "ign_messy"),
    "augment": ("aug", "aug_norot", "label_drop", "fake", "distract", "prefix_drop", "strnoise", "det_drop", "img_drop"),
    "eval": ("eval_every",)}

def parse_config_args(ap, argv=None):
    """--config PATH: YAML file of flat keys = the argparse dest names. Precedence: argparse defaults < config file < explicit flags. Unknown
    keys, values of the wrong type and values outside `choices` exit with an error (set_defaults alone checks neither). Returns (args, the
    resolved dict of every key, to be written as <tag>.config.yaml)."""
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", default=""); path = pre.parse_known_args(argv)[0].config
    if path:
        import yaml
        C = yaml.safe_load(open(path)) or {}
        if not isinstance(C, dict): ap.error(f"--config {path}: not a mapping of keys")
        acts = {x.dest: x for x in ap._actions if x.dest not in ("help", "config", "show_config")}
        bad = sorted(set(C) - set(acts))
        if bad: ap.error(f"--config {path}: unknown key(s) {bad}")
        for k, v in C.items():
            x = acts[k]
            if isinstance(x, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(v, bool): ap.error(f"--config {path}: {k} must be true / false, not {v!r}")
            elif x.type in (int, float):
                if isinstance(v, bool) or not isinstance(v, (int, float, str)): ap.error(f"--config {path}: {k} must be a number, not {v!r}")
                try: f = float(v)                          # YAML 1.1 reads 2e-4 as a string
                except ValueError: ap.error(f"--config {path}: {k} must be {x.type.__name__}, not {v!r}")
                if x.type is int and f != int(f): ap.error(f"--config {path}: {k} must be an integer, not {v!r}")
                C[k] = int(f) if x.type is int else f
            elif not isinstance(v, str): ap.error(f"--config {path}: {k} must be a string, not {v!r}")
            if x.choices is not None and C[k] not in x.choices: ap.error(f"--config {path}: {k} = {C[k]!r} not in {list(x.choices)}")
        ap.set_defaults(**C)
    a = ap.parse_args(argv)
    return a, {k: v for k, v in vars(a).items() if k not in ("config", "show_config")}

def dump_config(R, path):
    """Write the resolved keys as YAML, grouped under section comments (keys outside CFG_SECTIONS under '# other')."""
    import yaml
    done, out = set(), [f"# resolved LinkFormer config (tools/linkformer.py --config); flat keys = argparse dest names\n"]
    for sec, ks in list(CFG_SECTIONS.items()) + [("other", tuple(sorted(set(R) - {k for ks in CFG_SECTIONS.values() for k in ks})))]:
        part = {k: R[k] for k in ks if k in R and k not in done}; done |= set(part)
        if part: out.append(f"# {sec}\n" + yaml.safe_dump(part, sort_keys=False, default_flow_style=False))
    if str(path) == "-": print("".join(out), end=""); return
    Path(path).parent.mkdir(parents=True, exist_ok=True); open(path, "w").write("".join(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="YAML file of flat keys = these argparse dest names (configs/*.yaml); explicit flags override it; "
                    "the resolved config is written as <tag>.config.yaml next to the --save ckpt (or in results/linkformer/)")
    ap.add_argument("--show_config", action="store_true", help="print the resolved config (defaults < --config < flags) and exit")
    ap.add_argument("--prefetch", type=int, default=0, help="batched dot head: a background thread prepares this many optimizer steps ahead "
                    "(same boards in the same order, same groups; padded batches in pinned memory, host->device copies started); 0 = off")
    ap.add_argument("--gradlog", action="store_true", help="training: pre-clip grad norm per step (kept on the GPU, read every 250 steps): mean / median / "
                    "max / clipped share / spikes in the 250-step line; spike steps (norm > 3 x the median of the previous 1,000 steps) and the "
                    "top-5 steps of each 12.5k window with their boards (id, link / class loss, parts, texts) -> logs/linkformer/<tag>.spikes.jsonl")
    ap.add_argument("--gradlog_spike", type=float, default=3.0, help="--gradlog: spike = grad norm > this x the median of the previous 1,000 steps")
    ap.add_argument("--gradlog_window", type=int, default=12500, help="--gradlog: steps per window of the top-5 grad-norm rows")
    ap.add_argument("--roll_every", type=int, default=0, help="rolling ckpt <save>_roll<step>.pt every this many steps from --roll_from on; only the "
                    "last --roll_keep are kept (older ones deleted); 0 = off")
    ap.add_argument("--roll_keep", type=int, default=3); ap.add_argument("--roll_from", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=0.0, help="skip a ckpt save (with a DISK ALERT line) when its file system has less free space "
                    "than this (GB); 0 = off")
    ap.add_argument("--roll_optim", action="store_true", help="--roll_every: also save the AdamW and scheduler state (for a restart from that step)")
    ap.add_argument("--stop_at", type=int, default=0, help="stop training after this step (the lr schedule still runs over --steps); 0 = off")
    ap.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2 (0.999 = the torch default of all runs before 2026-10-23)")
    ap.add_argument("--zloss", type=float, default=0.0, help="weight of the z-loss: mean over parts of logsumexp^2 of the link logits (+ class logits)")
    ap.add_argument("--qknorm", action="store_true", help="LayerNorm on q and k per head in every attention block (before RoPE)")
    ap.add_argument("--zclip", type=float, default=0.0, help="ZClip: clip the grad norm to mean + this x std of its EMA (alpha --zclip_alpha) "
                    "instead of the fixed clip 1.0; the first --zclip_warmup steps use the fixed clip and collect the statistics; 0 = off")
    ap.add_argument("--zclip_alpha", type=float, default=0.97); ap.add_argument("--zclip_warmup", type=int, default=25)
    ap.add_argument("--timing", action="store_true", help="add wall-clock timing to the 250-step lines: s/step, data wait, batch build + host->device "
                    "copy, boards/s, peak GPU memory, eval time (no CUDA syncs added)")
    ap.add_argument("--train", default="dev3"); ap.add_argument("--eval", default="dev3"); ap.add_argument("--size", default="L")
    ap.add_argument("--tokens", default="E1"); ap.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--accum", type=int, default=1, help="boards per optimizer step"); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--aug", action="store_true")
    ap.add_argument("--save", default=""); ap.add_argument("--init", default="")
    ap.add_argument("--final_eval", default="", help="sets evaluated once at the end (per-image counts kept)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--tag", default="")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="", help="wandb project (empty = off)"); ap.add_argument("--wandb_group", default="")
    ap.add_argument("--data", default="", help="frozen dataset folder under linkdata/, e.g. CrossClass_data_v1")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA"); ap.add_argument("--ckpt_every", type=int, default=0)
    ap.add_argument("--mix", default="", help="second training set sampled with probability --mix_p (e.g. ktrain)"); ap.add_argument("--mix_p", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes for augmentation / encoding")
    ap.add_argument("--head", default="dot", choices=["dot", "lg"], help="dot = softmax over texts + none column; lg = LightGlue-style matchability")
    ap.add_argument("--geobias", action="store_true", help="attention bias from pair box geometry in the query box's own size")
    ap.add_argument("--rankrel", action="store_true", help="geometry bias also sees nearest-neighbour ranks between parts and texts (both directions, mutual nearest)")
    ap.add_argument("--strrel", action="store_true", help="geometry bias also sees string relations of text pairs: same prefix, number difference, consecutive")
    ap.add_argument("--glayers", action="store_true", help="geometry bias with its own output head per layer (shared hidden layer)")
    ap.add_argument("--local", type=float, default=0.0, help="local attention: each token sees only tokens within this many median part sizes (0 = whole board)")
    ap.add_argument("--imgfeat", default="", help="npz of frozen image features of the WACV-clean boxes (<image>__p / __t), e.g. linkdata/CrossClass_data_v1/wacv_clean_v1_dinov2b.npz")
    ap.add_argument("--img_lr_mult", type=float, default=20.0, help="learning-rate factor of the (zero-initialised) image adapter")
    ap.add_argument("--aug_norot", action="store_true", help="augment without 90-degree rotations / mirroring (image features do not rotate)")
    ap.add_argument("--freeze_backbone", action="store_true", help="with --imgfeat: train only the image adapter, every other weight fixed")
    ap.add_argument("--clshead", action="store_true", help="also predict each part's class (9 classes); eval reports cls_head and cls_rule")
    ap.add_argument("--cls_w", type=float, default=1.0, help="weight of the class loss")
    ap.add_argument("--clsmode", default="parallel", choices=["parallel", "mix", "pool", "sel"], help="class head: parallel = linear on the part token; "
                    "mix = link-weighted sum of per-text class predictions + fallback; pool = MLP on [part token, link-weighted text token, none prob]; "
                    "sel = W_in [part state; state of the SELECTED text or e_none; none flag] -> --clshead_tf blocks -> 9 (owner design 2026-10-23)")
    ap.add_argument("--clshead_tf", type=int, default=2, help="--clsmode sel: transformer blocks over the class tokens (trunk position encoding); 0 = MLP")
    ap.add_argument("--cls_detach_text", action="store_true", help="--clsmode sel, training: the selected text token enters the class token "
                    "detached (h_text(i).detach()): the class loss reaches the trunk only through the part token; values unchanged")
    ap.add_argument("--cls_link", default="gt", choices=["gt", "pred", "mix"], help="--clsmode sel, training: the selected text = the true link (gt), "
                    "the argmax of the link logits (pred), or gt for the first half of the steps then pred with p 0.5 (mix); eval / test: Hungarian")
    ap.add_argument("--tcls_w", type=float, default=0.5, help="mix / pool: weight of the per-text class loss")
    ap.add_argument("--distract", type=float, default=0.0, help="augment: non-designator texts (values, net names, numbers) per label")
    ap.add_argument("--noise", type=float, default=0.0, help="augment: real noise boxes per label (vias, print, non-component pads of the side; "
                    "boxes from <noise_dir>, PARSeq features from <noise_dir>feat); needs --txtfeat")
    ap.add_argument("--noise_dir", default="linkdata/kicad_noise")
    ap.add_argument("--detscore", default="", help="json pool of real detector class scores per true class (tools/det_score_pool.py, val9): part tokens get "
                    "a 9-d detector score vector, sampled from the pool for KiCad parts, the real one (preds_wacv_1280_cls9.json) for WACV parts")
    ap.add_argument("--det_drop", type=float, default=0.0, help="augment: share of KiCad parts whose detector scores are zero (no detector info)")
    ap.add_argument("--fptype", default="", help="dir of KiCad footprint types (tools/kicad_fptype.py): class target of a KiCad part = its footprint type "
                    "when known, else the designator-prefix class")
    ap.add_argument("--prefix_drop", type=float, default=0.0, help="augment: share of labels whose letters OCR lost ('C12' -> '12')")
    ap.add_argument("--strtype", action="store_true", help="string tokens tell digits-only and other text apart from unknown / empty")
    ap.add_argument("--txtfeat", default="", help="dir of PARSeq text-image features per KiCad side (tools/text_features.py), e.g. linkdata/kicad_textfeat")
    ap.add_argument("--txtfeat_wacv", default="", help="npz of PARSeq features of the WACV-clean texts, e.g. linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz")
    ap.add_argument("--batched", action="store_true", help="the boards of one optimizer step in one padded forward (same math, faster)")
    ap.add_argument("--bps", type=int, default=0, help="boards per optimizer step (overrides --accum; implies --batched): sorted by size, "
                    "cut into padded groups with sizes within --size_ratio and at most --tok_budget padded tokens, gradients accumulated")
    ap.add_argument("--size_ratio", type=float, default=2.0); ap.add_argument("--tok_budget", type=int, default=16384)
    ap.add_argument("--strnoise", action="store_true", help="training: 30 %% of text strings dropped, 15 %% corrupted like OCR (with --tokens E2)")
    ap.add_argument("--label_drop", type=float, default=0.15, help="augment: share of labels dropped (their parts become none)")
    ap.add_argument("--fake", type=float, default=0.05, help="augment: fake designator-like labels per text (name no part)")
    ap.add_argument("--none_w", type=float, default=1.0, help="loss weight of parts whose target is none (batched dot head)")
    ap.add_argument("--ign_messy", action="store_true", help="no loss on parts whose label is printed nearer another part (KiCad train)")
    ap.add_argument("--arhead", action="store_true", help="add the causal (autoregressive) link head (ARHead); with --freeze_backbone only it is trained")
    ap.add_argument("--compile", action="store_true", help="torch.compile the padded training forward + loss (dynamic shapes; single GPU)")
    ap.add_argument("--fused_adam", action="store_true", help="fused AdamW kernel (same update rule)")
    ap.add_argument("--profile", default="", help="'A:B': NVTX ranges per phase and a CUDA profiler range (nsys --capture-range=cudaProfilerApi) "
                    "over steps A..B-1, print s/step, then exit")
    ap.add_argument("--overwrite", action="store_true", help="run although results/linkformer/<tag>.json, the --save ckpt or its _step ckpts exist")
    ap.add_argument("--pos", default="rope", choices=["rope", "abs", "none", "rope4"], help="position: rope = 2D RoPE in attention (default, all "
                    "runs before 2026-10-23); abs = no RoPE, SuperGlue-style MLP of the normalised box centre added at the input; none = no RoPE, no "
                    "abs; rope4 = 4D corner RoPE (x1, y1, x2, y2; 4 blocks of 16 dims per head, 8 frequencies, wavelengths 0.5-512 median parts)")
    ap.add_argument("--nosize", action="store_true", help="token geometry input = token type only (no log w, log h, w >= h); with --pos rope4")
    ap.add_argument("--pimgfeat", default="", help="dir of part image features per KiCad side, <id>.npz {feat (n parts, D), ok (n)} in kicad.json "
                    "order (tools/part_roi.py render); part tokens get MLP(feat) or the learned no-image vector")
    ap.add_argument("--pimgfeat_wacv", default="", help="npz of part image features of the WACV-clean boards, one (n parts, D) array per board id "
                    "(tools/part_roi.py photo: data/partfeat_photo/wacv_annot.npz)")
    ap.add_argument("--txtfeat_wacv_raw", default="", help="npz {board id: (texts, 688)} PARSeq features of the RAW WACV boards (no \"clean\" key: dev3 / all45 / test42; tools/text_features.py boxes); eval only (2026-10-23)")
    ap.add_argument("--pimgfeat_wacv_raw", default="", help="npz {board id: (parts, D)} part image features of the RAW WACV boards (tools/part_roi.py boxes); eval only (2026-10-23)")
    ap.add_argument("--img_drop", type=float, default=0.1, help="training: share of part images replaced by the no-image vector (with --pimgfeat)")
    a, CFG = parse_config_args(ap)
    if a.show_config: dump_config(CFG, "-"); sys.exit(0)
    torch.set_num_threads(a.threads)
    if a.strnoise: STRNOISE = (0.30, 0.15)
    AUGP, NONE_W, IGN_MESSY, AUG_ROT = (a.label_drop, a.fake), a.none_w, a.ign_messy, not a.aug_norot
    DISTRACT, PREFIX_DROP, CLS_W, TCLS_W, NOISE, DET_DROP = a.distract, a.prefix_drop, a.cls_w, a.tcls_w, a.noise, a.det_drop; ZLOSS = a.zloss
    assert not a.noise or a.txtfeat, "--noise needs --txtfeat (noise boxes carry only a text-image feature)"
    if a.bps: a.accum, a.batched = a.bps, True
    if a.batched: BATCH_EVAL = True
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.tag = a.tag or f"{a.train}_{a.size}_{a.tokens}"     # the tag is known from the start and saved in the args (owner 2026-10-22)
    old = [ROOT/"results/linkformer"/f"{a.tag}.json"] + ([Path(a.save)] + sorted(Path(a.save).parent.glob(Path(a.save).stem + "_step*.pt")) if a.save else [])
    old = [str(p) for p in old if p.exists()]
    if old and not a.overwrite: sys.exit(f"[{a.tag}] refused at start: {', '.join(old)} exist(s); choose another --tag / --save or pass --overwrite")
    CFG["tag"] = a.tag
    META = {}
    if is_main:                                       # run identity + full-md5 fingerprints of every input this run reads (tools/run_meta.py)
        import run_meta as RM
        META = RM.start()
        cfg_out = (Path(a.save).parent if a.save else ROOT/"results/linkformer")/f"{a.tag}.config.yaml"; dump_config(CFG, cfg_out)
        META["config"] = dict(file=a.config or None, md5=a.config and RM.md5_file(a.config), resolved=str(cfg_out))
        ins = dict(data=ROOT/"linkdata"/a.data) if a.data else {f"data:{f}": ROOT/"linkdata"/f for f in ("wacv.json", "folds.json", "kicad199.json", "kicad_all.json")}
        ins.update(txtfeat=a.txtfeat and ROOT/a.txtfeat, txtfeat_wacv=a.txtfeat and a.txtfeat_wacv and ROOT/a.txtfeat_wacv,
                   noise_dir=a.noise and ROOT/a.noise_dir, noise_feat=a.noise and ROOT/(a.noise_dir.rstrip("/")+"feat"), fptype=a.fptype and ROOT/a.fptype,
                   detscore=a.detscore and ROOT/a.detscore, detscore_preds=a.detscore and ROOT/"data/preds_wacv_1280_cls9.json",
                   imgfeat=a.imgfeat and ROOT/a.imgfeat, init=a.init and Path(a.init),
                   pimgfeat=a.pimgfeat and ROOT/a.pimgfeat, pimgfeat_wacv=a.pimgfeat_wacv and ROOT/a.pimgfeat_wacv,
                   txtfeat_wacv_raw=a.txtfeat_wacv_raw and ROOT/a.txtfeat_wacv_raw, pimgfeat_wacv_raw=a.pimgfeat_wacv_raw and ROOT/a.pimgfeat_wacv_raw)
        META["inputs"] = RM.fingerprints(ins); META["fingerprint_seconds"] = round(sum(v["sec"] for v in META["inputs"].values()), 1)
        print(f"[{a.tag}] fingerprints {META['fingerprint_seconds']} s: " + ", ".join(f"{k} {v['sec']} s" for k, v in META["inputs"].items()), flush=True)
    sets = load_sets(a.data); AMP = a.amp
    a.imgdim = 0; a.txtdim = 0
    if a.txtfeat:                                     # PARSeq features: KiCad sides from <dir>/<side id>.npy, WACV-clean from --txtfeat_wacv
        D = ROOT/a.txtfeat; nk = nw = 0
        for name in list(sets):
            keep = []; nc = META.setdefault("board_filter", {})[name] = dict(boards=len(sets[name]), dropped_no_feat=0, dropped_len=0)
            for b in sets[name]:
                if "clean" in b: keep.append(b); continue
                if b.get("source") != "kicad": keep.append(b); continue
                f = D/f"{b['id']}.npy"
                if not f.exists(): nc["dropped_no_feat"] += 1; continue               # side not rendered / not aligned: left out
                F_ = np.load(f)
                if len(F_) != len(b["texts"]): nc["dropped_len"] += 1; continue
                if "tf" not in b["texts"][0]:
                    for t, v in zip(b["texts"], F_): t["tf"] = v.astype(np.float32)
                    nk += 1
                if a.noise and "noise" not in b:                                      # noise boxes + features (fp16), sampled in augment()
                    fb, ff = ROOT/a.noise_dir/f"{b['id']}.json", ROOT/(a.noise_dir.rstrip("/")+"feat")/f"{b['id']}.npy"
                    b["noise"] = []
                    if fb.exists() and ff.exists():
                        nb_, nf_ = json.load(open(fb))["boxes"], np.load(ff)
                        if len(nb_) == len(nf_): b["noise"] = [dict(box=bx, text="", angle=None, noise=True, tf=v) for bx, v in zip(nb_, nf_)]
                keep.append(b)
            sets[name] = keep; nc["kept"] = len(keep)
        if a.txtfeat_wacv:
            Z = np.load(ROOT/a.txtfeat_wacv); seen = set()
            for bs in sets.values():
                for b in bs:
                    if "clean" in b and f"{b['id']}__t" in Z and id(b) not in seen:
                        seen.add(id(b)); nw += 1
                        for t, v in zip(b["texts"], Z[f"{b['id']}__t"]): t["tf"] = v.astype(np.float32)
        if a.txtfeat_wacv_raw:                        # raw WACV boards (no "clean" key): their own PARSeq features, eval only
            Zr = np.load(ROOT/a.txtfeat_wacv_raw); seen = set(); nr = [0, 0]
            for bs in sets.values():
                for b in bs:
                    if "clean" in b or b.get("source") == "kicad" or id(b) in seen: continue
                    seen.add(id(b)); ok_ = b["id"] in Zr.files and len(Zr[b["id"]]) == len(b["texts"]); nr[ok_] += 1
                    if ok_:
                        for t, v in zip(b["texts"], Zr[b["id"]]): t["tf"] = v.astype(np.float32)
            if is_main: print(f"[{a.tag}] raw WACV text features: {nr[1]} boards, {nr[0]} without (missing / count mismatch)", flush=True)
        TXT_DIM = a.txtdim = 688
        if a.fake or a.distract: print(f"[{a.tag}] WARNING: fake / distractor texts have no image feature (zeros) -> set --fake 0 --distract 0", flush=True)
        if is_main: print(f"[{a.tag}] text features: {nk} KiCad sides, {nw} WACV boards; sets now " + ", ".join(f"{k} {len(v)}" for k, v in sets.items() if k in ("ktrain", "kval", "kvalsub")), flush=True)
        if is_main and a.noise:
            tr = sets.get(a.train, []); print(f"[{a.tag}] noise boxes: {sum(bool(b.get('noise')) for b in tr)} of {len(tr)} train sides, "
                                               f"{sum(len(b.get('noise', [])) for b in tr)} boxes, {a.noise} per label in augment", flush=True)
    if a.fptype:                                      # class target of KiCad parts from the footprint name (not the designator prefix)
        nch = nk_ = seen = 0; done = set()
        for bs in sets.values():
            for b in bs:
                if b.get("source") != "kicad" or id(b) in done: continue
                done.add(id(b)); f = ROOT/a.fptype/f"{b['id']}.json"
                if not f.exists(): continue
                ft = json.load(open(f))["fptype"]
                if len(ft) != len(b["parts"]): continue
                seen += 1
                for pp, t in zip(b["parts"], ft):
                    if t >= 0: nk_ += 1; nch += int(t != pp.get("cls", -1)); pp["cls"] = t
        if is_main: print(f"[{a.tag}] footprint types: {seen} KiCad sides, {nk_} parts typed, class changed for {nch}", flush=True)
    a.detdim = 0
    if a.detscore:                                    # detector class scores on part tokens
        J = json.load(open(ROOT/a.detscore)); DET_POOL = {int(c): np.array([v[1:10] for v in vs], np.float32) for c, vs in J["pool"].items() if vs}
        DET_POOL["all"] = np.concatenate([v for v in DET_POOL.values()]); DET_DIM = a.detdim = 9
        P9 = json.load(open(ROOT/"data/preds_wacv_1280_cls9.json")); done = set(); nw_ = nm_ = 0
        for name, bs in sets.items():
            for b in bs:
                if id(b) in done: continue
                done.add(id(b))
                if b.get("source") == "kicad":       # eval copies: one fixed sample per part (training re-samples in augment)
                    r_ = random.Random(b["id"])
                    for pp in b["parts"]: pp["ds"] = det_sample(pp.get("cls", -1), r_)
                elif b["id"] in P9:                  # WACV photo: the real detector scores of the best-overlapping detection (score >= 0.25)
                    D = [d for d in P9[b["id"]] if d[4] >= 0.25 and d[6] is not None]
                    if not D: continue
                    db = np.array([d[:4] for d in D], float)
                    for pp in b["parts"]:
                        x0, y0, x1, y1 = pp["box"]; iw = np.clip(np.minimum(x1, db[:, 2]) - np.maximum(x0, db[:, 0]), 0, None); ih = np.clip(np.minimum(y1, db[:, 3]) - np.maximum(y0, db[:, 1]), 0, None)
                        io = iw*ih / ((x1-x0)*(y1-y0) + (db[:, 2]-db[:, 0])*(db[:, 3]-db[:, 1]) - iw*ih + 1e-9); k_ = int(io.argmax()); nw_ += 1
                        if io[k_] >= 0.5: pp["ds"] = np.array(D[k_][6:15], np.float32); nm_ += 1
        if is_main: print(f"[{a.tag}] detector scores: pool {', '.join(f'{c}:{len(v)}' for c, v in DET_POOL.items() if c != 'all')}; WACV parts matched {nm_} of {nw_}", flush=True)
    if a.imgfeat:                                     # before the DataLoader workers fork: attach image features to the WACV-clean boxes (boards with the "clean" key, same order as the npz)
        Z = np.load(ROOT/a.imgfeat); IMG_DIM = a.imgdim = int(next(iter(Z.values())).shape[1]); seen = set()
        for bs in sets.values():
            for b in bs:
                if "clean" in b and f"{b['id']}__p" in Z and id(b) not in seen:
                    seen.add(id(b))
                    for o, v in zip(b["parts"], Z[f"{b['id']}__p"]): o["img"] = v.astype(np.float32)
                    for o, v in zip(b["texts"], Z[f"{b['id']}__t"]): o["img"] = v.astype(np.float32)
        if is_main: print(f"[{a.tag}] image features {a.imgfeat}: dim {IMG_DIM}, {len(seen)} boards", flush=True)
    a.pimgdim = 0
    if a.pimgfeat or a.pimgfeat_wacv or a.pimgfeat_wacv_raw:   # part image features, before the DataLoader workers fork (kept float16; encode() casts)
        Zw = np.load(ROOT/a.pimgfeat_wacv) if a.pimgfeat_wacv else None; done = set(); dims = set()
        Zr2 = np.load(ROOT/a.pimgfeat_wacv_raw) if a.pimgfeat_wacv_raw else None   # raw WACV boards (no "clean" key), eval only
        pc = dict(boards=0, kicad_sides=0, kicad_len_mismatch=0, wacv_clean_boards=0, wacv_raw_boards=0, parts=0, parts_with_image=0)
        for bs in sets.values():
            for b in bs:
                if id(b) in done: continue
                done.add(id(b)); pc["boards"] += 1; pc["parts"] += len(b["parts"]); F_ = O_ = None
                if b.get("source") == "kicad" and a.pimgfeat and (ROOT/a.pimgfeat/f"{b['id']}.npz").exists():
                    Z = np.load(ROOT/a.pimgfeat/f"{b['id']}.npz"); F_, O_ = Z["feat"], Z["ok"]; key = "kicad_sides"
                elif "clean" in b and Zw is not None and b["id"] in Zw.files:
                    F_ = Zw[b["id"]]; O_ = np.ones(len(F_), bool); key = "wacv_clean_boards"
                elif "clean" not in b and b.get("source") != "kicad" and Zr2 is not None and b["id"] in Zr2.files:
                    F_ = Zr2[b["id"]]; O_ = np.ones(len(F_), bool); key = "wacv_raw_boards"
                if F_ is None: continue
                if len(F_) != len(b["parts"]): pc["kicad_len_mismatch"] += 1; continue
                pc[key] += 1; dims.add(F_.shape[1])
                for pp, v, o in zip(b["parts"], F_, O_):
                    if o: pp["pimg"] = v; pc["parts_with_image"] += 1
        assert len(dims) == 1, f"--pimgfeat: feature dims {dims} (none found or mixed)"
        PIMG_DIM = a.pimgdim = dims.pop(); META["pimg"] = pc
        if is_main: print(f"[{a.tag}] part image features: dim {PIMG_DIM}, {pc}; img_drop {a.img_drop}", flush=True)
    if a.init:
        ck = torch.load(a.init, map_location="cpu"); vocab = Vocab(ck["vocab"])
    else:
        vocab = Vocab.from_boards(sets["all45"] + sets.get("ktrain", []) + sets.get("kval", []) + sets.get("kicadtrain", []) + sets.get("kicadval", []) + sets.get("kicad199", []))
        if a.strtype: vocab = Vocab(vocab.prefixes + ["<NUM>", "<TXT>"])          # string types: digits only / other text
    trb = [b for b in sets[a.train] if b["parts"] and b["texts"]]
    mixb = [b for b in sets[a.mix] if b["parts"] and b["texts"]] if a.mix else []
    # training stream first (DataLoader workers fork before CUDA is initialised)
    loader = torch.utils.data.DataLoader(BoardStream(trb, mixb, a.mix_p, vocab, a.aug, a.seed*100+rank), batch_size=None, collate_fn=lambda x: x,
                                         num_workers=a.workers, prefetch_factor=(8 if a.workers else None), persistent_workers=a.workers > 0)
    stream = iter(loader)
    evs = {s: [encode(b, vocab) for b in sets[s] if b["parts"] and b["texts"]] for s in a.eval.split(",") if s} if is_main else {}
    if world > 1:
        import datetime, torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2)); torch.cuda.set_device(local); dev = torch.device("cuda", local)
    else:
        dev = torch.device(a.device)
    d, L = SIZES[a.size]
    assert not a.strrel or (a.tokens == "E2" and a.geobias), "--strrel needs --tokens E2 and --geobias"; assert not a.glayers or a.geobias; assert not a.rankrel or a.geobias, "--rankrel needs --geobias"
    model = LinkFormer(d, L, len(vocab.idx)+2, a.tokens == "E2", a.dropout, a.head, a.geobias, a.strrel, a.glayers, a.local, a.imgdim, a.clshead, a.txtdim, a.clsmode, a.rankrel, a.detdim, a.arhead, pos=a.pos, nosize=a.nosize, pimgdim=a.pimgdim, clshead_tf=a.clshead_tf, qknorm=a.qknorm).to(dev)
    model.cls_link = a.cls_link; model.cls_detach_text = a.cls_detach_text
    model.img_drop = a.img_drop if a.pimgdim else 0.0
    if a.init:
        miss = model.load_state_dict(ck["model"], strict=not (a.imgdim or a.arhead))
        if a.imgdim and is_main: print(f"[{a.tag}] init {a.init}: new parameters {miss.missing_keys}", flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    if a.compile and world == 1 and a.batched and a.head == "dot": model.train_loss = torch.compile(model.train_loss, dynamic=True)
    acc = max(1, a.accum // world)                    # boards per rank per optimizer step; total = acc * world (= --accum)
    npar = sum(p.numel() for p in model.parameters()); tag = a.tag or f"{a.train}_{a.size}_{a.tokens}"
    if is_main: print(f"[{tag}] params {npar/1e6:.1f}M, train boards {len(trb)} ({sum(len(b['parts']) for b in trb)} parts), device {dev}, ranks {world} x {acc} boards/step, init {a.init or '-'}", flush=True)
    if a.freeze_backbone:                             # train only the image adapter: does the image add information to a fixed linker?
        keep = "ar." if a.arhead else "img."; AR_ONLY = bool(a.arhead)
        for n_, p in model.named_parameters(): p.requires_grad = n_.startswith(keep)
    groups = [dict(params=[p for n_, p in model.named_parameters() if not n_.startswith("img.")])]
    if a.imgdim: groups.append(dict(params=list(model.img.parameters()), lr=a.lr*a.img_lr_mult))
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd, betas=(0.9, a.beta2), fused=a.fused_adam and dev.type == "cuda")
    def lr_at(s):
        w = min(1.0, (s+1)/a.warmup)
        return w * (0.1 + 0.9*0.5*(1+math.cos(math.pi*min(s, a.steps)/a.steps)) if a.cosine else w)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    wb = None
    if a.wandb and is_main:
        try:
            import wandb
            kw = dict(project=a.wandb, group=a.wandb_group or None, name=tag, config=dict(vars(a), params=npar, train_boards=len(trb), ranks=world))
            try: wb = wandb.init(**kw)
            except Exception as ex: print(f"[{tag}] wandb online failed ({repr(ex)[:80]}), logging offline", flush=True); wb = wandb.init(mode="offline", **kw)
        except Exception as ex: print(f"[{tag}] wandb unavailable: {repr(ex)[:80]}", flush=True)
    if is_main: META.update(wandb=dict(project=a.wandb, id=wb.id) if wb else None, gpu=RM.gpu(dev))
    from contextlib import nullcontext
    log = []; t0 = time.time(); run = None; runp = None; TM = dict(data=0.0, batch=0.0, eval=0.0, groups=0, tok_real=0, tok_pad=0); tw = time.time()   # --timing
    PQ = None
    ZC = {}                                           # --zclip state (GPU tensors)
    ROLL = []                                         # --roll_every: paths of the kept rolling ckpts
    GL = None                                         # --gradlog (batched dot head): per-step grad norm + per-board losses, read every 250 steps
    if a.gradlog and a.batched and a.head == "dot":
        GL = dict(bl=[], meta=[], steps=[], hist=[], win=[]); model.board_log = True
        GLF = ROOT/"logs/linkformer"/f"{tag}.spikes.jsonl"; GLF.parent.mkdir(parents=True, exist_ok=True)
        def gradlog_read():
            """One host read for the steps since the last call: stats for the 250-step line; spike rows and the 12.5k top-5 to GLF."""
            st = GL["steps"]; GL["steps"] = []
            if not st: return ""
            norms = torch.stack([x[2] for x in st]).float().cpu().numpy(); bls = [x[3].float().cpu().numpy() for x in st]; spikes = 0
            with open(GLF, "a") as fo:
                for (s_, lr_, _, _, meta), g_, bl in zip(st, norms, bls):
                    prev = GL["hist"][-1000:]; med = float(np.median(prev)) if len(prev) >= 100 else None; GL["hist"].append(float(g_))
                    row = dict(step=s_, norm=round(float(g_), 4), median=med and round(med, 4), lr=lr_,
                               boards=[dict(id=i_, link=round(float(l_[0]), 4), cls=round(float(l_[1]), 4), parts=n_, texts=m_) for (i_, n_, m_), l_ in zip(meta, bl)])
                    if med and g_ > a.gradlog_spike*med: spikes += 1; fo.write(json.dumps(dict(kind="spike", **row)) + "\n")
                    GL["win"].append((float(g_), row)); GL["win"] = sorted(GL["win"], key=lambda r: -r[0])[:5]
                    if s_ % a.gradlog_window == 0:
                        for _, r in GL["win"]: fo.write(json.dumps(dict(kind="top5", window_end=s_, **r)) + "\n")
                        GL["win"] = []
                GL["hist"] = GL["hist"][-1000:]
            return (f", grad norm mean {norms.mean():.3f} median {np.median(norms):.3f} max {norms.max():.3f} clipped {100*(norms > 1.0).mean():.1f} % "
                    f"spikes {spikes}")
    if a.prefetch and a.batched and a.head == "dot":  # producer thread: next steps' boards -> groups -> padded batches on the GPU (pinned copies)
        import threading, queue
        PQ = queue.Queue(maxsize=a.prefetch)
        def producer():
            try:
                while True:
                    es_ = [next(stream) for _ in range(acc)]
                    PQ.put([([(e_["id"], e_["n"], e_["m"]) for e_ in g_], dense_batch(g_, dev, targets=True, pin=dev.type == "cuda"))
                            for g_ in (size_groups(es_, a.size_ratio, a.tok_budget) if a.bps else [es_])])   # (board meta, batch) per group
            except BaseException as ex: PQ.put(ex)
        threading.Thread(target=producer, daemon=True).start()
    pa, pb = map(int, a.profile.split(":")) if a.profile else (0, 0)
    ph = (lambda s: torch.cuda.nvtx.range(s)) if a.profile else (lambda s: nullcontext()); NVTX = bool(a.profile)
    for step in range(1, a.steps+1):
        model.cls_frac = (step - 1) / a.steps                                  # --cls_link mix: gt for the first half
        if a.profile and step in (pa, pb):
            torch.cuda.synchronize()
            if step == pa: torch.cuda.cudart().cudaProfilerStart(); tp = time.time()
            else: torch.cuda.cudart().cudaProfilerStop(); print(f"[{tag}] profile steps {pa}..{pb-1}: {(time.time()-tp)/(pb-pa):.4f} s/step", flush=True); sys.exit(0)
        opt.zero_grad(); tot = 0.0; parts = 0.0
        if a.batched:
            tq = time.time()
            with ph("data"): es = PQ.get() if PQ is not None else [next(stream) for _ in range(acc)]   # PQ: a list of ready batches
            TM["data"] += time.time() - tq
            if isinstance(es, BaseException): raise es
            groups = es if PQ is not None else size_groups(es, a.size_ratio, a.tok_budget) if a.bps else [es]
            for gi, g in enumerate(groups):
                with (net.no_sync() if world > 1 and gi < len(groups)-1 else nullcontext()):
                    with ph("forward"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                        if a.head == "dot":                                                       # one padded batch, loss included
                            tq = time.time()
                            with ph("batch_h2d"): db = g[1] if PQ is not None else dense_batch(g, dev, targets=True)
                            gm = g[0] if PQ is not None else [(e_["id"], e_["n"], e_["m"]) for e_ in g]   # (id, parts, texts) of the group
                            if GL is not None: GL["meta"] += gm
                            if a.timing: TM["groups"] += 1; TM["tok_real"] += sum(n_ + m_ for _, n_, m_ in gm); TM["tok_pad"] += len(gm) * (-(-max(n_ + m_ for _, n_, m_ in gm) // 16) * 16)
                            TM["batch"] += time.time() - tq
                            loss = net(db) / acc
                            if GL is not None: GL["bl"].append(model._board_loss)
                        else:
                            g = [from_numpy(e) for e in g]; outs = net([to_dev(e, dev) for e in g])
                            loss = sum(loss_lg(o, e["tmask"].to(dev), e["ign"]) for o, e in zip(outs, g)) / acc
                    with ph("backward"): loss.backward(); tot += loss.detach()                       # no host sync here
                    if getattr(model, "_parts", None) is not None: parts = parts + model._parts / acc
        for micro in (range(0) if a.batched else range(acc)):
            e = from_numpy(next(stream))
            with (net.no_sync() if world > 1 and micro < acc-1 else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                    logits = net(to_dev(e, dev))
                loss = (loss_lg(logits, e["tmask"].to(dev), e["ign"]) if a.head == "lg" else loss_fn(logits.float(), e["tmask"].to(dev))) / acc
                loss.backward(); tot += loss.item()
        with ph("optim"):
            if a.zclip and step > a.zclip_warmup:                               # ZClip: clip to EMA mean + z x EMA std, all on the GPU
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")); thr = ZC["mu"] + a.zclip * ZC["sd"]
                torch._foreach_mul_([p_.grad for p_ in model.parameters() if p_.grad is not None], torch.clamp(thr / (gn + 1e-6), max=1.0))
                gu = torch.minimum(gn, thr); ZC["mu"] = a.zclip_alpha * ZC["mu"] + (1 - a.zclip_alpha) * gu
                ZC["sd"] = torch.sqrt(a.zclip_alpha * ZC["sd"]**2 + (1 - a.zclip_alpha) * (gu - ZC["mu"])**2)
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if a.zclip:                                                       # warm-up: statistics of the first steps
                    ZC.setdefault("buf", []).append(gn.detach().float())
                    if step == a.zclip_warmup: b_ = torch.stack(ZC.pop("buf")); ZC.update(mu=b_.mean(), sd=b_.std().clamp_min(1e-6))
            if GL is not None: GL["steps"].append((step, sched.get_last_lr()[0], gn.detach(), torch.cat(GL["bl"]), GL["meta"])); GL["bl"], GL["meta"] = [], []
            opt.step(); sched.step()
        run = tot if run is None else 0.98*run + 0.02*tot                                             # a GPU tensor when batched
        if torch.is_tensor(parts): runp = parts if runp is None else 0.98*runp + 0.02*parts           # link / class / text-class parts
        if not is_main: continue
        lp = (lambda: dict(zip(PART_TERMS, [round(v, 4) for v in runp.tolist()]))) if runp is not None else (lambda: {})
        with ph("log"):
            if a.timing and step % 250 == 0:
                dt = time.time() - tw - TM["eval"]; tw = time.time()
                print(f"[{tag}] timing step {step}: {dt/250:.4f} s/step, data wait {1e3*TM['data']/250:.1f} ms, batch build + h2d {1e3*TM['batch']/250:.1f} ms, "
                      f"{a.accum*250/dt:.1f} boards/s, peak mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GiB, eval {TM['eval']:.1f} s, "
                      f"{TM['groups']/250:.2f} groups/step, padding {100*(1 - TM['tok_real']/max(TM['tok_pad'], 1)):.1f} %" if dev.type == "cuda" else "", flush=True)
                TM.update(data=0.0, batch=0.0, eval=0.0, groups=0, tok_real=0, tok_pad=0)
            if GL is not None and step % 250 == 0: print(f"[{tag}] gradlog step {step}" + gradlog_read(), flush=True)
            if not wb and step % 250 == 0: print(f"[{tag}] step {step} train_loss {float(run):.4f}" + "".join(f" {k} {v:.4f}" for k, v in lp().items()), flush=True)
        if wb and step % 50 == 0: wb.log(dict(loss=float(run), lr=sched.get_last_lr()[0], steps_per_s=step/(time.time()-t0)), step=step)
        if a.ckpt_every and a.save and step % a.ckpt_every == 0 and step < a.steps:
            with ph("ckpt"):
                Path(a.save).parent.mkdir(parents=True, exist_ok=True); cp_ = a.save.replace(".pt", f"_step{step}.pt")
                if disk_ok(cp_, a.min_free_gb, tag): torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a), step=step, meta=META), cp_)
        if a.roll_every and a.save and step >= a.roll_from and step % a.roll_every == 0:              # --roll_every: keep the last --roll_keep
            with ph("ckpt"):
                rp = a.save.replace(".pt", f"_roll{step}.pt"); Path(rp).parent.mkdir(parents=True, exist_ok=True)
                if disk_ok(rp, a.min_free_gb, tag): torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a), step=step, meta=META,
                                **(dict(opt=opt.state_dict(), sched=sched.state_dict()) if a.roll_optim else {})), rp); ROLL.append(rp)
                while len(ROLL) > a.roll_keep: os.remove(ROLL.pop(0))
        if evs and (step % a.eval_every == 0 or step == a.steps or step == a.stop_at):
            with ph("eval"): tq = time.time(); res = {s: evaluate(model, es, dev)[0] for s, es in evs.items()}; TM["eval"] += time.time() - tq
            log.append(dict(step=step, loss=round(float(run), 4), loss_parts=lp(), **res))
            if wb: wb.log({f"{s}/{m}": r[m] for s, r in res.items() for m in ("acc", "prec", "f1")}, step=step)
            print(f"[{tag}] step {step} loss {float(run):.3f} " + " ".join(f"{s}: acc {r['acc']:.3f} prec {r['prec']:.3f}" + (f" cls {r['cls_head']:.3f}/{r['cls_rule']:.3f}" + (f" gt-link {r['cls_gt']:.3f}" if "cls_gt" in r else "") if "cls_head" in r else "")
                    + (f" lossL {r['loss_link']:.4f}" if "loss_link" in r else "") + (f" lossC {r['loss_cls']:.4f}" if "loss_cls" in r else "") for s, r in res.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
        if a.stop_at and step >= a.stop_at: print(f"[{tag}] --stop_at {a.stop_at}: training stopped", flush=True); break
    if world > 1:
        import torch.distributed as dist
        if not is_main: dist.destroy_process_group(); sys.exit(0)
    final = {}
    for s in [x for x in a.final_eval.split(",") if x]:
        es = [encode(b, vocab) for b in sets[s] if b["parts"] and b["texts"]]
        for mode in (("hung", "sig") if a.head == "lg" else ("hung",)):
            key = s if mode == "hung" else f"{s}@{mode}"; summ, per = evaluate(model, es, dev, mode); final[key] = dict(summary=summ, per_image=per)
            print(f"[{tag}] final {key}: acc {summ['acc']:.4f} prec {summ['prec']:.4f} f1 {summ['f1']:.4f}" + (f" cls_head {summ['cls_head']:.4f} cls_rule {summ['cls_rule']:.4f} (n {summ['cls_n']})" + (f" cls_gt_link {summ['cls_gt']:.4f}" if "cls_gt" in summ else "") if "cls_head" in summ else ""), flush=True)
            if wb: wb.summary.update({f"final/{key}/{m}": summ[m] for m in ("acc", "prec", "f1")})
    META["end"] = RM.now()
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        if disk_ok(a.save, a.min_free_gb, tag): torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a), step=min(a.stop_at or a.steps, a.steps), meta=META), a.save)
    out = ROOT/"results/linkformer"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), params=npar, log=log, final=final, seconds=round(time.time()-t0), ranks=world, meta=META), open(out/f"{tag}.json", "w"), indent=1)
    if wb: wb.finish()
    if world > 1: dist.destroy_process_group()
