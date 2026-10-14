#!/usr/bin/env python3
"""Whole-board transformer linker (LINKING.md §10: input = two object sets, output = score matrix + none column -> Hungarian).
Tokens: one per part and one per text. Content: type, log width / height, orientation (E1); + prefix and digits of the string (E2).
Position (box centre, in units of the board's median part size) enters only through 2D rotary embeddings (RoPE) in self-attention.
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
Usage: linkformer.py --train dev3 --eval dev3 --size L --tokens E1 --steps 1000 [--device cuda] [--tag name]
  board sets: dev3, test42, all45, fold<k>train, fold<k>test, kicad199, kicadtrain, kicadval, kicad300, kicad1000;
  with --data CrossClass_data_v1: ktrain, kval, kvalsub (~25 % of kval), kdense (train sides with WACV density 2.1-11.2),
  clean_dev3 / clean_test41 / clean_all44 (WACV-clean v1, tools/wacv_clean.py)"""
import argparse, json, math, os, re, sys, time, hashlib, random
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
    def __call__(self, s): m = PFX.match(s); return self.idx.get(m.group(1), 1) if m else 1

def augment(b, rng, p_drop=0.15, p_fake=0.05, jitter=0.1):
    """A randomly transformed copy of a board: rotation by k*90 degrees and mirror, box-edge jitter (fraction of the box size),
    dropped labels (their parts become 'none'), fake designator-like labels that name no part."""
    k, mir = rng.randrange(4), rng.random() < 0.5
    def tf(box):
        x0, y0, x1, y1 = box
        if mir: x0, x1 = -x1, -x0
        for _ in range(k): x0, y0, x1, y1 = y0, -x1, y1, -x0          # rotate 90 degrees
        w, h = x1-x0, y1-y0
        return [x0+rng.uniform(-jitter, jitter)*w, y0+rng.uniform(-jitter, jitter)*h, x1+rng.uniform(-jitter, jitter)*w, y1+rng.uniform(-jitter, jitter)*h]
    parts = [dict(p, box=tf(p["box"])) for p in b["parts"]]
    texts = [dict(t, box=tf(t["box"])) for t in b["texts"] if rng.random() >= p_drop]
    if STRNOISE[0] or STRNOISE[1]:
        for t in texts:
            u = rng.random()
            if u < STRNOISE[0]: t["ocr"] = ""
            elif u < STRNOISE[0] + STRNOISE[1]: t["ocr"] = corrupt(t["text"], rng)
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
        mm = PFX.match(s)
        if mm:
            ds = mm.group(2)[-4:]; dig[n+j, 4-len(ds):] = [int(c) for c in ds]
    strs = [t["text"] for t in b["texts"]]
    tgt = []   # per part: list of target text indices, [] = none
    for p in b["parts"]: tgt.append(([p["link"]] if p["link"] is not None else []) if "link" in p else [j for j, s in enumerate(strs) if p["des"] and s == p["des"]])
    ign = [bool(p.get("ign")) for p in b["parts"]]
    tmask = np.zeros((n, m+1), bool)                 # target cells per part row (all same-string texts, or the none column)
    for i, t in enumerate(tgt): tmask[i, t if t else [m]] = True
    tmask[np.array(ign, bool)] = True                # unknown label: every cell is a target -> no loss
    if as_numpy:
        return dict(id=b["id"], n=n, m=m, feats=feats.astype(np.float32), pos=pos.astype(np.float32), pref=pref.astype(np.int64), dig=dig.astype(np.int64),
                    box=B.astype(np.float32), tmask=tmask, tgt=tgt, ign=ign, strs=strs, des=[p["des"] for p in b["parts"]])
    f = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(id=b["id"], n=n, m=m, feats=f(feats), pos=f(pos), pref=f(pref, torch.long), dig=f(dig, torch.long), box=f(B), tmask=torch.from_numpy(tmask),
                tgt=tgt, ign=ign, strs=strs, des=[p["des"] for p in b["parts"]])

def load_model(path, dev):
    """A saved checkpoint -> (model in eval mode, vocab, its training args)."""
    ck = torch.load(path, map_location="cpu"); A = ck["args"]; vocab = Vocab(ck["vocab"]); d, L = SIZES[A["size"]]
    m = LinkFormer(d, L, len(vocab.idx)+2, A["tokens"] == "E2", 0.0, A.get("head", "dot"), A.get("geobias", False))
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

def dense_batch(es, dev, targets=False):
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
    return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in out.items()}

def rope_cs(pos, freqs):
    """cos / sin of the 2D RoPE angles per axis, (..., 1, N, q): computed once per forward, shared by q, k and all layers."""
    return [(torch.cos(ang), torch.sin(ang)) for ang in ((pos[..., a:a+1] * freqs).unsqueeze(-3) for a in (0, 1))]

def rope_apply(x, cs):
    """Same rotation as rope_2d, with precomputed cos / sin."""
    hd = x.shape[-1]; out = []
    for (c, s), half in zip(cs, (x[..., :hd//2], x[..., hd//2:])):
        x1, x2 = half[..., 0::2], half[..., 1::2]; out += [torch.stack([x1*c - x2*s, x1*s + x2*c], -1).flatten(-2)]
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
    def __init__(self, d, heads, dropout=0.0):
        super().__init__(); self.h = heads; self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.p = dropout; self.drop = nn.Dropout(dropout)
        self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d); self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, pos, freqs, bias=None):
        N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(N, 3, self.h, d//self.h).permute(1, 2, 0, 3)
        q, k = rope_2d(q, pos, freqs), rope_2d(k, pos, freqs)
        am = bias[None].to(q.dtype) if bias is not None else None                  # (1, heads, N, N), added to the attention scores
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=am, dropout_p=self.p if self.training else 0.0)[0].permute(1, 0, 2).reshape(N, d)
        x = x + self.drop(self.o(a)); return x + self.drop(self.mlp(self.n2(x)))
    def forward_b(self, x, cs, am):
        """Padded batch: x (B, N, d), cs = rope_cs(pos), am (B, heads or 1, 1 or N, N) additive mask / bias (-inf on padded keys), already
        in the attention dtype."""
        B, N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d//self.h).permute(2, 0, 3, 1, 4)
        q, k = rope_apply(q, cs), rope_apply(k, cs)
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

class LinkFormer(nn.Module):
    def __init__(self, d, layers, n_prefix, use_string, dropout=0.0, head="dot", geobias=False):
        super().__init__(); self.use_string = use_string; hd = 64; heads = d // hd; self.head = head; self.geobias = geobias
        self.inp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.pref = nn.Embedding(n_prefix, d); self.dig = nn.ModuleList([nn.Embedding(11, d) for _ in range(4)])
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)]); self.nf = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d)
        if head == "lg": self.ms = nn.Linear(d, 1); self.mt = nn.Linear(d, 1)          # matchability of parts / texts
        else: self.none = nn.Linear(d, 1)
        if geobias: self.gb = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, heads))
        lam = torch.logspace(math.log10(0.5), math.log10(512), hd // 4)   # wavelengths from half a part to 512 parts
        self.register_buffer("freqs", 2*math.pi/lam)
    def forward(self, e):
        if isinstance(e, list): return self.forward_batch(e)
        if "tm" in e: return self.train_loss(e)
        x = self.inp(e["feats"])
        if self.use_string: x = x + self.pref(e["pref"]) + sum(emb(e["dig"][:, k]) for k, emb in enumerate(self.dig))
        bias = self.gb(pair_geometry(e["box"], e["feats"][:, 0])).permute(2, 0, 1) if self.geobias else None   # (heads, N, N)
        for blk in self.blocks: x = blk(x, e["pos"], self.freqs, bias)
        return self.head_out(self.nf(x), e["n"])
    def forward_batch(self, es):
        """Several boards in one padded batch (same math as one forward per board; padded tokens are masked out as keys)."""
        Ns = [e["feats"].shape[0] for e in es]; N = max(Ns); B = len(es)
        pad = lambda t, v=0: F.pad(t, (0, 0)*(t.dim()-1) + (0, N-t.shape[0]), value=v)
        valid = torch.zeros(B, N, dtype=torch.bool, device=es[0]["feats"].device)
        for b, n in enumerate(Ns): valid[b, :n] = True
        x = self.inp(torch.stack([pad(e["feats"]) for e in es])); pos = torch.stack([pad(e["pos"]) for e in es])
        if self.use_string:
            dig = torch.stack([pad(e["dig"], 10) for e in es])
            x = x + self.pref(torch.stack([pad(e["pref"]) for e in es])) + sum(emb(dig[..., k]) for k, emb in enumerate(self.dig))
        am = torch.zeros(B, 1, 1, N, device=x.device).masked_fill(~valid[:, None, None, :], float("-inf"))
        if self.geobias:
            am = am + torch.stack([F.pad(self.gb(pair_geometry(e["box"], e["feats"][:, 0])).permute(2, 0, 1), (0, N-n, 0, N-n)) for e, n in zip(es, Ns)])
        am = am.to(torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype).contiguous(); cs = rope_cs(pos, self.freqs)
        for blk in self.blocks: x = blk.forward_b(x, cs, am)
        x = self.nf(x)
        return [self.head_out(x[b, :n], e["n"]) for b, (e, n) in enumerate(zip(es, Ns))]
    def dense(self, d):
        """Padded batch from dense_batch -> final token features (B, N, d). Same math as forward_batch, without per-board loops.
        The bias must be contiguous: a permuted (strided) attn_mask makes SDPA fall back from the memory-efficient kernel to the slow math path."""
        B, N = d["valid"].shape; x = self.inp(d["feats"])
        if self.use_string: x = x + self.pref(d["pref"]) + sum(emb(d["dig"][..., k]) for k, emb in enumerate(self.dig))
        am = torch.zeros(B, 1, 1, N, device=x.device).masked_fill(~d["valid"][:, None, None, :], float("-inf"))
        if self.geobias: am = am + self.gb(pair_geometry(d["box"], d["feats"][..., 0])).permute(0, 3, 1, 2)
        am = am.to(torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype).contiguous(); cs = rope_cs(d["pos"], self.freqs)
        for blk in self.blocks: x = blk.forward_b(x, cs, am)
        return self.nf(x)
    def train_loss(self, d):
        """Sum over the boards of loss_fn (mean over each board's parts), in one padded batch: part rows x (text tokens + none)."""
        x = self.dense(d); B, N, D = x.shape; P = d["tm"].shape[1]; hp = x[:, :P]; n = d["n"][:, None]
        L = torch.cat([self.wq(hp) @ self.wk(x).transpose(1, 2) / math.sqrt(D), self.none(hp)], 2).float()      # (B, P, N+1)
        col = torch.cat([(torch.arange(N, device=x.device)[None] >= n) & d["valid"], d["valid"].new_ones(B, 1)], 1)   # text tokens + none
        L = L.masked_fill(~col[:, None, :], float("-inf"))
        per = torch.logsumexp(L, 2) - torch.logsumexp(L.masked_fill(~d["tm"], float("-inf")), 2)                # padded part rows: 0
        if "w" in d: per = per * d["w"]
        return (per.sum(1) / d["n"]).sum()
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

def summarize(per):
    c = {k: sum(v[k] for v in per.values()) for k in ("linkable", "right", "links")}
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    return dict(acc=round(acc, 4), prec=round(prec, 4), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), **c)

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
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
                outs = model([to_dev(e, dev) for e in g])
            for e, o in zip(g, outs): per[e["id"]] = count(e, decode(o, mode))
        for e in encs:
            if not (e["n"] and e["m"]): per[e["id"]] = count(e, {})
        model.train(); return summarize(per), per
    for e in encs:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
            per[e["id"]] = count(e, decode(model(to_dev(e, dev)), mode) if e["n"] and e["m"] else {})
    model.train(); return summarize(per), per

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--batched", action="store_true", help="the boards of one optimizer step in one padded forward (same math, faster)")
    ap.add_argument("--bps", type=int, default=0, help="boards per optimizer step (overrides --accum; implies --batched): sorted by size, "
                    "cut into padded groups with sizes within --size_ratio and at most --tok_budget padded tokens, gradients accumulated")
    ap.add_argument("--size_ratio", type=float, default=2.0); ap.add_argument("--tok_budget", type=int, default=16384)
    ap.add_argument("--strnoise", action="store_true", help="training: 30 %% of text strings dropped, 15 %% corrupted like OCR (with --tokens E2)")
    ap.add_argument("--label_drop", type=float, default=0.15, help="augment: share of labels dropped (their parts become none)")
    ap.add_argument("--fake", type=float, default=0.05, help="augment: fake designator-like labels per text (name no part)")
    ap.add_argument("--none_w", type=float, default=1.0, help="loss weight of parts whose target is none (batched dot head)")
    ap.add_argument("--ign_messy", action="store_true", help="no loss on parts whose label is printed nearer another part (KiCad train)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the padded training forward + loss (dynamic shapes; single GPU)")
    ap.add_argument("--fused_adam", action="store_true", help="fused AdamW kernel (same update rule)")
    ap.add_argument("--profile", default="", help="'A:B': NVTX ranges per phase and a CUDA profiler range (nsys --capture-range=cudaProfilerApi) "
                    "over steps A..B-1, print s/step, then exit")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.strnoise: STRNOISE = (0.30, 0.15)
    AUGP, NONE_W, IGN_MESSY = (a.label_drop, a.fake), a.none_w, a.ign_messy
    if a.bps: a.accum, a.batched = a.bps, True
    if a.batched: BATCH_EVAL = True
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    sets = load_sets(a.data); AMP = a.amp
    if a.init:
        ck = torch.load(a.init, map_location="cpu"); vocab = Vocab(ck["vocab"])
    else:
        vocab = Vocab.from_boards(sets["all45"] + sets.get("ktrain", []) + sets.get("kval", []) + sets.get("kicadtrain", []) + sets.get("kicadval", []) + sets.get("kicad199", []))
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
    model = LinkFormer(d, L, len(vocab.idx)+2, a.tokens == "E2", a.dropout, a.head, a.geobias).to(dev)
    if a.init: model.load_state_dict(ck["model"])
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    if a.compile and world == 1 and a.batched and a.head == "dot": model.train_loss = torch.compile(model.train_loss, dynamic=True)
    acc = max(1, a.accum // world)                    # boards per rank per optimizer step; total = acc * world (= --accum)
    npar = sum(p.numel() for p in model.parameters()); tag = a.tag or f"{a.train}_{a.size}_{a.tokens}"
    if is_main: print(f"[{tag}] params {npar/1e6:.1f}M, train boards {len(trb)} ({sum(len(b['parts']) for b in trb)} parts), device {dev}, ranks {world} x {acc} boards/step, init {a.init or '-'}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=a.fused_adam and dev.type == "cuda")
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
    from contextlib import nullcontext
    log = []; t0 = time.time(); run = None
    pa, pb = map(int, a.profile.split(":")) if a.profile else (0, 0)
    ph = (lambda s: torch.cuda.nvtx.range(s)) if a.profile else (lambda s: nullcontext())
    for step in range(1, a.steps+1):
        if a.profile and step in (pa, pb):
            torch.cuda.synchronize()
            if step == pa: torch.cuda.cudart().cudaProfilerStart(); tp = time.time()
            else: torch.cuda.cudart().cudaProfilerStop(); print(f"[{tag}] profile steps {pa}..{pb-1}: {(time.time()-tp)/(pb-pa):.4f} s/step", flush=True); sys.exit(0)
        opt.zero_grad(); tot = 0.0
        if a.batched:
            with ph("data"): es = [next(stream) for _ in range(acc)]
            groups = size_groups(es, a.size_ratio, a.tok_budget) if a.bps else [es]
            for gi, g in enumerate(groups):
                with (net.no_sync() if world > 1 and gi < len(groups)-1 else nullcontext()):
                    with ph("forward"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                        if a.head == "dot": loss = net(dense_batch(g, dev, targets=True)) / acc      # one padded batch, loss included
                        else:
                            g = [from_numpy(e) for e in g]; outs = net([to_dev(e, dev) for e in g])
                            loss = sum(loss_lg(o, e["tmask"].to(dev), e["ign"]) for o, e in zip(outs, g)) / acc
                    with ph("backward"): loss.backward(); tot += loss.detach()                       # no host sync here
        for micro in (range(0) if a.batched else range(acc)):
            e = from_numpy(next(stream))
            with (net.no_sync() if world > 1 and micro < acc-1 else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                    logits = net(to_dev(e, dev))
                loss = (loss_lg(logits, e["tmask"].to(dev), e["ign"]) if a.head == "lg" else loss_fn(logits.float(), e["tmask"].to(dev))) / acc
                loss.backward(); tot += loss.item()
        with ph("optim"): torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = tot if run is None else 0.98*run + 0.02*tot                                             # a GPU tensor when batched
        if not is_main: continue
        if wb and step % 50 == 0: wb.log(dict(loss=float(run), lr=sched.get_last_lr()[0], steps_per_s=step/(time.time()-t0)), step=step)
        if a.ckpt_every and a.save and step % a.ckpt_every == 0 and step < a.steps:
            Path(a.save).parent.mkdir(parents=True, exist_ok=True); torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a), step=step), a.save.replace(".pt", f"_step{step}.pt"))
        if evs and (step % a.eval_every == 0 or step == a.steps):
            res = {s: evaluate(model, es, dev)[0] for s, es in evs.items()}
            log.append(dict(step=step, loss=round(float(run), 4), **res))
            if wb: wb.log({f"{s}/{m}": r[m] for s, r in res.items() for m in ("acc", "prec", "f1")}, step=step)
            print(f"[{tag}] step {step} loss {float(run):.3f} " + " ".join(f"{s}: acc {r['acc']:.3f} prec {r['prec']:.3f}" for s, r in res.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    if world > 1:
        import torch.distributed as dist
        if not is_main: dist.destroy_process_group(); sys.exit(0)
    final = {}
    for s in [x for x in a.final_eval.split(",") if x]:
        es = [encode(b, vocab) for b in sets[s] if b["parts"] and b["texts"]]
        for mode in (("hung", "sig") if a.head == "lg" else ("hung",)):
            key = s if mode == "hung" else f"{s}@{mode}"; summ, per = evaluate(model, es, dev, mode); final[key] = dict(summary=summ, per_image=per)
            print(f"[{tag}] final {key}: acc {summ['acc']:.4f} prec {summ['prec']:.4f} f1 {summ['f1']:.4f}", flush=True)
            if wb: wb.summary.update({f"final/{key}/{m}": summ[m] for m in ("acc", "prec", "f1")})
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True); torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a)), a.save)
    out = ROOT/"results/linkformer"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), params=npar, log=log, final=final, seconds=round(time.time()-t0), ranks=world), open(out/f"{tag}.json", "w"), indent=1)
    if wb: wb.finish()
    if world > 1: dist.destroy_process_group()
