#!/usr/bin/env python3
"""Whole-board transformer linker (LINKING.md §10: input = two object sets, output = score matrix + none column -> Hungarian).
Tokens: one per part and one per text. Content: type, log width / height, orientation (E1); + prefix and digits of the string (E2).
Position (box centre, in units of the board's median part size) enters only through 2D rotary embeddings (RoPE) in self-attention.
Head: s_ij = <W_q h_i, W_k h_j> / sqrt(d) for part i, text j; one 'none' score per part. Loss: per-part softmax over texts + none,
target = every text whose string equals the part's designator (or none). Decoding: Hungarian on -log p, own none column per part.
Usage: linkformer.py --train dev3 --eval dev3 --size L --tokens E1 --steps 1000 [--device cpu] [--tag name]
  board sets: dev3, test42, all45, fold<k>train, fold<k>test, kicad199"""
import argparse, json, math, os, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); H = Path.home()
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
SIZES = {"S": (128, 4), "M": (256, 6), "L": (384, 8), "XL": (512, 12)}
PFX = re.compile(r"^([A-Z]+)(\d+)([A-Z]?)$")

def load_sets():
    wacv = {b["id"]: b for b in json.load(open(ROOT/"linkdata/wacv.json"))}
    kic = json.load(open(ROOT/"linkdata/kicad199.json"))
    folds = json.load(open(ROOT/"linkdata/folds.json"))   # copy of gdino_fewshot/wacv_in/folds.json (written by linkdata.py)
    sets = {"dev3": [wacv[s] for s in DEV], "all45": list(wacv.values()), "test42": [b for s, b in wacv.items() if s not in DEV], "kicad199": kic}
    for k, v in folds.items():
        sets[f"fold{k}test"] = [b for b in wacv.values() if b["board"] in v["test_boards"]]
        sets[f"fold{k}train"] = [b for b in wacv.values() if b["board"] not in v["test_boards"]]
    return sets

class Vocab:
    def __init__(self, boards):
        pf = sorted({PFX.match(t["text"]).group(1) for b in boards for t in b["texts"] if PFX.match(t["text"])})
        self.idx = {p: i+2 for i, p in enumerate(pf)}   # 0 = part (no string), 1 = unknown prefix
    def __call__(self, s): m = PFX.match(s); return self.idx.get(m.group(1), 1) if m else 1

def encode(b, vocab):
    """Board -> tensors. Units = median part size (longer side)."""
    P = np.array([p["box"] for p in b["parts"]], float).reshape(-1, 4); T = np.array([t["box"] for t in b["texts"]], float).reshape(-1, 4)
    unit = float(np.median(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1]))) if len(P) else 1.0
    B = np.concatenate([P, T]) / unit; w = np.maximum(B[:, 2]-B[:, 0], 1e-3); h = np.maximum(B[:, 3]-B[:, 1], 1e-3)
    n, m = len(P), len(T); is_text = np.r_[np.zeros(n), np.ones(m)]
    feats = np.stack([is_text, np.log(w), np.log(h), (w >= h).astype(float)], 1)
    pos = np.stack([(B[:, 0]+B[:, 2])/2, (B[:, 1]+B[:, 3])/2], 1); pos -= pos.mean(0)
    pref = np.r_[np.zeros(n, int), [vocab(t["text"]) for t in b["texts"]]]
    dig = np.full((n+m, 4), 10, int)   # right-aligned digits, 10 = pad
    for j, t in enumerate(b["texts"]):
        mm = PFX.match(t["text"])
        if mm:
            ds = mm.group(2)[-4:]; dig[n+j, 4-len(ds):] = [int(c) for c in ds]
    strs = [t["text"] for t in b["texts"]]
    tgt = []   # per part: list of target text indices, [] = none
    for p in b["parts"]: tgt.append([j for j, s in enumerate(strs) if p["des"] and s == p["des"]])
    f = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(n=n, m=m, feats=f(feats), pos=f(pos), pref=f(pref, torch.long), dig=f(dig, torch.long), tgt=tgt, strs=strs, des=[p["des"] for p in b["parts"]])

def rope_2d(x, pos, freqs):
    """x: (heads, N, hd); first half of hd rotated by x-position, second half by y-position, one frequency per pair."""
    hd = x.shape[-1]; q = hd // 4
    out = []
    for a, half in enumerate((x[..., :hd//2], x[..., hd//2:])):
        ang = pos[:, a:a+1] * freqs[None, :q]                      # (N, q)
        c, s = torch.cos(ang), torch.sin(ang); x1, x2 = half[..., 0::2], half[..., 1::2]
        out += [torch.stack([x1*c - x2*s, x1*s + x2*c], -1).flatten(-2)]
    return torch.cat(out, -1)

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.h = heads; self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d); self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, pos, freqs):
        N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(N, 3, self.h, d//self.h).permute(1, 2, 0, 3)
        q, k = rope_2d(q, pos, freqs), rope_2d(k, pos, freqs)
        a = F.scaled_dot_product_attention(q[None], k[None], v[None])[0].permute(1, 0, 2).reshape(N, d)
        x = x + self.o(a); return x + self.mlp(self.n2(x))

class LinkFormer(nn.Module):
    def __init__(self, d, layers, n_prefix, use_string):
        super().__init__(); self.use_string = use_string; hd = 64; heads = d // hd
        self.inp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.pref = nn.Embedding(n_prefix, d); self.dig = nn.ModuleList([nn.Embedding(11, d) for _ in range(4)])
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.nf = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d); self.none = nn.Linear(d, 1)
        lam = torch.logspace(math.log10(0.5), math.log10(512), hd // 4)   # wavelengths from half a part to 512 parts
        self.register_buffer("freqs", 2*math.pi/lam)
    def forward(self, e):
        x = self.inp(e["feats"])
        if self.use_string: x = x + self.pref(e["pref"]) + sum(emb(e["dig"][:, k]) for k, emb in enumerate(self.dig))
        for blk in self.blocks: x = blk(x, e["pos"], self.freqs)
        x = self.nf(x); hp, ht = x[:e["n"]], x[e["n"]:]
        S = self.wq(hp) @ self.wk(ht).T / math.sqrt(x.shape[1])
        return torch.cat([S, self.none(hp)], 1)                     # (n, m+1), last column = none

def loss_fn(logits, tgt):
    lse = torch.logsumexp(logits, 1); m = logits.shape[1]-1; ls = []
    for i, t in enumerate(tgt): ls.append(lse[i] - torch.logsumexp(logits[i, t if t else [m]], 0))
    return torch.stack(ls).mean()

@torch.no_grad()
def decode(logits):
    """Hungarian on -log p; each part has its own none column."""
    lp = F.log_softmax(logits, 1).cpu().numpy(); n, m1 = lp.shape; m = m1-1
    C = np.full((n, m+n), 1e6); C[:, :m] = -lp[:, :m]; C[np.arange(n), m+np.arange(n)] = -lp[:, m]
    r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}

def score(encs, links):
    c = dict(linkable=0, right=0, links=0)
    for e, L in zip(encs, links):
        sset = set(e["strs"])
        for i, d in enumerate(e["des"]):
            j = L.get(i); got = e["strs"][j] if j is not None else None; c["links"] += got is not None
            if d and d in sset: c["linkable"] += 1; c["right"] += got == d
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    return dict(acc=round(acc, 4), prec=round(prec, 4), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), **c)

def evaluate(model, encs, dev):
    model.eval(); links = [decode(model({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in e.items()})) if e["n"] and e["m"] else {} for e in encs]
    model.train(); return score(encs, links)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dev3"); ap.add_argument("--eval", default="dev3"); ap.add_argument("--size", default="L")
    ap.add_argument("--tokens", default="E1"); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--tag", default="")
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    sets = load_sets(); vocab = Vocab(sets["all45"] + sets["kicad199"])
    tr = [encode(b, vocab) for b in sets[a.train] if b["parts"] and b["texts"]]
    evs = {s: [encode(b, vocab) for b in sets[s] if b["parts"] and b["texts"]] for s in a.eval.split(",")}
    d, L = SIZES[a.size]; dev = torch.device(a.device)
    model = LinkFormer(d, L, len(vocab.idx)+2, a.tokens == "E2").to(dev)
    npar = sum(p.numel() for p in model.parameters()); tag = a.tag or f"{a.train}_{a.size}_{a.tokens}"
    print(f"[{tag}] params {npar/1e6:.1f}M, train boards {len(tr)} ({sum(e['n'] for e in tr)} parts), device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/100))
    log = []; t0 = time.time(); run = 0.0
    for step in range(1, a.steps+1):
        e = tr[np.random.randint(len(tr))]; e_d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in e.items()}
        loss = loss_fn(model(e_d), e["tgt"]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98*run + 0.02*loss.item() if step > 1 else loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            res = {s: evaluate(model, es, dev) for s, es in evs.items()}
            log.append(dict(step=step, loss=round(run, 4), **{s: r for s, r in res.items()}))
            print(f"[{tag}] step {step} loss {run:.3f} " + " ".join(f"{s}: acc {r['acc']:.3f} prec {r['prec']:.3f}" for s, r in res.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    out = ROOT/"results/linkformer"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), params=npar, log=log), open(out/f"{tag}.json", "w"), indent=1)
