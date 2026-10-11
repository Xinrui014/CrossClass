#!/usr/bin/env python3
"""Whole-board transformer linker (LINKING.md §10: input = two object sets, output = score matrix + none column -> Hungarian).
Tokens: one per part and one per text. Content: type, log width / height, orientation (E1); + prefix and digits of the string (E2).
Position (box centre, in units of the board's median part size) enters only through 2D rotary embeddings (RoPE) in self-attention.
Head: s_ij = <W_q h_i, W_k h_j> / sqrt(d) for part i, text j; one 'none' score per part. Loss: per-part softmax over texts + none,
target = every text whose string equals the part's designator (or none). Decoding: Hungarian on -log p, own none column per part.
Training options: --aug (90-degree rotations + mirror, box jitter, dropped labels, fake labels), --dropout, --accum (boards per
optimizer step), cosine schedule, --save / --init checkpoints (fine-tuning), --workers (background board preparation),
multi-GPU data parallel when launched with torchrun (--accum boards per step are split across the ranks). Results JSON keeps per-image counts so any image subset
(e.g. the 42 test images) can be scored afterwards (tools/linkformer_report.py). --wandb <project> logs loss, lr, eval and final scores.
Usage: linkformer.py --train dev3 --eval dev3 --size L --tokens E1 --steps 1000 [--device cuda] [--tag name]
  board sets: dev3, test42, all45, fold<k>train, fold<k>test, kicad199, kicadtrain, kicadval, kicad300, kicad1000;
  with --data CrossClass_data_v1: ktrain, kval, kvalsub (~25 % of kval), kdense (train sides with WACV density 2.1-11.2)"""
import argparse, json, math, os, re, sys, time, hashlib, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); H = Path.home()
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]
SIZES = {"S": (128, 4), "M": (256, 6), "L": (384, 8), "XL": (512, 12), "XXL": (768, 16), "3XL": (1024, 24)}
PFX = re.compile(r"^([A-Z]+)(\d+)([A-Z]?)$")
FAKE_PFX = ["TP", "R", "C", "U", "J", "D", "Q", "L", "FB", "SW", "Y", "P"]

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
    pref = np.r_[np.zeros(n, int), [vocab(t["text"]) for t in b["texts"]]]
    dig = np.full((n+m, 4), 10, int)   # right-aligned digits, 10 = pad
    for j, t in enumerate(b["texts"]):
        mm = PFX.match(t["text"])
        if mm:
            ds = mm.group(2)[-4:]; dig[n+j, 4-len(ds):] = [int(c) for c in ds]
    strs = [t["text"] for t in b["texts"]]
    tgt = []   # per part: list of target text indices, [] = none
    for p in b["parts"]: tgt.append([j for j, s in enumerate(strs) if p["des"] and s == p["des"]])
    tmask = np.zeros((n, m+1), bool)                 # target cells per part row (all same-string texts, or the none column)
    for i, t in enumerate(tgt): tmask[i, t if t else [m]] = True
    if as_numpy:
        return dict(id=b["id"], n=n, m=m, feats=feats.astype(np.float32), pos=pos.astype(np.float32), pref=pref.astype(np.int64), dig=dig.astype(np.int64),
                    tmask=tmask, tgt=tgt, strs=strs, des=[p["des"] for p in b["parts"]])
    f = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(id=b["id"], n=n, m=m, feats=f(feats), pos=f(pos), pref=f(pref, torch.long), dig=f(dig, torch.long), tmask=torch.from_numpy(tmask),
                tgt=tgt, strs=strs, des=[p["des"] for p in b["parts"]])

def from_numpy(e): return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in e.items()}

class BoardStream(torch.utils.data.IterableDataset):
    """Endless stream of (augmented) encoded training boards; each DataLoader worker has its own RNG. Yields numpy arrays (no shared memory)."""
    def __init__(self, trb, mixb, mix_p, vocab, aug, seed): super().__init__(); self.trb, self.mixb, self.mix_p, self.vocab, self.aug, self.seed = trb, mixb, mix_p, vocab, aug, seed
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(self.seed*1000 + (wi.id if wi else 0))
        while True:
            mix = bool(self.mixb) and rng.random() < self.mix_p; b = rng.choice(self.mixb if mix else self.trb)
            e = encode(augment(b, rng) if (self.aug or mix) else b, self.vocab, as_numpy=True)
            if e["n"] and e["m"]: yield e

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
    def __init__(self, d, heads, dropout=0.0):
        super().__init__(); self.h = heads; self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.p = dropout; self.drop = nn.Dropout(dropout)
        self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d); self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, pos, freqs):
        N, d = x.shape; q, k, v = self.qkv(self.n1(x)).view(N, 3, self.h, d//self.h).permute(1, 2, 0, 3)
        q, k = rope_2d(q, pos, freqs), rope_2d(k, pos, freqs)
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], dropout_p=self.p if self.training else 0.0)[0].permute(1, 0, 2).reshape(N, d)
        x = x + self.drop(self.o(a)); return x + self.drop(self.mlp(self.n2(x)))

class LinkFormer(nn.Module):
    def __init__(self, d, layers, n_prefix, use_string, dropout=0.0):
        super().__init__(); self.use_string = use_string; hd = 64; heads = d // hd
        self.inp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.pref = nn.Embedding(n_prefix, d); self.dig = nn.ModuleList([nn.Embedding(11, d) for _ in range(4)])
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)]); self.nf = nn.LayerNorm(d)
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

def loss_fn(logits, tmask):
    """Per-part softmax over texts + none; -log of the probability mass on the target cells (same math as the old per-part loop)."""
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~tmask, float("-inf")), 1)).mean()

@torch.no_grad()
def decode(logits):
    """Hungarian on -log p; each part has its own none column."""
    lp = F.log_softmax(logits, 1).cpu().numpy(); n, m1 = lp.shape; m = m1-1
    C = np.full((n, m+n), 1e6); C[:, :m] = -lp[:, :m]; C[np.arange(n), m+np.arange(n)] = -lp[:, m]
    r, c = linear_sum_assignment(C); return {int(i): (int(j) if j < m else None) for i, j in zip(r, c)}

def count(e, L):
    sset = set(e["strs"]); c = dict(linkable=0, right=0, links=0)
    for i, d in enumerate(e["des"]):
        j = L.get(i); got = e["strs"][j] if j is not None else None; c["links"] += got is not None
        if d and d in sset: c["linkable"] += 1; c["right"] += got == d
    return c

def summarize(per):
    c = {k: sum(v[k] for v in per.values()) for k in ("linkable", "right", "links")}
    acc = c["right"]/max(c["linkable"], 1); prec = c["right"]/max(c["links"], 1)
    return dict(acc=round(acc, 4), prec=round(prec, 4), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), **c)

def to_dev(e, dev): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in e.items()}

AMP = False
def evaluate(model, encs, dev):
    model.eval(); per = {}
    for e in encs:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
            per[e["id"]] = count(e, decode(model(to_dev(e, dev)).float()) if e["n"] and e["m"] else {})
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
    a = ap.parse_args(); torch.set_num_threads(a.threads)
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
    model = LinkFormer(d, L, len(vocab.idx)+2, a.tokens == "E2", a.dropout).to(dev)
    if a.init: model.load_state_dict(ck["model"])
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    acc = max(1, a.accum // world)                    # boards per rank per optimizer step; total = acc * world (= --accum)
    npar = sum(p.numel() for p in model.parameters()); tag = a.tag or f"{a.train}_{a.size}_{a.tokens}"
    if is_main: print(f"[{tag}] params {npar/1e6:.1f}M, train boards {len(trb)} ({sum(len(b['parts']) for b in trb)} parts), device {dev}, ranks {world} x {acc} boards/step, init {a.init or '-'}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
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
    for step in range(1, a.steps+1):
        opt.zero_grad(); tot = 0.0
        for micro in range(acc):
            e = from_numpy(next(stream))
            with (net.no_sync() if world > 1 and micro < acc-1 else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
                    logits = net(to_dev(e, dev))
                loss = loss_fn(logits.float(), e["tmask"].to(dev)) / acc; loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = tot if run is None else 0.98*run + 0.02*tot
        if not is_main: continue
        if wb and step % 50 == 0: wb.log(dict(loss=run, lr=sched.get_last_lr()[0], steps_per_s=step/(time.time()-t0)), step=step)
        if a.ckpt_every and a.save and step % a.ckpt_every == 0 and step < a.steps:
            Path(a.save).parent.mkdir(parents=True, exist_ok=True); torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a), step=step), a.save.replace(".pt", f"_step{step}.pt"))
        if evs and (step % a.eval_every == 0 or step == a.steps):
            res = {s: evaluate(model, es, dev)[0] for s, es in evs.items()}
            log.append(dict(step=step, loss=round(run, 4), **res))
            if wb: wb.log({f"{s}/{m}": r[m] for s, r in res.items() for m in ("acc", "prec", "f1")}, step=step)
            print(f"[{tag}] step {step} loss {run:.3f} " + " ".join(f"{s}: acc {r['acc']:.3f} prec {r['prec']:.3f}" for s, r in res.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    if world > 1:
        import torch.distributed as dist
        if not is_main: dist.destroy_process_group(); sys.exit(0)
    final = {}
    for s in [x for x in a.final_eval.split(",") if x]:
        es = [encode(b, vocab) for b in sets[s] if b["parts"] and b["texts"]]; summ, per = evaluate(model, es, dev); final[s] = dict(summary=summ, per_image=per)
        print(f"[{tag}] final {s}: acc {summ['acc']:.4f} prec {summ['prec']:.4f} f1 {summ['f1']:.4f}", flush=True)
        if wb: wb.summary.update({f"final/{s}/{m}": summ[m] for m in ("acc", "prec", "f1")})
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True); torch.save(dict(model=model.state_dict(), vocab=vocab.prefixes, args=vars(a)), a.save)
    out = ROOT/"results/linkformer"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), params=npar, log=log, final=final, seconds=round(time.time()-t0), ranks=world), open(out/f"{tag}.json", "w"), indent=1)
    if wb: wb.finish()
    if world > 1: dist.destroy_process_group()
