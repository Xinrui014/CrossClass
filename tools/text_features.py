#!/usr/bin/env python3
"""Text-image features from a frozen PARSeq reader (2026-10-15), for linkers that see the label's image instead of an OCR string.
Per label box: crop = box enlarged by 15 % (+ jitter for KiCad), read in two orientations (horizontal box: 0 / 180 deg, vertical box:
90 / 270 deg), keep the one PARSeq is more sure of; feature (688-d, fp16) = per-position character probabilities for the first 8 positions
over 38 classes (end, 0-9, A-Z with both cases merged, other) + the mean PARSeq encoder token (384).
KiCad renders (tools/kicad_render_texts.py) get photo-like degradation per crop (colour, blur, resolution loss, noise, JPEG, a stray line),
tuned so PARSeq reads the degraded renders about as well as the WACV photos; sides whose offset_spread_mm > 0.05 are skipped; <side>.stat.json
counts blank crops (grey std < 8: nothing printed at the box, e.g. text hidden under copper) so such sides can be left out. Modes:
  kicad: linkdata/kicad_render/<side>.jpg/.json -> linkdata/kicad_textfeat/<side>.npy (one row per kicad.json text of the side)
  wacv:  WACV-clean v1 photos, annotated text boxes, no degradation -> linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz (<image>__t)
  boxes: the text boxes of an end-to-end boards JSON (--boards) on the WACV photos -> --out npz {board id: (texts, 688)}
  noise: the noise boxes of tools/kicad_noise.py on the renders, with damage -> linkdata/kicad_noisefeat/<side>.npy (one row per box)
Usage: CUDA_VISIBLE_DEVICES=N text_features.py kicad|wacv|boxes [--worker I --nworkers N] [--limit K] [--boards F --out F] [--device cpu]
(python: ~/venvs/gdino_fewshot; PARSeq code ~/third_party/parseq, timm in ~/pydeps/parseq)"""
import argparse, glob, io, json, os, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
sys.path[:0] = [os.path.expanduser("~/pydeps/parseq"), os.path.expanduser("~/third_party/parseq")]
from strhub.models.parseq.model import PARSeq
from strhub.data.utils import Tokenizer
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["kicad", "wacv", "boxes", "noise"]); ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = min(32, cpus))")
ap.add_argument("--boards", default="", help="boxes mode: e2e boards JSON (id, texts[].box) on WACV photos"); ap.add_argument("--out", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--nworkers", type=int, default=1); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
dev = torch.device(a.device); torch.set_num_threads(a.threads or min(32, os.cpu_count() or 8))
CS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"; tok = Tokenizer(CS)
net = PARSeq(len(tok), 25, [32, 128], [4, 8], 384, 6, 4, 12, 12, 4, 1, True, 1, 0.1)
net.load_state_dict(torch.load(os.path.expanduser("~/third_party/parseq_weights/parseq-bb5792a6.pt"), map_location="cpu")); net = net.to(dev).eval()
# token id -> one of 38 classes: 0 end, 1-10 digits, 11-36 letters (case merged), 37 other
cmap = torch.full((len(tok),), 37, dtype=torch.long)
for i in range(len(tok)):
    c = tok._itos[i] if i < len(tok._itos) else ""
    if i == tok.eos_id: cmap[i] = 0
    elif c.isdigit(): cmap[i] = 1 + int(c)
    elif c.isalpha() and c.isascii(): cmap[i] = 11 + ord(c.upper()) - 65
cmap = cmap.to(dev)
mean = torch.tensor(0.5, device=dev)

def degrade(im, rng):
    """Photo-like damage for a clean render crop."""
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.2)); im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.6, 1.1))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.2))
    h = min(im.size)
    if rng.random() < 0.7: im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9) * max(h, 8) / 30))
    if rng.random() < 0.5:
        s = rng.uniform(0.6, 1.0); w0, h0 = im.size; im = im.resize((max(4, int(w0*s)), max(4, int(h0*s))), Image.BILINEAR).resize((w0, h0), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32); arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 6), arr.shape)
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.1:
        d = ImageDraw.Draw(im); w0, h0 = im.size; g = rng.randint(0, 255)
        d.line([(rng.uniform(0, w0), 0), (rng.uniform(0, w0), h0)] if rng.random() < 0.5 else [(0, rng.uniform(0, h0)), (w0, rng.uniform(0, h0))], fill=(g, g, g), width=max(1, int(h0*rng.uniform(0.03, 0.12))))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=rng.randint(40, 92)); return Image.open(buf).convert("RGB")

def crop(img, b, rng=None):
    x0, y0, x1, y1 = b; w, h = max(x1-x0, 1), max(y1-y0, 1); m = [0.15]*4
    if rng: m = [0.15 + rng.uniform(-0.05, 0.05) for _ in range(4)]
    c = img.crop((int(x0-m[0]*w), int(y0-m[1]*h), int(x1+m[2]*w+1), int(y1+m[3]*h+1)))
    return c, h > 1.3*w

def prep(c):
    c = c.resize((128, 32), Image.BICUBIC); return torch.from_numpy(np.asarray(c, np.float32)/255).permute(2, 0, 1)

@torch.no_grad()
def features(crops):
    """crops: list of (PIL crop, vertical) -> (n, 688) fp16 numpy."""
    xs = []
    for c, v in crops:
        for ang in ((90, 270) if v else (0, 180)): xs.append(prep(c.rotate(ang, expand=True) if ang else c))
    out = []
    for k in range(0, len(xs), 1024):
        x = (torch.stack(xs[k:k+1024]).to(dev) - mean)/0.5
        mem = net.encode(x); pool = mem.mean(1)                                           # (b, 384)
        p = net(tok, x).softmax(-1)                                                       # (b, T, tokens)
        T = p.shape[1]; q = torch.zeros(p.shape[0], T, 38, device=dev).index_add_(2, cmap[:p.shape[2]], p)
        if T < 8: q = torch.cat([q, torch.zeros(q.shape[0], 8-T, 38, device=dev).index_fill_(2, torch.tensor([0], device=dev), 1.0)], 1)
        q = q[:, :8]; conf = q.max(-1).values.log().sum(-1)                               # confidence of the 8-position reading
        out.append(torch.cat([q.flatten(1), pool], 1).float().cpu()); out[-1] = (out[-1], conf.cpu())
    F = torch.cat([o[0] for o in out]); C = torch.cat([o[1] for o in out])
    F, C = F.view(-1, 2, F.shape[1]), C.view(-1, 2)                                       # two orientations per crop
    return F[torch.arange(F.shape[0]), C.argmax(1)].numpy().astype(np.float16)

if a.mode == "kicad":
    OUT = ROOT/"linkdata/kicad_textfeat"; OUT.mkdir(parents=True, exist_ok=True)
    sides = sorted(glob.glob(str(ROOT/"linkdata/kicad_render/*.json")))
    if a.limit: sides = sides[:a.limit]
    n = 0
    for k, f in enumerate(sides):
        if k % a.nworkers != a.worker: continue
        s = json.load(open(f))
        if s.get("offset_spread_mm", 1) > 0.05 or (OUT/f"{s['id']}.npy").exists() or not s["texts"]: continue
        img = Image.open(ROOT/f"linkdata/kicad_render/{s['id']}.jpg").convert("RGB"); rng = random.Random(s["id"])
        cr, blank = [], 0
        for b in s["texts"]:
            c, v = crop(img, b, rng); blank += float(np.asarray(c.convert("L"), np.float32).std()) < 8; cr.append((degrade(c, rng), v))
        np.save(OUT/f"{s['id']}.npy", features(cr)); n += 1
        json.dump(dict(texts=len(cr), blank=blank), open(OUT/f"{s['id']}.stat.json", "w"))   # blank crop = no visible print there
        if n % 200 == 0: print(f"worker {a.worker}: {n} sides", flush=True)
    print(f"worker {a.worker} done: {n} sides", flush=True)
elif a.mode == "noise":                              # noise boxes of tools/kicad_noise.py (vias, print, non-component pads), same crop + damage as labels
    OUT = ROOT/"linkdata/kicad_noisefeat"; OUT.mkdir(parents=True, exist_ok=True)
    sides = sorted(glob.glob(str(ROOT/"linkdata/kicad_noise/*.json")))
    if a.limit: sides = sides[:a.limit]
    n = 0
    for k, f in enumerate(sides):
        if k % a.nworkers != a.worker: continue
        s = json.load(open(f))
        if (OUT/f"{s['id']}.npy").exists(): continue
        if not s["px"]: np.save(OUT/f"{s['id']}.npy", np.zeros((0, 688), np.float16)); continue
        img = Image.open(ROOT/f"linkdata/kicad_render/{s['id']}.jpg").convert("RGB"); rng = random.Random(s["id"] + "_noise")
        np.save(OUT/f"{s['id']}.npy", features([(degrade(c, rng), v) for c, v in (crop(img, b, rng) for b in s["px"])])); n += 1
        if n % 200 == 0: print(f"worker {a.worker}: {n} sides", flush=True)
    print(f"worker {a.worker} done: {n} sides", flush=True)
elif a.mode == "boxes":                              # any text boxes on WACV photos, e.g. OCR candidates of the end-to-end boards
    sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import RAW
    photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = {}
    for b in json.load(open(a.boards)):
        if not b["texts"]: out[b["id"]] = np.zeros((0, 688), np.float16); continue
        img = Image.open(photo(b["id"])).convert("RGB"); out[b["id"]] = features([crop(img, t["box"]) for t in b["texts"]])
    np.savez_compressed(a.out, **out); print("boxes done", len(out), sum(len(v) for v in out.values()))
else:
    sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import RAW
    photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = {}
    for b in json.load(open(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1.json")):
        img = Image.open(photo(b["id"])).convert("RGB")
        out[f"{b['id']}__t"] = features([crop(img, t["box"]) for t in b["texts"]]) if b["texts"] else np.zeros((0, 688), np.float16)
    np.savez_compressed(ROOT/"linkdata/CrossClass_data_v1/wacv_clean_v1_parseq.npz", **out); print("wacv done", len(out))
