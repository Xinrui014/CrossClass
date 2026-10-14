#!/usr/bin/env python3
"""Where does the linker go wrong? One checkpoint on the KiCad validation sides (held out from training) and on WACV-clean v1 (41 test
images): scores (KiCad also by board density), error types, and a crop per error for the page page/errors/.
Error types (linkable = the part's label is printed and boxed):
  neighbour  linkable part linked to a text that is another part's label      free   linkable part linked to a text that names no part
  none       linkable part left without a label                               extra  part with no printed label linked to some text
Crops: part = cyan, its true label = green, the model's pick = red, the part that the picked label really belongs to = yellow.
WACV crops come from the photos, KiCad crops are drawn from the layout boxes (parts grey, texts orange with their string).
Writes results/linking/errors_<run>.json (counts, no crops) and page/errors/{errors.json, sheets/*.jpg} (git-ignored: WACV photos).
Usage: link_error_gallery.py --ckpt ckpt/dv1_E1_XXL_150k_lr1e-4.pt [--device cuda] [--kicad_sample 128]   (python: ~/venvs/gdino_fewshot)"""
import argparse, json, os, sys, random, collections, hashlib
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
H = Path.home(); sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(H/"projects/gdino_fewshot/assoc"))
import linkformer as LF
from assoc_lib import RAW
ROOT = Path(os.environ.get("CROSSCLASS", H/"projects/crossclass")); D = ROOT/"linkdata/CrossClass_data_v1"
OUT = ROOT/"page/errors"; SH = OUT/"sheets"; SH.mkdir(parents=True, exist_ok=True)
FONT = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
CROP, GRID = 360, 8                                    # crop side in px, crops per sheet row / column
CY, GR, RD, YL, OR = (34, 211, 238), (74, 222, 128), (248, 64, 64), (250, 204, 21), (251, 146, 60)
ctr = lambda b: np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
BINS = [(0, 1, "< 1"), (1, 2.1, "1 – 2.1"), (2.1, 11.2, "2.1 – 11.2 (WACV-like)"), (11.2, 1e9, "≥ 11.2")]

@torch.no_grad()
def analyse(model, vocab, boards, dev):
    """Per board: link map, softmax, error records; plus counts and how far the true label is (rank among texts by centre distance)."""
    recs, C, rank1 = [], collections.Counter(), collections.Counter()
    per = {}
    for b in boards:
        if not (b["parts"] and b["texts"]): continue
        e = LF.encode(b, vocab); lg = model(LF.to_dev(e, dev)).float(); L = LF.decode(lg); P = torch.softmax(lg, 1).cpu().numpy()
        tg, ig, m = e["tgt"], e["ign"], e["m"]; owner = {j: i for i, t in enumerate(tg) for j in t}
        TC = np.array([ctr(t["box"]) for t in b["texts"]]); c = collections.Counter()
        for i, t in enumerate(tg):
            if ig[i]: continue
            j = L.get(i); c["links"] += j is not None; d = np.linalg.norm(TC-ctr(b["parts"][i]["box"]), axis=1); order = np.argsort(d)
            rk = lambda jj: int(np.where(order == jj)[0][0])+1
            if t:
                c["linkable"] += 1; rank1["linkable"] += 1; rank1["true_nearest"] += rk(t[0]) == 1; rank1["true_top3"] += rk(t[0]) <= 3
                if j in t: c["right"] += 1; continue
                typ = "none" if j is None else ("neighbour" if j in owner else "free")
            else:
                if j is None: continue
                typ = "extra"
            c[typ] += 1
            recs.append(dict(board=b["id"], part=i, type=typ, true=t[0] if t else None, pred=j,
                             true_str=b["texts"][t[0]]["text"] if t else "", pred_str=b["texts"][j]["text"] if j is not None else "",
                             owner=owner.get(j) if j is not None else None, owner_des=b["parts"][owner[j]]["des"] if j is not None and j in owner else "",
                             rank_true=rk(t[0]) if t else None, rank_pred=rk(j) if j is not None else None,
                             p_pred=round(float(P[i, j if j is not None else m]), 3), p_true=round(float(P[i, t[0]]), 3) if t else None,
                             argmax_true=bool(t) and int(P[i].argmax()) == t[0]))
        per[b["id"]] = dict(c); C.update(c)
    return recs, C, per, rank1

def summ(c):
    acc, prec = c["right"]/max(c["linkable"], 1), c["right"]/max(c["links"], 1)
    return dict(acc=round(100*acc, 2), prec=round(100*prec, 2), f1=round(2*acc*prec/max(acc+prec, 1e-9), 4), wrong=c["linkable"]-c["right"],
                **{k: c[k] for k in ("linkable", "right", "links", "neighbour", "free", "none", "extra")})

def region(b, r):
    """Square crop around the part, its true label and the pick (and the pick's owner), with room around them."""
    P = np.array([p["box"] for p in b["parts"]]); unit = float(np.median(np.maximum(P[:, 2]-P[:, 0], P[:, 3]-P[:, 1])))
    bs = [b["parts"][r["part"]]["box"]] + [b["texts"][j]["box"] for j in (r["true"], r["pred"]) if j is not None]
    if r["owner"] is not None: bs.append(b["parts"][r["owner"]]["box"])
    B = np.array(bs); x0, y0, x1, y1 = B[:, 0].min(), B[:, 1].min(), B[:, 2].max(), B[:, 3].max()
    side = max(x1-x0, y1-y0) + 6*unit; side = max(side, 12*unit); cx, cy = (x0+x1)/2, (y0+y1)/2
    return cx-side/2, cy-side/2, side

def draw(img, b, r, x0, y0, side, kicad):
    s = CROP/side; T = lambda q: [(q[0]-x0)*s, (q[1]-y0)*s, (q[2]-x0)*s, (q[3]-y0)*s]
    vis = lambda q: q[2] > x0 and q[0] < x0+side and q[3] > y0 and q[1] < y0+side
    d = ImageDraw.Draw(img); f = FONT(max(9, min(14, int(0.9*s*np.median([t["box"][3]-t["box"][1] for t in b["texts"]])))))
    for p in b["parts"]:
        if vis(p["box"]): d.rectangle(T(p["box"]), outline=(90, 90, 90) if kicad else CY, width=1, fill=(215, 215, 215) if kicad else None)
    for t in b["texts"]:
        if not vis(t["box"]): continue
        q = T(t["box"]); d.rectangle(q, outline=OR, width=1)
        if kicad: d.text(((q[0]+q[2])/2, (q[1]+q[3])/2), t["text"], fill=(120, 60, 0), font=f, anchor="mm")
    pc = ctr(T(b["parts"][r["part"]]["box"]))
    if r["owner"] is not None: d.rectangle(T(b["parts"][r["owner"]]["box"]), outline=YL, width=3)
    if r["true"] is not None:
        q = T(b["texts"][r["true"]]["box"]); d.rectangle(q, outline=GR, width=3); d.line([tuple(pc), tuple(ctr(q))], fill=GR, width=3)
    if r["pred"] is not None:
        q = T(b["texts"][r["pred"]]["box"]); d.rectangle(q, outline=RD, width=3)
        a, z = pc, ctr(q); n = max(int(np.linalg.norm(z-a)/8), 1)
        for k in range(0, n, 2): d.line([tuple(a+(z-a)*k/n), tuple(a+(z-a)*min(k+1, n)/n)], fill=RD, width=3)   # dashed
    d.rectangle(T(b["parts"][r["part"]]["box"]), outline=CY if not kicad else (0, 150, 200), width=4)
    return img

def render(boards, recs, kicad, tag):
    """Crops into sprite sheets; each record gets sheet / cell."""
    by = {b["id"]: b for b in boards}; sheets = []; cur = None
    photo = lambda bid: next(f for f in sorted((RAW/bid).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cache = {}
    for n, r in enumerate(recs):
        if n % (GRID*GRID) == 0:
            if cur: sheets.append(cur)
            cur = Image.new("RGB", (GRID*CROP, GRID*CROP), (20, 20, 20))
        b = by[r["board"]]; x0, y0, side = region(b, r)
        if kicad: im = Image.new("RGB", (CROP, CROP), (250, 250, 247))
        else:
            if r["board"] not in cache: cache.clear(); cache[r["board"]] = Image.open(photo(r["board"])).convert("RGB")
            im = cache[r["board"]].crop((int(x0), int(y0), int(x0+side), int(y0+side))).resize((CROP, CROP), Image.LANCZOS); side = int(x0+side)-int(x0); x0, y0 = int(x0), int(y0)
        draw(im, b, r, x0, y0, side, kicad); k = n % (GRID*GRID); cur.paste(im, ((k % GRID)*CROP, (k//GRID)*CROP))
        r["sheet"], r["cell"] = f"sheets/{tag}_{len(sheets)}.jpg", k
    if cur: sheets.append(cur)
    for i, s in enumerate(sheets): s.save(SH/f"{tag}_{i}.jpg", quality=76)
    return len(sheets)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--kicad_sample", type=int, default=128); a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu"); A = ck["args"]; vocab = LF.Vocab(ck["vocab"]); d, L = LF.SIZES[A["size"]]; dev = torch.device(a.device)
    model = LF.LinkFormer(d, L, len(vocab.idx)+2, A["tokens"] == "E2").to(dev); model.load_state_dict(ck["model"]); model.eval(); run = Path(a.ckpt).stem
    val = set(json.load(open(D/"split.json"))["val_groups"]); kv = [b for b in json.load(open(D/"kicad.json")) if b["group"] in val]
    wc = [b for b in json.load(open(D/"wacv_clean_v1.json")) if b["id"] not in LF.DEV]
    kr, kC, kper, krank = analyse(model, vocab, kv, dev); print("kicad val", summ(kC), flush=True)
    wr, wC, wper, wrank = analyse(model, vocab, wc, dev); print("wacv clean41", summ(wC), flush=True)
    dens = {b["id"]: b["density"] for b in kv}; bins = []
    for lo, hi, name in BINS:
        c = collections.Counter()
        for bid, v in kper.items():
            if lo <= dens[bid] < hi: c.update(v)
        bins.append(dict(bin=name, sides=sum(1 for bid in kper if lo <= dens[bid] < hi), **summ(c)))
    out = dict(run=run, kicad_val=dict(sides=len(kper), **summ(kC), true_nearest=round(100*krank["true_nearest"]/krank["linkable"], 1),
                                       true_top3=round(100*krank["true_top3"]/krank["linkable"], 1), by_density=bins),
               wacv_clean41=dict(images=len(wper), **summ(wC), true_nearest=round(100*wrank["true_nearest"]/wrank["linkable"], 1),
                                 true_top3=round(100*wrank["true_top3"]/wrank["linkable"], 1),
                                 per_image={k: summ(collections.Counter(v)) for k, v in sorted(wper.items())}))
    for s, R in (("kicad_val", kr), ("wacv_clean41", wr)):
        lk = [r for r in R if r["type"] != "extra"]
        out[s]["errors"] = dict(conflict=sum(r["argmax_true"] for r in lk), rank_true_med=float(np.median([r["rank_true"] for r in lk])) if lk else None,
                                pred_nearer=sum(1 for r in lk if r["rank_pred"] is not None and r["rank_pred"] < r["rank_true"]))
    json.dump(out, open(ROOT/f"results/linking/errors_{run}.json", "w"), indent=1)
    rng = random.Random(0); per_t = max(1, a.kicad_sample//4); ks = []
    for t in ("neighbour", "free", "none", "extra"):
        pool = [r for r in kr if r["type"] == t]; ks += rng.sample(pool, min(per_t, len(pool)))
    order = {"neighbour": 0, "free": 1, "none": 2, "extra": 3}
    wr.sort(key=lambda r: (r["board"], order[r["type"]], r["part"])); ks.sort(key=lambda r: (order[r["type"]], r["board"]))
    nk = render(kv, ks, True, "kicad"); nw = render(wc, wr, False, "wacv")
    for r in ks: r["board_name"] = next(b["board"] for b in kv if b["id"] == r["board"])
    json.dump(dict(summary=out, kicad=ks, wacv=wr, crop=CROP, grid=GRID), open(OUT/"errors.json", "w"))
    print("sheets kicad", nk, "wacv", nw, "| cards", len(ks), len(wr))
    print(json.dumps({k: {x: y for x, y in v.items() if x not in ("per_image",)} for k, v in out.items() if k != "run"}, indent=1))
