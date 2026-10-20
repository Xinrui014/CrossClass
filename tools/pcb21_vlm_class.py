#!/usr/bin/env python3
"""LLM decides the class AFTER linking and reading (owner idea 2026-10-19): for every detector box (score >= 0.25) that the linker
links to a label (soft-link mass of v2d2_I_noise05_100k on the rule boxes >= 0.5), a local VLM (Qwen3.8-27B-FP8, vLLM OpenAI
endpoint, tools/serve_qwen38_fp8_1gpu.sh) gets: one image with two panels (LEFT close-up of the part + its strongest linked label,
RIGHT wider surroundings; part outlined magenta, label cyan), the PARSeq reading(s) of the linked label(s) with link probability, the
detector's top-3 guesses, and the dataset's class conventions. It answers one JSON line {class, confidence, label} where label is
its own reading of the cyan text. Unlinked boxes are not sent (they keep C). Resume-safe: answers/vlmcls_<dataset>_<tag>.jsonl.
  pcb21_vlm_class.py --dataset 21pcb|wacv --tag v1 [--port 9393] [--workers 24] [--limit N]   (any python3 with PIL + numpy)"""
import argparse, base64, io, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); E = ROOT/"data"; OUT = ROOT/"answers"; OUT.mkdir(exist_ok=True)
NAMES = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]
DEV = ["ML450_Top1", "RPI3B_Top", "DuetWIFI_Top"]; T = Path("/tmp/xinrui_e2e")
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="21pcb", choices=["21pcb", "wacv"]); ap.add_argument("--tag", default="v1")
ap.add_argument("--port", type=int, default=9393); ap.add_argument("--model", default="qwen3.8-27b"); ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
if a.dataset == "21pcb":
    LD = E/"21pcb/e2e"; P9 = json.load(open(LD/"preds_cls9.json")); B = {b["id"]: b for b in json.load(open(LD/"e2e_boards_all.json"))}
    Z = np.load(LD/"e2e_tf_all.npz"); SOFT = json.load(open(LD/"soft_noise05_infil.json")); photo = lambda st: E/"21pcb/images"/f"{st}.jpg"
    imgs = sorted(json.load(open(E/"gt9_21pcb.json")))
else:
    import sys; sys.path.insert(0, os.path.expanduser("~/projects/gdino_fewshot/assoc")); from assoc_lib import RAW
    P9 = json.load(open(E/"preds_wacv_1280_cls9.json")); B = {b["id"]: b for b in json.load(open(T/"e2e_boards_all.json"))}
    Z = np.load(T/"e2e_tf_all.npz"); SOFT = json.load(open(T/"soft_noise05_infil.json"))
    photo = lambda st: next(f for f in sorted((RAW/st).iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = sorted(s for s in json.load(open(E/"gt9_wacv.json")) if s not in DEV)
CH = "#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
read = lambda st, j: "".join(CH[c] for c in Z[st][j][:304].astype(np.float32).reshape(8, 38).argmax(1)).split("#")[0]
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"; ans_path = OUT/f"vlmcls_{a.dataset}_{a.tag}.jsonl"
done = set()
if ans_path.exists():
    for l in open(ans_path):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
CONV = ("Class conventions of this dataset: ferrite beads, EMI filters and transformers count as inductor; LEDs and TVS diodes count as diode; "
        "pin headers, jumpers, terminal blocks and RJ-45 jacks count as connector; 0-ohm links count as resistor; crystals and oscillators count as "
        "oscillator; fuses, polyfuses, test points and mechanical parts count as ignore.")
CHOICES = ", ".join(NAMES) + ", ignore, unsure"

def panel(im, box, side, out=512, label=None):
    W, Hh = im.size; cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2; side = int(min(side, W, Hh))
    x0 = int(min(max(cx-side/2, 0), max(W-side, 0))); y0 = int(min(max(cy-side/2, 0), max(Hh-side, 0)))
    c = im.crop((x0, y0, x0+side, y0+side)).resize((out, out), Image.LANCZOS); s = out/side; dr = ImageDraw.Draw(c); w = max(2, round(out/220))
    R = lambda q: [(q[0]-x0)*s-w, (q[1]-y0)*s-w, (q[2]-x0)*s+w, (q[3]-y0)*s+w]
    if label: dr.rectangle(R(label), outline=(0, 220, 255), width=w)
    dr.rectangle(R(box), outline=(255, 0, 255), width=w); return c

def image_for(im, part, label):
    u = [min(part[0], label[0]), min(part[1], label[1]), max(part[2], label[2]), max(part[3], label[3])]
    L = max(u[2]-u[0], u[3]-u[1]); left = panel(im, u, max(1.6*L, 160), label=label)
    Lp = max(part[2]-part[0], part[3]-part[1]); right = panel(im, part, min(max(8*Lp, 3*L, 400), 2400), label=label)
    out = Image.new("RGB", (1032, 512), (255, 255, 255)); out.paste(left, (0, 0)); out.paste(right, (520, 0)); return out

def prompt(reads, det):
    rd = "; ".join(f"'{r or '?'}' (link probability {p:.2f})" for r, p in reads)
    dg = ", ".join(f"{NAMES[c]} {v:.2f}" for c, v in det)
    return ("The picture has two panels of the same printed circuit board: LEFT is a close-up, RIGHT shows more surroundings. In both, one electronic "
            "component is outlined in magenta and its printed designator label (found by a linking model) is outlined in cyan. "
            f"An OCR model read the label as: {rd}. The OCR can miss or confuse characters (for example D read as 0, a missing first letter, "
            f"rotated text), so check the cyan text in the image yourself. An image detector guessed the component type as: {dg}; it often "
            "confuses look-alike parts (ferrite beads or small diodes that look like resistors, tantalum or coloured parts). "
            "Use the designator prefix and your knowledge of designator conventions (R resistor, C capacitor, L/FB/E bead or inductor, D/LED/TVS "
            "diode, Q transistor, U IC, Y/X crystal, J/P/JP connector, SW/S switch, F fuse, TP test point ...) together with the appearance. "
            f"{CONV} Choose exactly one of: {CHOICES}.\n"
            'Answer with exactly one line of JSON: {"class": "<one of the choices>", "confidence": "high|medium|low", "label": "<the designator you read in the cyan box>"}')

def ask(jb, text):
    body = {"model": a.model, "temperature": 0, "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jb).decode()}},
                                                       {"type": "text", "text": text}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600)); return (r["choices"][0]["message"].get("content") or "").strip()

def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try: j = json.loads(m.group(0)) if m else {}
    except Exception: j = {}
    c = str(j.get("class", "")).strip().lower(); low = [n.lower() for n in NAMES]
    cls = low.index(c) if c in low else ("ignore" if c == "ignore" else None)
    conf = str(j.get("confidence", "")).strip().lower()
    return cls, (conf if conf in ("high", "medium", "low") else "low"), str(j.get("label", ""))[:20]

jobs = []
for st in imgs:
    for i, k in enumerate(B[st]["det_index"]):
        r = SOFT[st].get(str(i))
        if not r or sum(p for _, p in r["texts"]) < 0.5 or f"{st}|{k}" in done: continue
        tx = sorted(r["texts"], key=lambda x: -x[1]); reads = [(read(st, j), p) for j, p in tx if p >= 0.1][:2]
        d = np.array(P9[st][k][6:15]); d = d / d.sum(); det = [(int(c), float(d[c])) for c in np.argsort(-d)[:3]]
        jobs.append((st, k, P9[st][k][:4], B[st]["texts"][tx[0][0]]["box"], reads, det))
if a.limit: jobs = jobs[:a.limit]
print(f"[{time.strftime('%H:%M:%S')}] {a.dataset}: {len(jobs)} queries ({len(done)} already done)", flush=True)
cache = {}
def run(job):
    st, k, part, label, reads, det = job
    if st not in cache: cache.clear(); cache[st] = Image.open(photo(st)).convert("RGB")
    buf = io.BytesIO(); image_for(cache[st], part, label).save(buf, "JPEG", quality=88); t = time.time()
    txt = ask(buf.getvalue(), prompt(reads, det)); cls, conf, lab = parse(txt)
    return dict(key=f"{st}|{k}", image=st, det=k, reads=reads, det_top3=det, raw=txt, cls=cls, confidence=conf, label=lab, secs=round(time.time()-t, 2))
t0 = time.time(); n = 0
with open(ans_path, "a") as f:
    for st in sorted({j[0] for j in jobs}):              # one photo at a time (the big photo stays in memory)
        with ThreadPoolExecutor(a.workers) as ex:
            for fu in as_completed([ex.submit(run, j) for j in jobs if j[0] == st]):
                try: f.write(json.dumps(fu.result()) + "\n"); f.flush(); n += 1
                except Exception as e: print("error", e, flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {st}: {n}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
print("VLM_DONE", n)
