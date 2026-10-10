#!/usr/bin/env python3
"""Ask a local VLM (vLLM OpenAI endpoint) to DECIDE THE CLASS of every PCB-Dense detection (score >= 0.25) on WACV — not to read
the designator. Same crops as qwen_label.py (square, 3x the part, >= 160 px native, resized to 448..768 px, part outlined in
magenta; cached JPEGs in crops/ are reused). Variants: v2 = appearance + surroundings, printed text allowed as evidence;
v1 = appearance only (told to ignore text). Answer = one JSON line {class, confidence, evidence}.
--view both (owner 2026-10-05: "the crop is too small for small parts"): ONE image with two panels side by side — left = the
close-up as before (3x the part, >= 160 px native), right = the surroundings (max(8x the part, 400 px) native, <= 1600), both 512 px,
the part outlined in magenta in both. Cached in crops_both/.
Resume-safe: answers_<tag>.jsonl keyed "<image>|<detection index>".
Project root = $CROSSCLASS (default ~/projects/crossclass): reads data/preds_wacv_1280.json + data/wacv_images/, writes answers/,
crops/, crops_both/.  Usage: qwen_class.py --tag qc3 --variant v2 --view both --port 9393 [--images A B ...] [--workers 24]"""
import argparse, base64, io, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass"))   # project root: data/ answers/ crops/ crops_both/
E = ROOT/"data"; OUT = ROOT/"answers"; OUT.mkdir(exist_ok=True); (ROOT/"crops").mkdir(exist_ok=True)
NAMES = ["resistor", "capacitor", "inductor", "connector", "diode", "switch", "transistor", "IC", "oscillator"]   # PCB-Dense class ids 0..8
CHOICES = ", ".join(NAMES) + ", unsure"
COMMON = ("The magenta box outlines one electronic component on a printed circuit board. Decide what type of component it is. "
          "Look at its package shape, size, colour, pins and what surrounds it on the board. ")
V = {"v2": COMMON + "You may also use any printed silkscreen designator or marking next to it as evidence. ",
     "v1": COMMON + "Judge from appearance and context only; ignore any printed text. "}
TAIL = (f"Choose exactly one of: {CHOICES}. Inductor includes ferrite beads; diode includes LEDs; connector includes pin headers "
        "and jumpers; oscillator means a crystal or oscillator; IC means any integrated-circuit package.\n"
        'Answer with exactly one line of JSON: {"class": "<one of the choices>", "confidence": "high|medium|low", '
        '"evidence": "appearance|label|both"}')
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--variant", default="v2", choices=list(V))
ap.add_argument("--port", type=int, default=9393); ap.add_argument("--model", default="qwen3.8-27b"); ap.add_argument("--images", nargs="*")
ap.add_argument("--workers", type=int, default=24); ap.add_argument("--thr", type=float, default=0.25); ap.add_argument("--dataset", default="wacv")
ap.add_argument("--view", default="close", choices=["close", "both"])
a = ap.parse_args(); PROMPT = V[a.variant] + TAIL
if a.view == "both":
    PROMPT = ("The picture has two panels of the same printed circuit board: LEFT is a close-up of one component, RIGHT shows the same "
              "component with more of its surroundings; in both panels that component is outlined in magenta. ") + PROMPT.replace(
              "The magenta box outlines one electronic component on a printed circuit board. ", "")
    (ROOT/"crops_both").mkdir(exist_ok=True)
IMG = ROOT/"data"/f"{a.dataset}_images"; URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"
preds = json.load(open(E/f"preds_{a.dataset}_1280.json")); ans_path = OUT/f"answers_{a.tag}.jsonl"
done = set()
if ans_path.exists():
    for l in open(ans_path):
        try: done.add(json.loads(l)["key"])
        except Exception: pass


def crop_for(im, b):   # identical to qwen_label.py
    W, Hh = im.size; L = max(b[2]-b[0], b[3]-b[1]); side = int(min(max(3*L, 160), 1600))
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    x0 = int(min(max(cx-side/2, 0), max(W-side, 0))); y0 = int(min(max(cy-side/2, 0), max(Hh-side, 0)))
    c = im.crop((x0, y0, x0+side, y0+side)); s = min(max(448/side, 1.0) if side < 448 else 768/side if side > 768 else 1.0, 4.0)
    if s != 1.0: c = c.resize((round(side*s), round(side*s)), Image.BICUBIC if s > 1 else Image.LANCZOS)
    dr = ImageDraw.Draw(c); w = max(2, round(c.size[0]/220))
    dr.rectangle([(b[0]-x0)*s-w, (b[1]-y0)*s-w, (b[2]-x0)*s+w, (b[3]-y0)*s+w], outline=(255, 0, 255), width=w)
    return c


def panel(im, b, side, out=512):
    W, Hh = im.size; cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; side = int(min(side, W, Hh))
    x0 = int(min(max(cx-side/2, 0), max(W-side, 0))); y0 = int(min(max(cy-side/2, 0), max(Hh-side, 0)))
    c = im.crop((x0, y0, x0+side, y0+side)); s = out/side; c = c.resize((out, out), Image.BICUBIC if s > 1 else Image.LANCZOS)
    dr = ImageDraw.Draw(c); w = max(2, round(out/220)); dr.rectangle([(b[0]-x0)*s-w, (b[1]-y0)*s-w, (b[2]-x0)*s+w, (b[3]-y0)*s+w], outline=(255, 0, 255), width=w)
    return c


def crop_both(im, b):   # left: close-up (3x, >= 160 native); right: context (max(8x, 400) native, <= 1600)
    L = max(b[2]-b[0], b[3]-b[1]); a_ = panel(im, b, min(max(3*L, 160), 1600)); b_ = panel(im, b, min(max(8*L, 400), 1600))
    out = Image.new("RGB", (1024+8, 512), (255, 255, 255)); out.paste(a_, (0, 0)); out.paste(b_, (520, 0)); return out


def ask(jpeg_bytes):
    body = {"model": a.model, "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}},
                                                       {"type": "text", "text": PROMPT}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600)); return (r["choices"][0]["message"].get("content") or "").strip(), r["usage"]


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try: j = json.loads(m.group(0)) if m else {}
    except Exception: j = {}
    c = str(j.get("class", "")).strip().lower(); low = [n.lower() for n in NAMES]
    cls = low.index(c) if c in low else None
    conf = str(j.get("confidence", "")).strip().lower(); ev = str(j.get("evidence", "")).strip().lower()
    return cls, (conf if conf in ("high", "medium", "low") else "low"), (ev if ev in ("appearance", "label", "both") else "unknown")


jobs = [(st, k, p) for st in (a.images or sorted(preds)) for k, p in enumerate(preds[st]) if p[4] >= a.thr and f"{st}|{k}" not in done]
print(f"[{time.strftime('%H:%M:%S')}] {a.variant}: {len(jobs)} queries ({len(done)} already done)", flush=True)
cache = {}


def run(job):
    st, k, p = job; fn = ROOT/("crops_both" if a.view == "both" else "crops")/f"{a.dataset}__{st}__{k}.jpg"
    if fn.exists(): jb = fn.read_bytes()
    else:
        if st not in cache: cache[st] = Image.open(next(IMG.glob(st+".*"))).convert("RGB")
        c = (crop_both if a.view == "both" else crop_for)(cache[st], p[:4]); c.save(fn, quality=85); jb = fn.read_bytes()
    t = time.time(); txt, use = ask(jb); cls, conf, ev = parse(txt)
    return dict(key=f"{st}|{k}", image=st, det=k, box=p[:4], score=p[4], det_cls=int(p[5]), raw=txt, qwen_cls=cls, confidence=conf, evidence=ev,
                secs=round(time.time()-t, 2), prompt_tokens=use.get("prompt_tokens"), completion_tokens=use.get("completion_tokens"))


t0 = time.time(); n = 0
with open(ans_path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
    for st in sorted({j[0] for j in jobs}):
        futs = [ex.submit(run, j) for j in jobs if j[0] == st]
        for fu in as_completed(futs):
            try: r = fu.result()
            except Exception as e: print("ERR", repr(e)[:200], flush=True); continue
            f.write(json.dumps(r)+"\n"); f.flush(); n += 1
        cache.pop(st, None)
        print(f"[{time.strftime('%H:%M:%S')}] {st} done; {n}/{len(jobs)}; {n/max(time.time()-t0, 1):.2f} q/s", flush=True)
print("QWEN_CLASS_DONE", n, round(time.time()-t0), flush=True)
