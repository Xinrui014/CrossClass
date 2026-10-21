#!/usr/bin/env python3
"""Keyboard-only review page for the 21PCB batch-1 text-box strings (owner 2026-10-20: labelme hides the description behind a
double click). Local HTML on the L40 only (owner rule: no artifact pages for annotation).
  build : from data/21pcb/text_check_b1_ocr.json (tools/pcb21_text_ocr.py) write page/text_review/{index.html, items.json, img/}.
          Per text box: a context crop (box in magenta) and an upright crop (rotated as OCR read it). Order: text_check (OCR unsure or
          differs from the old string), text_other (confident, not designator-like), text (confident designator, agrees).
          Suggested string = the old string if the box had one (unless it was marked "?" and OCR is confident), else the OCR
          reading; the other one is offered as the alternative.
  serve PORT : serve the page on 127.0.0.1:PORT; POST /save stores {id: {value, status, at}} in data/21pcb/text_review_answers.json
          (atomic rewrite) and appends every save to data/21pcb/text_review_log.jsonl; "at" = server UTC time.
Open from the laptop: ssh -L PORT:localhost:PORT l40, then http://localhost:PORT
Keys: Enter = keep the input and go on; Shift+Enter = take the alternative; type to correct; '-' = not text; '?' = cannot read;
ArrowDown / ArrowUp = next / previous without saving; Esc = reset the input. index.html#todo = only unanswered + flagged boxes.
Grid page grid.html (owner 2026-10-20: one box at a time is too slow): 40 unanswered boxes per screen (upright crop + suggestion);
click a tile or 'x' = flag it as wrong; Space = save the page (unflagged -> ok, flagged -> status "flag", fixed later in
index.html#todo) and show the next; Backspace = previous page; 'c' = context crop of the tile under the cursor; hash #check /
#other / #text = one group only.
  pcb21_text_review.py build   (env ~/venvs/gdino_fewshot)
  pcb21_text_review.py pages   (rewrite index.html + grid.html only)
  pcb21_text_review.py serve 8767"""
import json, os, re, sys, threading
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(os.environ.get("CROSSCLASS", Path.home()/"projects/crossclass")); D = ROOT/"data/21pcb"; PAGE = ROOT/"page/text_review"
ANSF = D/"text_review_answers.json"; LOGF = D/"text_review_log.jsonl"

def build():
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    S = json.load(open(D/"text_check_b1_ocr.json")); (PAGE/"img").mkdir(parents=True, exist_ok=True); items = []
    ROT = {0: None, 90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}       # OCR rotated clockwise by rot
    for st, rows in S.items():
        im = Image.open(D/"images"/f"{st}.jpg").convert("RGB"); W0, H0 = im.size
        for r in rows:
            b = r["box"]; w, h = b[2]-b[0], b[3]-b[1]; why = r["check"]
            grp = "other" if why == ["not designator-like"] else "check" if why else "text"
            side = max(3*max(w, h), 220); cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; X0, Y0 = cx-side/2, cy-side/2
            c = im.crop((int(X0), int(Y0), int(X0+side), int(Y0+side))).resize((280, 280), Image.LANCZOS); z = 280/side
            ImageDraw.Draw(c).rectangle([(b[0]-X0)*z, (b[1]-Y0)*z, (b[2]-X0)*z, (b[3]-Y0)*z], outline=(255, 0, 255), width=2)
            p = 0.2*min(w, h) + 4; u = im.crop((int(max(0, b[0]-p)), int(max(0, b[1]-p)), int(min(W0, b[2]+p)), int(min(H0, b[3]+p))))
            if ROT[r["rot"]] is not None: u = u.transpose(ROT[r["rot"]])
            s = min(64/u.size[1], 360/u.size[0]); u = u.resize((max(1, round(u.size[0]*s)), max(1, round(u.size[1]*s))), Image.LANCZOS)
            iid = f"{st}#{r['i']}"; fn = re.sub(r"[^A-Za-z0-9_]", "_", iid)
            c.save(PAGE/"img"/f"{fn}_c.jpg", quality=85); u.save(PAGE/"img"/f"{fn}_u.jpg", quality=90)
            old = r["old"].lstrip("?"); sugg = old or r["read"]; alt = r["read"] if old and r["read"] and r["read"] != old else ""
            if alt and r["old"].startswith("?") and r["score"] >= 0.9: sugg, alt = alt, old     # '?' = the pre-label was unsure
            items.append(dict(id=iid, st=st, grp=grp, sugg=sugg, alt=alt, old=r["old"], read=r["read"], score=r["score"], why=why,
                              c=f"img/{fn}_c.jpg", u=f"img/{fn}_u.jpg"))
        print(st, len(rows), flush=True)
    order = {"check": 0, "other": 1, "text": 2}; items.sort(key=lambda x: (order[x["grp"]], x["st"], int(x["id"].split("#")[1])))
    json.dump(items, open(PAGE/"items.json", "w")); pages()
    print(len(items), "items ->", PAGE)

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>21PCB Text Review</title><style>
:root{--bg:#14171a;--card:#1d2226;--line:#2c343a;--fg:#e3e8ec;--dim:#93a0ab;--act:#ffd54a;--ok:#4cd28a;--fix:#7cc7ff;--bad:#ff8a80}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:0 16px 60vh}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2}
h1{font-size:17px;margin:0 0 4px}.dim{color:var(--dim)}.keys{font-size:12.5px}.keys b{color:var(--fg)}
#err{display:none;background:#5a1d1d;color:#ffd0d0;padding:4px 8px;border-radius:4px;margin-top:4px}
h2{font-size:15px;margin:22px 0 6px}
.row{display:grid;grid-template-columns:280px minmax(0,1fr);gap:12px;background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}
.row.act{border-color:var(--act)}.row img.c{width:280px;height:280px;display:block;border-radius:4px}
.right{display:flex;flex-direction:column;gap:6px;min-width:0}.up{background:#000;padding:6px;border-radius:4px;align-self:flex-start;max-width:100%}
.up img{display:block;max-width:100%}
input{font:600 22px/1.2 ui-monospace,monospace;background:#0e1113;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;width:14ch;max-width:100%}
.row.act input{border-color:var(--act)}.meta{font-size:12.5px;color:var(--dim)}.alt{color:var(--fix);font-weight:600}
.st{font-weight:700}.st.ok{color:var(--ok)}.st.fixed{color:var(--fix)}.st.not_text,.st.unsure{color:var(--bad)}
@media (max-width:600px){.row{grid-template-columns:1fr}.row img.c{width:100%;height:auto}}
</style></head><body>
<header><h1>21PCB batch 1: text-box strings</h1>
<div class="keys dim"><b>Enter</b> keep and next &middot; <b>Shift+Enter</b> take the blue alternative &middot; type to correct &middot;
<b>-</b> not text &middot; <b>?</b> cannot read &middot; <b>&darr; / &uarr;</b> next / previous without saving &middot; <b>Esc</b> reset</div>
<div id="prog" class="dim"></div><div id="err"></div></header><main id="list"></main>
<script>
const NAME={check:"Please check (OCR unsure or differs from the old string)",other:"Confident, not designator-like (skim)",text:"Confident designators (spot check)"};
let items=[],ans={},rows=[],cur=0,pending=[];
const $=s=>document.querySelector(s);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function stTxt(a){return a?({ok:"ok",fixed:"fixed",not_text:"not text",unsure:"cannot read",flag:"flagged in grid: please fix"}[a.status]+(a.status==="fixed"?": "+esc(a.value):"")):""}
function prog(){const g={};for(const it of items){g[it.grp]=g[it.grp]||[0,0];g[it.grp][1]++;if(ans[it.id]&&ans[it.id].status!=="flag")g[it.grp][0]++}
 $("#prog").textContent=Object.entries(g).map(([k,v])=>`${k}: ${v[0]} / ${v[1]}`).join("   ·   ")}
function render(){let h="",last="";items.forEach((it,i)=>{if(it.grp!==last){last=it.grp;h+=`<h2>${NAME[it.grp]}</h2>`}
 const a=ans[it.id];h+=`<div class="row" id="r${i}" data-i="${i}"><img class="c" loading="lazy" src="${it.c}" alt="">
 <div class="right"><div class="up"><img loading="lazy" src="${it.u}" alt=""></div>
 <input id="in${i}" value="${esc(a?a.value:it.sugg)}" autocomplete="off" spellcheck="false">
 <div class="meta">${it.alt?`alternative <span class="alt">${esc(it.alt)}</span> (OCR) &middot; `:""}old: ${esc(it.old||"-")} &middot; OCR: ${esc(it.read||"-")} ${it.score.toFixed(2)}${it.why.length?" &middot; "+esc(it.why.join(", ")):""}</div>
 <div class="meta">${esc(it.st)} &middot; <span class="st ${a?a.status:""}" id="s${i}">${stTxt(a)}</span></div></div></div>`});
 $("#list").innerHTML=h;rows=[...document.querySelectorAll(".row")];
 rows.forEach(r=>r.addEventListener("click",()=>go(+r.dataset.i)))}
function go(i){if(i<0||i>=items.length)return;rows[cur]&&rows[cur].classList.remove("act");cur=i;rows[i].classList.add("act");
 rows[i].scrollIntoView({block:"center"});const inp=$("#in"+i);inp.focus();inp.select()}
async function post(rec){try{const r=await fetch("save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
 if(!r.ok)throw new Error(r.status);return true}catch(e){return false}}
async function flush(){const left=[];for(const rec of pending){if(!(await post(rec)))left.push(rec)}pending=left;
 const e=$("#err");if(pending.length){e.style.display="block";e.textContent=`${pending.length} change(s) NOT saved yet (server unreachable); they are kept and retried.`}else e.style.display="none"}
function save(i,value){const it=items[i];value=value.trim();
 const status=value==="-"?"not_text":value==="?"?"unsure":(value===it.sugg?"ok":"fixed");
 ans[it.id]={value,status};const s=$("#s"+i);s.className="st "+status;s.textContent=stTxt(ans[it.id]);
 $("#in"+i).value=value;prog();pending.push({id:it.id,value,status});
 try{localStorage.setItem("pcb21_text_review_pending",JSON.stringify(pending))}catch(e){}flush()}
document.addEventListener("keydown",e=>{const it=items[cur];if(!it)return;
 if(e.key==="Enter"){e.preventDefault();let v=$("#in"+cur).value;if(e.shiftKey){if(!it.alt)return;v=it.alt}save(cur,v);go(cur+1)}
 else if(e.key==="ArrowDown"){e.preventDefault();go(cur+1)}else if(e.key==="ArrowUp"){e.preventDefault();go(cur-1)}
 else if(e.key==="Escape"){e.preventDefault();const inp=$("#in"+cur);inp.value=ans[it.id]?ans[it.id].value:it.sugg;inp.select()}});
setInterval(()=>{if(pending.length)flush()},5000);
(async()=>{items=await (await fetch("items.json")).json();try{ans=await (await fetch("answers")).json()}catch(e){ans={}}
 try{const p=JSON.parse(localStorage.getItem("pcb21_text_review_pending")||"[]");if(Array.isArray(p))pending=p}catch(e){}
 const todo=it=>!ans[it.id]||ans[it.id].status==="flag";if(location.hash==="#todo")items=items.filter(todo);
 render();prog();const first=items.findIndex(todo);go(first<0?0:first);flush()})();
</script></body></html>"""

GRID = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>21PCB Text Grid</title><style>
:root{--bg:#14171a;--card:#1d2226;--line:#2c343a;--fg:#e3e8ec;--dim:#93a0ab;--act:#ffd54a;--bad:#ff5a50}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:0 16px 40px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:3}
h1{font-size:17px;margin:0 0 4px}.dim{color:var(--dim)}.keys{font-size:12.5px}.keys b{color:var(--fg)}a{color:#7cc7ff}
#err{display:none;background:#5a1d1d;color:#ffd0d0;padding:4px 8px;border-radius:4px;margin-top:4px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin-top:10px}
.t{position:relative;background:var(--card);border:3px solid var(--line);border-radius:8px;padding:6px;cursor:pointer;min-width:0}
.t.cur{border-color:var(--act)}.t.flag{border-color:var(--bad);background:#3a1d1d}.t.flag.cur{outline:3px solid var(--act)}
.u{background:#000;border-radius:4px;padding:4px;min-height:40px;display:flex;align-items:center;justify-content:center}.u img{max-width:100%;display:block}
.s{font:700 20px/1.3 ui-monospace,monospace;margin-top:4px;overflow-wrap:anywhere}.m{font-size:11.5px;color:var(--dim)}
.ctx{display:none;position:absolute;z-index:4;top:100%;left:0;width:280px;height:280px;border:2px solid var(--act);border-radius:6px}
.t.showc .ctx{display:block}
</style></head><body>
<header><h1>21PCB batch 1: text strings, grid</h1>
<div class="keys dim">Look at each tile: is the big string what is printed? <b>Click</b> a wrong tile (or move with <b>arrows</b> and press
<b>x</b>) to flag it red. <b>Space</b> = save this page (others = ok) and go on &middot; <b>Backspace</b> = previous page &middot;
<b>c</b> = context of the tile under the cursor &middot; flagged tiles: fix them in <a href="index.html#todo">the list page (#todo)</a></div>
<div id="prog" class="dim"></div><div id="err"></div></header><div id="grid"></div>
<script>
const N=40;let items=[],ans={},queue=[],pos=0,cur=0,flags=new Set(),pending=[];
const $=s=>document.querySelector(s);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function page(){return queue.slice(pos,pos+N)}
function prog(){const n=Object.values(ans).length,f=Object.values(ans).filter(a=>a.status==="flag").length;
 $("#prog").textContent=`page ${Math.floor(pos/N)+1} / ${Math.max(1,Math.ceil(queue.length/N))} (${queue.length} boxes were open at load) · answered ${n} of ${items.length} · flagged ${f}`}
function show(){const P=page();flags=new Set(P.map((it,k)=>ans[it.id]&&ans[it.id].status==="flag"?k:-1).filter(k=>k>=0));
 if(!P.length){$("#grid").innerHTML=`<p>Nothing left in this view. Flagged boxes: <a href="index.html#todo">fix them in the list page</a>.</p>`;prog();return}
 $("#grid").innerHTML=P.map((it,k)=>`<div class="t${flags.has(k)?" flag":""}" id="t${k}" data-k="${k}"><div class="u"><img src="${it.u}" alt=""></div>
 <div class="s">${esc(it.sugg)||"&nbsp;"}</div><div class="m">${it.grp}${it.alt?" · alt "+esc(it.alt):""}</div><img class="ctx" loading="lazy" src="${it.c}" alt=""></div>`).join("");
 document.querySelectorAll(".t").forEach(t=>t.addEventListener("click",()=>{setCur(+t.dataset.k);toggle(+t.dataset.k)}));
 setCur(0);window.scrollTo(0,0);prog()}
function setCur(k){const P=page();if(k<0||k>=P.length)return;const o=$("#t"+cur);if(o)o.classList.remove("cur","showc");cur=k;const t=$("#t"+k);t.classList.add("cur");t.scrollIntoView({block:"nearest"})}
function toggle(k){flags.has(k)?flags.delete(k):flags.add(k);$("#t"+k).classList.toggle("flag",flags.has(k))}
function cols(){const ts=[...document.querySelectorAll(".t")];if(!ts.length)return 1;const y=ts[0].offsetTop;return Math.max(1,ts.filter(t=>t.offsetTop===y).length)}
async function post(rec){try{const r=await fetch("save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});if(!r.ok)throw 0;return true}catch(e){return false}}
let busy=false;async function flush(){if(busy)return;busy=true;const left=[];for(const rec of pending){if(!(await post(rec)))left.push(rec)}
 pending=left.concat(pending.slice(pending.length));busy=false;try{localStorage.setItem("pcb21_text_grid_pending",JSON.stringify(pending))}catch(e){}
 const e=$("#err");if(pending.length){e.style.display="block";e.textContent=`${pending.length} change(s) NOT saved yet (server unreachable); kept and retried.`}else e.style.display="none"}
function commit(){page().forEach((it,k)=>{const status=flags.has(k)?"flag":"ok";ans[it.id]={value:it.sugg,status};pending.push({id:it.id,value:it.sugg,status})});
 try{localStorage.setItem("pcb21_text_grid_pending",JSON.stringify(pending))}catch(e){}flush();pos+=N;show()}
document.addEventListener("keydown",e=>{const P=page();if(e.key===" "){e.preventDefault();if(P.length)commit()}
 else if(e.key==="Backspace"){e.preventDefault();if(pos>0){pos=Math.max(0,pos-N);show()}}
 else if(!P.length)return;
 else if(e.key==="x"||e.key==="X"){e.preventDefault();toggle(cur)}
 else if(e.key==="c"||e.key==="C"){e.preventDefault();$("#t"+cur).classList.toggle("showc")}
 else if(e.key==="ArrowRight"){e.preventDefault();setCur(cur+1)}else if(e.key==="ArrowLeft"){e.preventDefault();setCur(cur-1)}
 else if(e.key==="ArrowDown"){e.preventDefault();setCur(Math.min(P.length-1,cur+cols()))}else if(e.key==="ArrowUp"){e.preventDefault();setCur(Math.max(0,cur-cols()))}});
setInterval(()=>{if(pending.length)flush()},5000);
(async()=>{items=await (await fetch("items.json")).json();try{ans=await (await fetch("answers")).json()}catch(e){ans={}}
 try{const p=JSON.parse(localStorage.getItem("pcb21_text_grid_pending")||"[]");if(Array.isArray(p))pending=p}catch(e){}
 const g=location.hash.slice(1);queue=items.filter(it=>!ans[it.id]&&(!g||it.grp===g));show();flush()})();
</script></body></html>"""

def pages():
    (PAGE/"index.html").write_text(HTML); (PAGE/"grid.html").write_text(GRID)

def serve(port):
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    ids = {it["id"] for it in json.load(open(PAGE/"items.json"))}
    ans = json.load(open(ANSF)) if ANSF.exists() else {}; lock = threading.Lock()
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k): super().__init__(*a, directory=str(PAGE), **k)
        def log_message(self, *a): pass
        def _json(self, obj, code=200):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path.split("?")[0] == "/answers":
                with lock: return self._json(ans)
            return super().do_GET()
        def do_POST(self):
            if self.path.split("?")[0] != "/save": return self._json({"error": "not found"}, 404)
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                iid, value, status = str(d["id"]), str(d["value"])[:80], str(d["status"])
                assert iid in ids and status in ("ok", "fixed", "not_text", "unsure", "flag")
            except Exception as e: return self._json({"error": str(e)}, 400)
            rec = dict(value=value, status=status, at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            with lock:
                ans[iid] = rec
                with open(LOGF, "a") as f: f.write(json.dumps({"id": iid, **rec}) + "\n")
                tmp = ANSF.with_suffix(".tmp"); json.dump(ans, open(tmp, "w"), indent=0); os.replace(tmp, ANSF)
            self._json({"ok": True})
    print(f"serving {PAGE} on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

if __name__ == "__main__":
    if sys.argv[1] == "build": build()
    elif sys.argv[1] == "pages": pages()
    elif sys.argv[1] == "serve": serve(int(sys.argv[2]))
