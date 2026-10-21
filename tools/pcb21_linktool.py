#!/usr/bin/env python3
"""Offline link-labelling tool for 21PCB (owner 2026-10-20: check and fix part -> text links quickly, also box sizes and text
strings, on the laptop without internet). Python standard library only.
Bundle layout (tools/pcb21_link_prep.py): linktool.py (this file), img/<stem>.jpg, data/<stem>.json (pre-links). Edits are saved to
edits/<stem>.json (atomic rewrite on every change) and appended to edits/log.jsonl; data/ is never changed.
Run:  python3 linktool.py [PORT]   (default 8770) -> opens http://localhost:PORT in the browser. Stop with Ctrl+C.
Keys (also listed in the page): Space = all parts in view are checked, go to the next area with unchecked parts; Shift+Space =
previous area; click part, then click text = link (the selection is cleared); Shift+click = select a text inside a part; 0 = part has no text; drag an edge / corner of the selected box or
of its linked text = resize, drag inside = move; Del = delete the selected box; T / P then drag = new text /
part box; E = edit the string; 1-9 = class of the selected part, ` = ignore; U = flag unsure; Q = the selected guess is OK, go to the next unchecked guessed link (orange / red);
F = hide / show unlinked texts; S = string labels on / off; H = hide / show all boxes; [ / ] = previous / next photo; Ctrl+Z = undo; two-finger scroll, drag empty space or arrows = pan;
pinch, Ctrl+wheel or + / - = zoom."""
import json, os, re, sys, threading, webbrowser
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
HERE = Path(__file__).resolve().parent; DATA = HERE/"data"; EDITS = HERE/"edits"; LOCK = threading.Lock()

ok_stem = lambda stem: bool(re.fullmatch(r"[A-Za-z0-9_-]+", stem or "")) and (DATA/f"{stem}.json").exists()   # no path tricks

def doc_path(stem):
    e = EDITS/f"{stem}.json"; return e if e.exists() else DATA/f"{stem}.json"

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=str(HERE), **k)
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if u.path == "/list":
            out = []
            for f in sorted(DATA.glob("*.json")):
                d = json.load(open(doc_path(f.stem)))
                out.append(dict(stem=f.stem, batch=d["batch"], n=len(d["parts"]), checked=sum(1 for p in d["parts"] if p.get("checked"))))
            return self._json(out)
        if u.path == "/doc":
            stem = parse_qs(u.query).get("stem", [""])[0]
            if not ok_stem(stem): return self._json({"error": "no such photo"}, 404)
            return self._json(json.load(open(doc_path(stem))))
        return super().do_GET()
    def do_POST(self):
        if urlparse(self.path).path != "/save": return self._json({"error": "not found"}, 404)
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))); stem = d["stem"]
            assert ok_stem(stem) and isinstance(d["parts"], list) and isinstance(d["texts"], list)
        except Exception as e: return self._json({"error": repr(e)}, 400)
        at = datetime.now(timezone.utc).isoformat(timespec="seconds"); d["saved_at"] = at
        with LOCK:
            EDITS.mkdir(exist_ok=True); tmp = EDITS/f"{stem}.tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, EDITS/f"{stem}.json")
            with open(EDITS/"log.jsonl", "a") as f:
                f.write(json.dumps(dict(stem=stem, at=at, parts=len(d["parts"]), texts=len(d["texts"]),
                                        checked=sum(1 for p in d["parts"] if p.get("checked")))) + "\n")
        self._json({"ok": True, "at": at})

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>21PCB Link Tool</title><style>
:root{--bg:#14171a;--side:#1b2024;--line:#2c343a;--fg:#e3e8ec;--dim:#93a0ab;--act:#ffd54a;--bad:#ff5a50}
html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif;overflow:hidden}
#wrap{display:flex;height:100%}#cv{flex:1;min-width:0;display:block;cursor:crosshair}
#side{width:290px;flex:none;background:var(--side);border-left:1px solid var(--line);padding:10px;overflow:auto}
h1{font-size:15px;margin:0 0 6px}select,input,button{font:inherit;background:#0e1113;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:3px 6px}
select{width:100%}.dim{color:var(--dim)}.box{border:1px solid var(--line);border-radius:6px;padding:7px;margin:8px 0}
.cls{display:grid;grid-template-columns:repeat(2,1fr);gap:3px}.cls button{text-align:left;padding:2px 5px;font-size:12px}
.cls button.on{outline:2px solid var(--act)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#err{display:none;background:#5a1d1d;color:#ffd0d0;padding:5px;border-radius:5px;margin:6px 0}
#strin{width:100%;font:600 16px ui-monospace,monospace;box-sizing:border-box}.k{font-size:12px;line-height:1.55}.k b{color:var(--fg)}
#mode{font-weight:700;color:var(--act)}
</style></head><body><div id="wrap"><canvas id="cv" tabindex="0"></canvas><div id="side">
<h1>21PCB link tool</h1><select id="photo"></select>
<div id="prog" class="dim" style="margin-top:4px"></div><div style="margin-top:6px"><button id="strbtn"></button> <button id="hidebtn"></button></div><div id="err"></div><div id="mode"></div>
<div class="box" id="sel"><span class="dim">Nothing selected. Click a part or a text.</span></div>
<div class="box k dim">
<b>Space</b> parts in view are OK, next area &middot; <b>Shift+Space</b> back<br>
<b>click part</b>, then <b>click text</b> = link it (then nothing is selected)<br><b>Shift+click</b> = select a text inside a part<br><b>0</b> part has no printed text<br>
<b>drag an edge / corner</b> (yellow squares) of the selected box or of its linked text = resize; <b>drag inside</b> the selected box = move<br><b>Del</b> delete the selected box<br>
<b>T</b> + drag = new text box (linked to the selected part) &middot; <b>P</b> + drag = new part<br>
<b>E</b> edit the string &middot; <b>1-9</b> class, <b>`</b> ignore &middot; <b>U</b> unsure flag<br>
<b>S</b> strings on/off &middot; <b>H</b> hide/show all boxes<br><b>Q</b> current guess is OK, go to the next unchecked guess (orange / red) &middot; <b>F</b> hide/show unused texts<br>
<b>[ ]</b> photo &middot; <b>Ctrl+Z</b> undo<br><b>two-finger scroll</b> / drag / arrows = pan &middot; <b>pinch</b> / Ctrl+scroll / + - = zoom</div>
<div class="box k dim">Lines: <span style="color:#3ddc84">green</span> = designator matches the string,
<span style="color:#56c7c7">teal</span> = rule link (BoM), <span style="color:#ff9f1a">orange</span> = guessed by distance,
<span style="color:#ffffff">white</span> = set by you. <span style="color:#ff5a50">Red X</span> = part without text.
Thin dim boxes = checked parts.</div></div></div>
<script>
const CLS=["resistor","capacitor","inductor","connector","diode","switch","transistor","IC","oscillator"];
const COL={resistor:"#ff3c3c",capacitor:"#288cff",inductor:"#ffa000",connector:"#aa5aff",diode:"#00dc6e",switch:"#ff00c8",transistor:"#00e6e6",IC:"#ffff00",oscillator:"#ffffff",ignore:"#8c8c8c"};
const LCOL={match:"#3ddc84",rule:"#56c7c7",near:"#ff9f1a",owner:"#ffffff"};
const cv=document.getElementById("cv"),ctx=cv.getContext("2d"),$=s=>document.querySelector(s);
let list=[],doc=null,img=new Image(),view={s:1,ox:0,oy:0},sel=null,mode="",undo=[],hideFree=false,drag=null,pend={},saving=false,area=null,showStr=true,hideAll=false;
try{showStr=localStorage.getItem("pcb21_link_str")!=="off"}catch(e){}
const P=id=>doc.parts.find(p=>p.id===id),T=id=>doc.texts.find(t=>t.id===id);
function resize(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function W(){return cv.getBoundingClientRect().width}function Hh(){return cv.getBoundingClientRect().height}
const sx=x=>(x-view.ox)*view.s,sy=y=>(y-view.oy)*view.s,ix=x=>x/view.s+view.ox,iy=y=>y/view.s+view.oy;
function cen(b){return[(b[0]+b[2])/2,(b[1]+b[3])/2]}
function medSize(){const a=doc.parts.map(p=>Math.max(p.box[2]-p.box[0],p.box[3]-p.box[1])).sort((a,b)=>a-b);return a[a.length>>1]||50}
function linkedTexts(){const s=new Set();doc.parts.forEach(p=>p.link&&s.add(p.link));return s}
function draw(){if(!doc)return;const w=W(),h=Hh();ctx.fillStyle="#000";ctx.fillRect(0,0,w,h);
 if(img.complete&&img.naturalWidth)ctx.drawImage(img,-view.ox*view.s,-view.oy*view.s,doc.iw*view.s,doc.ih*view.s);
 if(hideAll){ctx.fillStyle="rgba(0,0,0,.6)";ctx.fillRect(8,8,250,24);ctx.fillStyle="#ffd54a";ctx.font="600 13px system-ui";ctx.fillText("Boxes hidden (H to show)",16,25);return}
 const x0=ix(0),y0=iy(0),x1=ix(w),y1=iy(h),vis=b=>b[2]>=x0&&b[0]<=x1&&b[3]>=y0&&b[1]<=y1,lt=linkedTexts(),big=view.s*14;
 ctx.lineWidth=1;for(const t of doc.texts){if(!vis(t.box))continue;const L=lt.has(t.id);if(!L&&hideFree&&!(sel&&sel.k==="t"&&sel.id===t.id))continue;
  ctx.setLineDash(L?[]:[4,3]);ctx.strokeStyle=L?"rgba(255,255,255,.9)":"rgba(200,200,200,.55)";ctx.lineWidth=L?1.5:1;
  ctx.strokeRect(sx(t.box[0]),sy(t.box[1]),(t.box[2]-t.box[0])*view.s,(t.box[3]-t.box[1])*view.s);
  if(showStr&&(t.box[3]-t.box[1])*view.s>11&&(L||!hideFree)){ctx.setLineDash([]);ctx.font="600 12px ui-monospace,monospace";const s=t.str||"(empty)",tw=ctx.measureText(s).width;
   ctx.fillStyle="rgba(0,0,0,.75)";ctx.fillRect(sx(t.box[0]),sy(t.box[1])-15,tw+6,14);ctx.fillStyle=L?"#7fe7ff":"#b8c4cc";ctx.fillText(s,sx(t.box[0])+3,sy(t.box[1])-4)}}
 ctx.setLineDash([]);
 for(const p of doc.parts){const t=p.link&&T(p.link);if(!t||!(vis(p.box)||vis(t.box)))continue;const a=cen(p.box),b=cen(t.box);
  ctx.strokeStyle=LCOL[p.how]||"#fff";ctx.lineWidth=p.checked?1.2:2.2;ctx.beginPath();ctx.moveTo(sx(a[0]),sy(a[1]));ctx.lineTo(sx(b[0]),sy(b[1]));ctx.stroke()}
 for(const p of doc.parts){if(!vis(p.box))continue;ctx.strokeStyle=COL[p.cls]||"#ccc";ctx.globalAlpha=p.checked?.55:1;ctx.lineWidth=p.checked?1:2;
  ctx.strokeRect(sx(p.box[0]),sy(p.box[1]),(p.box[2]-p.box[0])*view.s,(p.box[3]-p.box[1])*view.s);ctx.globalAlpha=1;
  if(!p.link){const c=cen(p.box),r=Math.max(4,Math.min(9,(p.box[2]-p.box[0])*view.s/3));ctx.strokeStyle=p.checked?"#a05050":"#ff5a50";ctx.lineWidth=2;ctx.beginPath();
   ctx.moveTo(sx(c[0])-r,sy(c[1])-r);ctx.lineTo(sx(c[0])+r,sy(c[1])+r);ctx.moveTo(sx(c[0])+r,sy(c[1])-r);ctx.lineTo(sx(c[0])-r,sy(c[1])+r);ctx.stroke()}
  if(showStr&&p.des&&!p.checked&&(p.how==="near"||!p.link)&&(p.box[3]-p.box[1])*view.s>10){ctx.font="600 11px ui-monospace,monospace";ctx.fillStyle="#ffd54a";ctx.fillText(p.des,sx(p.box[0]),sy(p.box[3])+12)}
  if(p.flag){ctx.fillStyle="#ff00ff";ctx.font="700 14px system-ui";ctx.fillText("?",sx(p.box[2])+2,sy(p.box[1])+12)}}
 if(sel){const o=sel.k==="p"?P(sel.id):T(sel.id);if(o){const b=o.box;ctx.strokeStyle="#ffd54a";ctx.lineWidth=2.5;ctx.setLineDash([6,3]);
  ctx.strokeRect(sx(b[0])-3,sy(b[1])-3,(b[2]-b[0])*view.s+6,(b[3]-b[1])*view.s+6);ctx.setLineDash([]);ctx.fillStyle="#ffd54a";
  if(sel.k==="p"&&o.link){const t=T(o.link);if(t){ctx.strokeStyle="#ffd54a";ctx.lineWidth=2;ctx.strokeRect(sx(t.box[0])-2,sy(t.box[1])-2,(t.box[2]-t.box[0])*view.s+4,(t.box[3]-t.box[1])*view.s+4)}}
  for(const c of cands()){const q=c.box,mx=(q[0]+q[2])/2,my=(q[1]+q[3])/2;ctx.fillStyle="#ffd54a";
   for(const[x,y]of[[q[0],q[1]],[q[2],q[1]],[q[0],q[3]],[q[2],q[3]],[mx,q[1]],[mx,q[3]],[q[0],my],[q[2],my]])ctx.fillRect(sx(x)-4,sy(y)-4,8,8)}}}
 if(area){ctx.strokeStyle="rgba(255,213,74,.35)";ctx.lineWidth=1;ctx.setLineDash([10,6]);ctx.strokeRect(sx(area[0]),sy(area[1]),(area[2]-area[0])*view.s,(area[3]-area[1])*view.s);ctx.setLineDash([])}
 if(drag&&drag.kind==="new"){const b=norm(drag.b);ctx.strokeStyle=drag.what==="t"?"#fff":"#ffd54a";ctx.lineWidth=2;ctx.strokeRect(sx(b[0]),sy(b[1]),(b[2]-b[0])*view.s,(b[3]-b[1])*view.s)}}
function norm(b){return[Math.min(b[0],b[2]),Math.min(b[1],b[3]),Math.max(b[0],b[2]),Math.max(b[1],b[3])]}
let raf=0;function redraw(){if(!raf)raf=requestAnimationFrame(()=>{raf=0;draw()})}
function prog(){const n=doc.parts.length,c=doc.parts.filter(p=>p.checked).length,g=doc.parts.filter(p=>!p.checked&&(p.how==="near"||!p.link)).length;
 const it=list.find(x=>x.stem===doc.stem);if(it)it.checked=c;const N=list.reduce((a,x)=>a+x.n,0),C=list.reduce((a,x)=>a+x.checked,0);
 $("#prog").textContent=`this photo: ${c} / ${n} checked, ${g} unchecked guesses · all: ${C} / ${N}`;
 [...$("#photo").options].forEach((o,i)=>{const x=list[i];o.textContent=`${x.checked>=x.n?"✓ ":""}${x.stem} (b${x.batch}) ${x.checked}/${x.n}`})}
function showSel(){const e=$("#sel");if(!sel){e.innerHTML='<span class="dim">Nothing selected. Click a part or a text.</span>';return}
 if(sel.k==="p"){const p=P(sel.id);if(!p){sel=null;return showSel()}const t=p.link&&T(p.link);
  e.innerHTML=`<div><b>Part</b> ${p.des?"<b>"+esc(p.des)+"</b>":"(no designator)"} · ${p.checked?"checked":"<span style='color:#ffd54a'>unchecked</span>"}${p.flag?" · <span style='color:#f0f'>unsure</span>":""}</div>
  <div class="dim">link: ${t?"<b style='color:#7fe7ff'>"+esc(t.str||"(empty)")+"</b> ("+p.how+")":"<span style='color:#ff5a50'>no text</span>"}</div>
  <div class="cls" style="margin-top:6px">${CLS.concat(["ignore"]).map((c,i)=>`<button data-c="${c}" class="${p.cls===c?"on":""}"><span class="sw" style="background:${COL[c]}"></span>${i<9?i+1:"`"} ${c}</button>`).join("")}</div>
  ${t?`<div style="margin-top:6px" class="dim">text string (E):</div><input id="strin" value="${esc(t.str)}">`:""}`;
  e.querySelectorAll(".cls button").forEach(b=>b.onclick=()=>setCls(b.dataset.c))}
 else{const t=T(sel.id);if(!t){sel=null;return showSel()}const ps=doc.parts.filter(p=>p.link===t.id);
  e.innerHTML=`<div><b>Text</b> · ${ps.length?"linked to "+ps.map(p=>esc(p.des||p.cls)).join(", "):"not linked"} · ${t.src}</div><div class="dim" style="margin-top:4px">string (E):</div><input id="strin" value="${esc(t.str)}">`}
 const si=$("#strin");if(si){si.onkeydown=ev=>{ev.stopPropagation();if(ev.key==="Enter"){setStr(si.value);si.blur();cv.focus()}else if(ev.key==="Escape"){si.blur();showSel()}}}}
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function push(){undo.push(JSON.stringify({parts:doc.parts,texts:doc.texts}));if(undo.length>300)undo.shift()}
function changed(){pend[doc.stem]=JSON.stringify(doc);prog();showSel();redraw();save()}
async function save(){if(saving)return;saving=true;while(Object.keys(pend).length){const stem=Object.keys(pend)[0],body=pend[stem];delete pend[stem];
 try{const r=await fetch("save",{method:"POST",headers:{"Content-Type":"application/json"},body});if(!r.ok)throw 0;$("#err").style.display="none"}
 catch(e){if(!(stem in pend))pend[stem]=body;$("#err").style.display="block";$("#err").textContent="NOT SAVED: is the terminal with linktool.py still running? Retrying.";await new Promise(r=>setTimeout(r,3000))}}saving=false}
window.addEventListener("beforeunload",e=>{if(Object.keys(pend).length){e.preventDefault();e.returnValue=""}});
function setLink(p,tid){push();const nm=q=>q.des||q.cls;let msg=tid?`Linked ${nm(p)} -> "${(T(tid)||{}).str||""}"`:`${nm(p)}: no text`;
 if(tid)doc.parts.forEach(q=>{if(q!==p&&q.link===tid){q.link=null;q.how="owner";q.checked=false;msg+=` · this text was taken from ${nm(q)}, which now has NO text`}});
 p.link=tid;p.how="owner";p.checked=true;changed();flash(msg)}
function setCls(c){const p=sel&&sel.k==="p"&&P(sel.id);if(!p)return;push();p.cls=c;p.checked=true;changed()}
function setStr(s){let t=null;if(sel&&sel.k==="t")t=T(sel.id);else if(sel&&sel.k==="p"){const p=P(sel.id);t=p&&p.link&&T(p.link)}if(!t)return;push();t.str=s.trim();t.edited=true;changed()}
function delSel(){if(!sel)return;push();if(sel.k==="p")doc.parts=doc.parts.filter(p=>p.id!==sel.id);else{const id=sel.id;doc.texts=doc.texts.filter(t=>t.id!==id);
 doc.parts.forEach(p=>{if(p.link===id){p.link=null;p.how="owner";p.checked=false}})}sel=null;changed()}
function smallest(arr,k,x,y){let best=null,ba=1e18;for(const o of arr){const b=o.box;if(x>=b[0]&&x<=b[2]&&y>=b[1]&&y<=b[3]){const a=(b[2]-b[0])*(b[3]-b[1]);if(a<ba){ba=a;best={k,id:o.id}}}}return best}
function hitT(x,y){const lt=linkedTexts();return smallest(doc.texts.filter(t=>!hideFree||lt.has(t.id)),"t",x,y)}
function hitP(x,y){return smallest(doc.parts,"p",x,y)}
function hit(x,y){const a=hitT(x,y),b=hitP(x,y);if(!a||!b)return a||b;const A=T(a.id).box,B=P(b.id).box;return (A[2]-A[0])*(A[3]-A[1])<(B[2]-B[0])*(B[3]-B[1])?a:b}
function cands(){const c=[];if(!sel)return c;const o=sel.k==="p"?P(sel.id):T(sel.id);if(!o)return c;c.push(o);
 if(sel.k==="p"&&o.link){const t=T(o.link);if(t)c.push(t)}return c}                 // the selected box and the text linked to a selected part
function handle(mx,my){const tol=8;for(const o of cands()){const b=o.box,L=sx(b[0]),R=sx(b[2]),Tp=sy(b[1]),B=sy(b[3]);
  if(mx<L-tol||mx>R+tol||my<Tp-tol||my>B+tol)continue;
  const ex=Math.abs(mx-L)<tol?0:Math.abs(mx-R)<tol?2:null,ey=Math.abs(my-Tp)<tol?1:Math.abs(my-B)<tol?3:null;
  if(ex!==null||ey!==null)return{o,ex,ey}}return null}                                  // an edge (ex or ey) or a corner (both)
function mpos(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
cv.addEventListener("mousemove",e=>{if(drag||!doc)return;const[mx,my]=mpos(e),h=mode?null:handle(mx,my);
 cv.style.cursor=h?(h.ex!==null&&h.ey!==null?((h.ex===0)===(h.ey===1)?"nwse-resize":"nesw-resize"):h.ex!==null?"ew-resize":"ns-resize"):"crosshair"});
cv.addEventListener("mousedown",e=>{if(!doc)return;const[mx,my]=mpos(e),x=ix(mx),y=iy(my);
 if(mode){drag={kind:"new",what:mode,b:[x,y,x,y]};return}
 const h=handle(mx,my);if(h){push();drag={kind:"resize",h};return}
 const o=sel&&(sel.k==="p"?P(sel.id):T(sel.id)),ht=hitT(x,y),hp=hitP(x,y);
 if(o&&!e.shiftKey&&x>=o.box[0]&&x<=o.box[2]&&y>=o.box[1]&&y<=o.box[3]&&(sel.k==="t"?(ht&&ht.id===sel.id):!ht)){
  drag={kind:"move",x,y,b:o.box.slice(),moved:false,pushed:false};return}
 if(sel&&sel.k==="p"&&ht&&!e.shiftKey){setLink(P(sel.id),ht.id);sel=null;showSel();redraw();return}   // linked: start the next pair fresh
 if(e.shiftKey&&ht){sel=ht;showSel();redraw();return}                                    // Shift+click = the text, even inside a part
 const k=hp||ht;if(k){sel=k;showSel();redraw();return}                                  // nothing selected: parts first
 drag={kind:"pan",mx,my,ox:view.ox,oy:view.oy,moved:false}});
window.addEventListener("mousemove",e=>{if(!drag)return;const[mx,my]=mpos(e),x=ix(mx),y=iy(my);
 if(drag.kind==="pan"){view.ox=drag.ox-(mx-drag.mx)/view.s;view.oy=drag.oy-(my-drag.my)/view.s;if(Math.abs(mx-drag.mx)+Math.abs(my-drag.my)>3)drag.moved=true;redraw()}
 else if(drag.kind==="new"){drag.b[2]=x;drag.b[3]=y;redraw()}
 else if(drag.kind==="resize"){const h=drag.h;if(h.ex!==null)h.o.box[h.ex]=x;if(h.ey!==null)h.o.box[h.ey]=y;redraw()}
 else if(drag.kind==="move"){const o=sel.k==="p"?P(sel.id):T(sel.id);if(!drag.pushed){push();drag.pushed=true}const dx=x-drag.x,dy=y-drag.y;o.box=[drag.b[0]+dx,drag.b[1]+dy,drag.b[2]+dx,drag.b[3]+dy];drag.moved=true;redraw()}});
window.addEventListener("mouseup",e=>{if(!drag)return;const d=drag;drag=null;
 if(d.kind==="pan"&&!d.moved){sel=null;showSel();redraw()}
 else if(d.kind==="resize"){const o=d.h.o;o.box=norm(o.box);if(doc.parts.includes(o))o.checked=true;changed()}
 else if(d.kind==="move"&&d.moved){const o=sel.k==="p"?P(sel.id):T(sel.id);if(sel.k==="p")o.checked=true;changed()}
 else if(d.kind==="new"){const b=norm(d.b);mode="";$("#mode").textContent="";if((b[2]-b[0])*view.s<4||(b[3]-b[1])*view.s<4){redraw();return}push();
  if(d.what==="t"){const id="t"+(Date.now()%1e9);doc.texts.push({id,box:b,str:"",src:"new"});const p=sel&&sel.k==="p"&&P(sel.id);
   if(p){doc.parts.forEach(q=>{if(q.link===id)q.link=null});p.link=id;p.how="owner";p.checked=true}sel={k:"t",id};changed();const si=$("#strin");if(si){si.focus()}}
  else{const id="p"+(Date.now()%1e9);doc.parts.push({id,box:b,cls:"resistor",des:"",link:null,how:"owner",checked:true,added:true});sel={k:"p",id};changed()}}});
cv.addEventListener("wheel",e=>{e.preventDefault();if(!doc)return;const[mx,my]=mpos(e),x=ix(mx),y=iy(my),u=e.deltaMode?20:1;
 if(e.ctrlKey||e.metaKey){const f=Math.exp(-Math.max(-60,Math.min(60,e.deltaY*u))*0.01);           // pinch or Ctrl+wheel = zoom at the cursor
  view.s=Math.min(8,Math.max(0.02,view.s*f));view.ox=x-mx/view.s;view.oy=y-my/view.s}
 else{view.ox+=e.deltaX*u/view.s;view.oy+=e.deltaY*u/view.s}redraw()},{passive:false});          // two-finger scroll = pan
function viewRect(){return[ix(0),iy(0),ix(W()),iy(Hh())]}
function areas(){const s=45/medSize(),w=W()/s,h=Hh()/s,out=[];const stepx=w*0.9,stepy=h*0.9;
 for(let y=0;y<doc.ih;y+=stepy)for(let x=0;x<doc.iw;x+=stepx)out.push([x,y,x+w,y+h]);return{s,out}}
function inside(p,r){const c=cen(p.box);return c[0]>=r[0]&&c[0]<=r[2]&&c[1]>=r[1]&&c[1]<=r[3]}
function gotoArea(r,s){view.s=s;view.ox=r[0];view.oy=r[1];area=r;redraw()}
function nextArea(back,fromStart){const{s,out}=areas(),vr=viewRect(),cx=(vr[0]+vr[2])/2,cy=(vr[1]+vr[3])/2;
 let cur=fromStart?-1:out.findIndex(r=>cx>=r[0]&&cx<r[0]+(r[2]-r[0])*0.9+1&&cy>=r[1]&&cy<r[1]+(r[3]-r[1])*0.9+1);if(cur<0&&!fromStart)cur=0;
 if(back){for(let i=cur-1;i>=0;i--){if(doc.parts.some(p=>inside(p,out[i])))return gotoArea(out[i],s)}return}
 for(let i=cur+1;i<out.length;i++){if(doc.parts.some(p=>!p.checked&&inside(p,out[i])))return gotoArea(out[i],s)}
 const left=doc.parts.find(p=>!p.checked);if(left){const i=out.findIndex(r=>inside(left,r));if(i>=0)return gotoArea(out[i],s)}
 area=null;fit();flash("All parts of this photo are checked. Press ] for the next photo.")}
function acceptView(){const r=viewRect(),ps=doc.parts.filter(p=>!p.checked&&inside(p,r));if(ps.length){push();ps.forEach(p=>p.checked=true);changed()}nextArea(false)}
function nextGuess(){const cur=sel&&sel.k==="p"&&P(sel.id);if(cur&&!cur.checked){push();cur.checked=true;changed()}
 const g=doc.parts.filter(p=>!p.checked&&(p.how==="near"||!p.link));if(!g.length){flash("No unchecked guesses left in this photo.");return}
 const p=g[0],c=cen(p.box),s=Math.max(view.s,60/medSize());view.s=s;view.ox=c[0]-W()/2/s;view.oy=c[1]-Hh()/2/s;sel={k:"p",id:p.id};area=null;showSel();redraw()}
function fit(){const s=Math.min(W()/doc.iw,Hh()/doc.ih);view.s=s;view.ox=-(W()/s-doc.iw)/2;view.oy=-(Hh()/s-doc.ih)/2;redraw()}
function setStr2(v){showStr=v;try{localStorage.setItem("pcb21_link_str",v?"on":"off")}catch(e){}$("#strbtn").textContent="Strings: "+(v?"on":"off")+" (S)";redraw()}
function setHide(v){hideAll=v;$("#hidebtn").textContent=(v?"Show":"Hide")+" boxes (H)";redraw()}
let ft=0;function flash(m){$("#mode").textContent=m;clearTimeout(ft);ft=setTimeout(()=>{if(!mode)$("#mode").textContent=""},3500)}
document.addEventListener("keydown",e=>{if(!doc)return;if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;const k=e.key;
 if(k===" "){e.preventDefault();e.shiftKey?nextArea(true):acceptView()}
 else if((e.ctrlKey||e.metaKey)&&(k==="z"||k==="Z")){e.preventDefault();const u=undo.pop();if(u){const d=JSON.parse(u);doc.parts=d.parts;doc.texts=d.texts;changed()}}
 else if(k==="0"&&sel&&sel.k==="p"){setLink(P(sel.id),null)}
 else if((k==="Delete"||k==="Backspace")&&sel){e.preventDefault();delSel()}
 else if(k==="t"||k==="T"){mode="t";$("#mode").textContent="Drag a new TEXT box"+(sel&&sel.k==="p"?" (it will be linked to the selected part)":"")+" · Esc cancels"}
 else if(k==="p"||k==="P"){mode="p";$("#mode").textContent="Drag a new PART box · Esc cancels"}
 else if(k==="Escape"){mode="";$("#mode").textContent="";sel=null;showSel();redraw()}
 else if(k==="e"||k==="E"){const si=$("#strin");if(si){e.preventDefault();si.focus();si.select()}}
 else if(/^[1-9]$/.test(k)&&sel&&sel.k==="p"){setCls(CLS[+k-1])}else if(k==="`"&&sel&&sel.k==="p"){setCls("ignore")}
 else if((k==="u"||k==="U")&&sel&&sel.k==="p"){push();const p=P(sel.id);p.flag=!p.flag;changed()}
 else if(k==="s"||k==="S"){setStr2(!showStr)}else if(k==="h"||k==="H"){setHide(!hideAll)}
 else if(k==="q"||k==="Q"){nextGuess()}else if(k==="f"||k==="F"){hideFree=!hideFree;flash(hideFree?"Unused texts hidden":"Unused texts shown");redraw()}
 else if(k==="]"||k==="["){const i=list.findIndex(x=>x.stem===doc.stem)+(k==="]"?1:-1);if(i>=0&&i<list.length)load(list[i].stem)}
 else if(k.startsWith("Arrow")){e.preventDefault();const d=120/view.s;if(k==="ArrowLeft")view.ox-=d;if(k==="ArrowRight")view.ox+=d;if(k==="ArrowUp")view.oy-=d;if(k==="ArrowDown")view.oy+=d;redraw()}
 else if(k==="+"||k==="="){view.s*=1.25;redraw()}else if(k==="-"){view.s/=1.25;redraw()}});
async function load(stem){doc=await (await fetch("doc?stem="+encodeURIComponent(stem))).json();undo=[];sel=null;area=null;showSel();
 $("#photo").value=stem;try{localStorage.setItem("pcb21_link_last",stem)}catch(e){}img=new Image();img.onload=()=>{fit();nextArea(false,true)};img.src=doc.img;prog();cv.focus()}
(async()=>{list=await (await fetch("list")).json();$("#photo").innerHTML=list.map(x=>`<option value="${x.stem}">${x.stem}</option>`).join("");
 $("#photo").onchange=()=>load($("#photo").value);$("#strbtn").onclick=()=>{setStr2(!showStr);$("#strbtn").blur()};$("#hidebtn").onclick=()=>{setHide(!hideAll);$("#hidebtn").blur()};setStr2(showStr);setHide(false);window.addEventListener("resize",resize);resize();let last=null;try{last=localStorage.getItem("pcb21_link_last")}catch(e){}
 load(list.some(x=>x.stem===last)?last:list[0].stem)})();
</script></body></html>"""

if __name__ == "__main__":
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8770); srv = None
    for port in range(port, port + 20):                      # a busy port -> try the next one
        try: srv = ThreadingHTTPServer(("127.0.0.1", port), H); break
        except OSError: print(f"port {port} is busy, trying {port + 1}", flush=True)
    if srv is None: sys.exit("no free port found")
    print(f"21PCB link tool: http://localhost:{port}   (edits are saved in {EDITS}; stop with Ctrl+C)", flush=True)
    if "--no-browser" not in sys.argv: threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("stopped")
