#!/usr/bin/env python3
"""Run identity for CrossClass results (owner 2026-10-22: save each run's configuration).
start(): code (git commit + changed tracked files, md5 of tools/linkformer.py and of the running script), raw sys.argv, host, python /
torch version, SLURM job id, CUDA_VISIBLE_DEVICES, start time (clock, ISO with time zone).
fingerprint(path): full md5 on every host, one method everywhere: a file = its md5; a folder = md5 over the sorted
'<relpath> <file md5>' lines, plus file count and bytes; each entry keeps the seconds it took.
sidecar(result, ckpts, ...): <result stem>.meta.json next to the result of a script that loads linker ckpts (result formats unchanged).
git runs only on the host of the run: a code tree on an sshfs mount, or one without .git (intern02: git archive), gets commit None."""
import datetime, hashlib, json, os, platform, socket, subprocess, sys, time
from pathlib import Path
CODE = Path(__file__).resolve().parent                 # the tools/ folder this module was imported from
SCHEMA = 1                                             # ckpts with meta (schema >= 1) must carry every architecture key

def now(): return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
STARTED = now()                                        # import time = the start of the script that imports this module

def md5_file(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(bs), b""): h.update(blk)
    return h.hexdigest()

def fingerprint(p):
    """Full md5 of a file, or of a folder (md5 over the sorted '<relpath> <file md5>' lines); md5 None = path missing."""
    t = time.time(); p = Path(p)
    if p.is_file(): r = dict(md5=md5_file(p), files=1, bytes=p.stat().st_size)
    elif p.is_dir():
        fs = sorted((f.relative_to(p).as_posix(), f) for f in p.rglob("*") if f.is_file())
        r = dict(md5=hashlib.md5("\n".join(f"{n} {md5_file(f)}" for n, f in fs).encode()).hexdigest(), files=len(fs), bytes=sum(f.stat().st_size for _, f in fs))
    else: r = dict(md5=None, files=0, bytes=0)
    return dict(path=str(p), **r, sec=round(time.time()-t, 2))

def fingerprints(paths):
    """{name: fingerprint} for {name: path} (empty paths skipped)."""
    return {k: fingerprint(v) for k, v in paths.items() if v}

def on_sshfs(p):
    try: mounts = [l.split()[1] for l in open("/proc/mounts") if l.split()[2] == "fuse.sshfs"]
    except OSError: return False
    return any(str(p) == m or str(p).startswith(m.rstrip("/") + "/") for m in mounts)

def git(root=CODE):
    """Commit and changed tracked files of the code tree (git on this host only)."""
    if on_sshfs(root): return dict(commit=None, dirty=None, changed=[], note="sshfs mount: git skipped")
    try:
        run = lambda *c: subprocess.run(["git", "-C", str(root), *c], capture_output=True, text=True, timeout=60, check=True).stdout
        ch = [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines()]
        return dict(commit=run("rev-parse", "HEAD").strip(), dirty=bool(ch), dirty_tools=any(f.startswith("tools/") for f in ch), changed=ch[:50])
    except Exception as ex: return dict(commit=None, dirty=None, changed=[], note=repr(ex)[:120])

def gpu(dev=None):
    """Name of the CUDA device a run used (None on CPU)."""
    import torch
    dev = torch.device(dev) if dev is not None else None
    return torch.cuda.get_device_name(dev) if dev is not None and dev.type == "cuda" else None

def start(script=None):
    """Identity of this process: code, command line, host, versions; the caller adds inputs, gpu, end."""
    import torch
    s = Path(script or sys.argv[0]).resolve()
    return dict(schema=SCHEMA, start=STARTED, argv=list(sys.argv), cwd=os.getcwd(), script=str(s), script_md5=md5_file(s) if s.is_file() else None,
                linkformer_md5=md5_file(CODE/"linkformer.py"), git=git(), host=socket.gethostname(), python=platform.python_version(),
                torch=str(torch.__version__), slurm_job=os.environ.get("SLURM_JOB_ID"), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                crossclass=os.environ.get("CROSSCLASS"))

def ckpt_info(path):
    """Identity of a linker ckpt: path, md5, step, tag and meta schema (schema None = old ckpt without meta)."""
    import torch
    ck = torch.load(path, map_location="cpu", mmap=True); A = ck.get("args", {}); m = ck.get("meta") or {}
    return dict(path=str(Path(path).resolve()), md5=md5_file(path), step=ck.get("step"), steps=A.get("steps"), tag=A.get("tag") or None, schema=m.get("schema"))

def sidecar(result, ckpts=(), dev=None, inputs=None, **extra):
    """Write <result stem>.meta.json: process identity, linker ckpts, input fingerprints, extra keys (decode mode, none bias ...)."""
    m = start(); m.update(ckpts=[ckpt_info(c) for c in ckpts], inputs=fingerprints(inputs or {}), gpu=gpu(dev), **extra)
    m["fingerprint_seconds"] = round(sum(v["sec"] for v in m["inputs"].values()), 1); m["end"] = now()
    out = Path(result).with_suffix(".meta.json"); json.dump(m, open(out, "w"), indent=1); return out
