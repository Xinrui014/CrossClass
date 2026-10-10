#!/usr/bin/env python3
"""Build the 'Who Gets the Class Right' page: <page>/compare.json (err_compare.py) + <page>/img/*.jpg + <page>/summary.html
-> <page>/index.html.  Metrics rows are the frozen 42-test numbers from results/qwen_label, results/qwen_class, results/assoc."""
import json, sys
from pathlib import Path
PAGE = Path(sys.argv[1]); d = json.load(open(PAGE/"compare.json"))
d["metrics"] = json.load(open(PAGE/"metrics.json"))   # rows written by hand from the frozen 42-test numbers
t = open(Path(__file__).parent/"compare_template.html").read().replace("/*DATA*/", json.dumps(d, ensure_ascii=False))
t = t.replace("{{SUMMARY}}", open(PAGE/"summary.html").read() if (PAGE/"summary.html").exists() else "<p>结果整理中。</p>")
open(PAGE/"index.html", "w").write(t); print("built", d["n_parts"], "parts,", sum(len(v) for v in d["samples"].values()), "samples")
