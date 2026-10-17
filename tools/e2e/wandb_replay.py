#!/usr/bin/env python3
"""Upload LinkFormer runs trained without wandb (intern02: no wandb in that env) to wandb from their result JSON (results/linkformer/<tag>.json):
the eval log (every --eval_every steps: running loss, kvalsub / clean_dev3 / dev3 metrics) and the final metrics. x axis: optimizer step,
plus 'boards' = boards seen. Usage: wandb_replay.py --group v2d2 TAG [TAG ...]   (python: ~/venvs/gdino_mm on the L40)"""
import argparse, json, os
import wandb
ap = argparse.ArgumentParser(); ap.add_argument("--group", default="v2d2"); ap.add_argument("--idprefix", default="replay"); ap.add_argument("tags", nargs="+"); a = ap.parse_args()
ROOT = os.path.expanduser("~/projects/crossclass")
for tag in a.tags:
    R = json.load(open(f"{ROOT}/results/linkformer/{tag}.json")); A = R["args"]; bps = A.get("bps") or A.get("accum", 1)
    run = wandb.init(entity="xinruiworkstation", project="crossclass", group=a.group, name=tag, id=f"{a.idprefix}-{tag}".replace(".", "p"), resume="allow",
                     config=dict(A, params=R["params"], ranks=R["ranks"], replayed_from="results/linkformer JSON (trained on intern02 without wandb)"))
    run.define_metric("boards"); run.define_metric("*", step_metric="boards")
    for e in R["log"]:
        row = {"boards": e["step"]*bps, "train/step": e["step"], "loss": e["loss"], **{f"loss/{k}": v for k, v in (e.get("loss_parts") or {}).items()}}
        for s, m in e.items():
            if isinstance(m, dict): row.update({f"{s}/{k}": m[k] for k in ("acc", "prec", "f1", "cls_head", "cls_rule") if k in m})
        run.log(row)
    run.summary.update({f"final/{s}/{k}": v["summary"][k] for s, v in R["final"].items() for k in ("acc", "prec", "f1", "cls_head", "cls_rule") if k in v["summary"]})
    run.summary["train_seconds"] = R["seconds"]; print(tag, run.url); run.finish()
