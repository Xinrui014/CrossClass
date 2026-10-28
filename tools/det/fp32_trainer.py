#!/usr/bin/env python3
"""FP32-safe ultralytics DetectionTrainer for the detector re-runs (cros-det 2026-10-28; main 16:12, owner "fix the code first, then re-run").
Problem (rs60b det_realsyn_v8lp2_e60b, read in the ultralytics 8.4.37 source): BN buffer model.28.cv3.3.0.bn.running_var grew past the
fp16 max 65,504 at epoch 28. (1) engine/validator.py "Force FP16 val during training" calls trainer.ema.ema.half() IN PLACE and
.float() after the val, so the value became Inf in the EMA itself and stayed Inf (ModelEMA.update: v *= d; v += (1 - d) * msd); every
later val ran with that Inf. (2) engine/trainer.py save_model() checks deepcopy(ema).half() for NaN/Inf and skips the save: no epochN /
last / best from epoch 28 on. (3) utils/torch_utils.py strip_optimizer() stores the final last.pt / best.pt as fp16 (x["model"].half()).
Fix, training unchanged (AMP autocast + GradScaler for the train steps as before; data, epochs, imgsz, lr, aug, batch unchanged):
  validate()   : the val during training runs in fp32 (trainer.amp is False only while validate() runs), so the EMA is never cast to fp16;
                 best.pt = best fitness (ultralytics DetMetrics.fitness = mAP50-95) on the run's own val set, now computed in fp32.
  save_model() : = 8.4.37 save_model() with the EMA stored in fp32 (no .half()); the NaN/Inf check is done on the fp32 EMA.
  final_eval() : = 8.4.37 final_eval() with strip_optimizer_fp32() (no .half()) and the final val in fp32 (validator.args.half False).
Side effect (report it): the EMA is no longer rounded to fp16 once per epoch (in 8.4.37 every val did this); checkpoints are larger
(EMA approx. 87 MB -> 174 MB per file). Smoke test only: env DET_SMOKE_INFLATE_EPOCH=k sets that running_var[0] to 1e6 (> fp16 max)
in the EMA before the val of 0-based epoch k, to prove that val, save, strip and inference stay finite.
DDP: ultralytics writes a temp file that runs `from fp32_trainer import FP32DetectionTrainer` in torch.distributed.run, so this
folder must be on PYTHONPATH (tools/det/train_yolo_kicad_v3.py sets it)."""
import io, os
from copy import deepcopy
from datetime import datetime
import torch
import ultralytics
from ultralytics.engine import trainer as T
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import torch_utils as TU
assert ultralytics.__version__ == "8.4.37", f"fp32_trainer copies ultralytics 8.4.37 code, found {ultralytics.__version__}"
SMOKE_KEY = "model.28.cv3.3.0.bn.running_var"


def strip_optimizer_fp32(f="best.pt", s="", updates=None):
    """= ultralytics 8.4.37 utils/torch_utils.strip_optimizer() without x["model"].half(): the stripped model stays fp32."""
    try:
        x = TU.torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        TU.LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}
    metadata = {"date": datetime.now().isoformat(), "version": TU.__version__,
                "license": "AGPL-3.0 License (https://ultralytics.com/license)", "docs": "https://docs.ultralytics.com"}
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    x["model"].float()  # cros-det: stays FP32 (8.4.37: .half())
    for p in x["model"].parameters():
        p.requires_grad = False
    args = {**TU.DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in TU.DEFAULT_CFG_KEYS}  # strip non-default keys
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    TU.LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB (fp32)")
    return combined


class FP32DetectionTrainer(DetectionTrainer):
    def validate(self):
        k = os.environ.get("DET_SMOKE_INFLATE_EPOCH")
        if k is not None and self.ema and self.epoch == int(k):  # smoke test only
            with torch.no_grad():
                dict(self.ema.ema.named_buffers())[SMOKE_KEY][0] = 1e6
            T.LOGGER.warning(f"SMOKE: {SMOKE_KEY}[0] set to 1e6 in the EMA before the val of epoch {self.epoch}")
        amp, self.amp = self.amp, False  # validator: args.half = device != cpu and trainer.amp -> fp32 val, EMA never cast to fp16
        try:
            return super().validate()
        finally:
            self.amp = amp

    def save_model(self):
        """= ultralytics 8.4.37 BaseTrainer.save_model() with the EMA stored in fp32 (8.4.37: deepcopy(...).half())."""
        ema = deepcopy(T.unwrap_model(self.ema.ema)).float()
        if (
            not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor))
            and self.epoch > self.start_epoch  # at least save checkpoint for the first epoch
        ):
            T.LOGGER.warning(f"Skipping checkpoint save at epoch {self.epoch}: EMA contains NaN/Inf (fp32 check)")
            return False
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": T.convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": T.__version__,
                "git": {"root": str(T.GIT.root), "branch": T.GIT.branch, "commit": T.GIT.commit, "origin": T.GIT.origin},
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        return True

    def final_eval(self):
        """= ultralytics 8.4.37 BaseTrainer.final_eval() with strip_optimizer_fp32() and the final val in fp32."""
        model = self.best if self.best.exists() else None
        with T.torch_distributed_zero_first(T.LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if T.RANK in {-1, 0}:
                ckpt = strip_optimizer_fp32(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer_fp32(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            T.LOGGER.info(f"\nValidating {model} (fp32)...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.validator.args.half = False  # cros-det: fp32 final val
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")
