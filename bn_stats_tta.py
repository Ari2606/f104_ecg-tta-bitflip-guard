"""
bn_stats_tta.py  --  gradient-free streaming adaptation of BatchNorm statistics (AdaBN-style, N=1).

The last cheap attempt to find a regime where test-time adaptation helps the frozen, noise-aware model.
No backward pass, no anchor pass, no optimiser state: after predicting each beat, every selected
BatchNorm layer's running statistics are moved a little toward that beat's statistics.

  Population statistics are tracked as EMAs of the FIRST and SECOND moment per channel
      m1 <- (1-m) m1 + m * mean_beat(x),   m2 <- (1-m) m2 + m * mean_beat(x^2)
  and running_mean = m1, running_var = m2 - m1^2  (so between-beat variance is kept; averaging
  per-beat variances would lose it). Statistics are measured on the activations the deployed
  (running-stat) forward actually produces, i.e. the same pass that makes the prediction.
  Safety bound: running_var is clamped to [source_var / max_ratio, source_var * max_ratio].

  scope="all"     every BatchNorm1d in the backbone          (full AdaBN)
  scope="dw_only" only the dw_bn layers (same layers our method adapts the affine of)

Prediction-then-adapt order is respected: the reported logits use the statistics from BEFORE this
beat. The frozen source copy used for the D_JS telemetry column is NOT part of the deployed method
(it exists only so the evaluation can report drift) and is not counted in any edge-cost figure.
Cost on device: one forward pass + a mean / mean-of-squares per BN layer; BN statistics are
writable state, so they are exposed to the same radiation faults as the adapted affine.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from streaming_eval import StepResult, TTAMethod, step_aux


@dataclass
class BNStatsConfig:
    momentum: float = 0.005
    max_ratio: float = 4.0
    scope: str = "all"            # "all" | "dw_only"


class BNStatsTTA(TTAMethod):
    name = "bn_stats"

    def __init__(self, cfg: Optional[BNStatsConfig] = None):
        self.cfg = cfg or BNStatsConfig()

    def reset(self, model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._src = copy.deepcopy(model).eval()                 # telemetry only (see docstring)
        self.bns: Dict[str, nn.BatchNorm1d] = {
            n: m for n, m in model.backbone.named_modules()
            if isinstance(m, nn.BatchNorm1d) and (self.cfg.scope == "all" or n.endswith("dw_bn"))}
        assert self.bns, "no BatchNorm layers selected"
        self.m1 = {n: m.running_mean.clone() for n, m in self.bns.items()}
        self.m2 = {n: m.running_var.clone() + m.running_mean.clone() ** 2 for n, m in self.bns.items()}
        self.src_var = {n: m.running_var.clone() for n, m in self.bns.items()}
        self._cap: Dict[str, tuple] = {}
        self._hooks = []
        for n, m in self.bns.items():
            self._hooks.append(m.register_forward_pre_hook(self._make_hook(n)))

    def _make_hook(self, name: str):
        def hook(mod, inp):
            x = inp[0].detach()
            self._cap[name] = (x.mean(dim=(0, 2)), (x * x).mean(dim=(0, 2)))
        return hook

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        self._cap.clear()
        with torch.no_grad():
            logits = model(x, aux=aux).logits                  # statistics from BEFORE this beat
            anchor = self._src(x, aux=aux).logits              # telemetry only
            mom, ratio = self.cfg.momentum, self.cfg.max_ratio
            for n, m in self.bns.items():
                b1, b2 = self._cap[n]
                self.m1[n].mul_(1 - mom).add_(b1, alpha=mom)
                self.m2[n].mul_(1 - mom).add_(b2, alpha=mom)
                var = (self.m2[n] - self.m1[n] ** 2).clamp(min=1e-8)
                var = torch.minimum(torch.maximum(var, self.src_var[n] / ratio), self.src_var[n] * ratio)
                m.running_mean.copy_(self.m1[n])
                m.running_var.copy_(var)
        return StepResult(logits=logits, anchor_logits=anchor, adapted=True)


if __name__ == "__main__":
    from base_cnn import CNNConfig, BeatCNN
    from streaming_eval import js_divergence
    torch.manual_seed(0)
    model = BeatCNN(CNNConfig(in_channels=1)).eval()
    model.freeze_anchor()
    # fresh model's BN stats are mean 0 / var 1; give them realistic non-trivial values by a calibration pass
    xs = torch.randn(256, 1, 256)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = None                                    # cumulative average
    model.train()
    with torch.no_grad():
        model(xs)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = 0.1

    def run(scale: float, scope: str, n: int = 300):
        mm = copy.deepcopy(model)
        meth = BNStatsTTA(BNStatsConfig(momentum=0.05, scope=scope))
        meth.reset(mm)
        first = None
        for i in range(n):
            xb = scale * torch.randn(1, 1, 256)
            r = meth.step(mm, xb, {})
            if i == 0:
                first = r.logits.clone()
                ref = model(xb).logits
                assert torch.allclose(first, ref, atol=1e-5), "first prediction must use unadapted statistics"
        d = {k: float((m.running_var / meth.src_var[k]).mean()) for k, m in meth.bns.items()}
        js = float(js_divergence(r.logits, r.anchor_logits).item())
        return mm, d, js

    _, d0, js0 = run(1.0, "all")
    assert all(0.7 < v < 1.4 for v in d0.values()), d0           # same distribution -> stats stay put
    _, d3, js3 = run(3.0, "all")
    assert any(v > 1.5 for v in d3.values()), d3                 # amplitude shift -> variances move up
    assert all(0.25 - 1e-6 <= v <= 4 + 1e-6 for v in d3.values()), "clamp must hold"
    mm, dd, _ = run(3.0, "dw_only")
    assert all(k.endswith("dw_bn") for k in dd)
    assert all(torch.isfinite(m.running_var).all() for m in mm.modules() if isinstance(m, nn.BatchNorm1d))
    print(f"no shift: mean var ratio {sum(d0.values())/len(d0):.2f}; x3 shift: {sum(d3.values())/len(d3):.2f}")
    print("Self-test passed.")
