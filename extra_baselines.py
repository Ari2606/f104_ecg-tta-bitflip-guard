"""
extra_baselines.py  --  this project: additional published TTA baselines and
"simple alternative to the elastic trust region" controls.

Every method implements `streaming_eval.TTAMethod`, runs through the SAME N=1,
predict-then-adapt `StreamingEvaluator` loop, adapts ONLY the sparse dw_bn affine
parameters (`model.configure_for_tta()`, the same edge-SRAM budget this project
gets), and reports `anchor_logits` from `model.forward_anchor()` so the drift column
is comparable across all methods.

Published methods (re-implemented from the papers' algorithm boxes and the authors'
public code; deviations are listed per class and are the honest scope of the claim
"we compared against X" -- they are N=1 / sparse-scope / 5-class-ECG adaptations):

  EATATTA   -- EATA, Niu et al., "Efficient Test-Time Model Adaptation without
               Forgetting", ICML 2022. Reliable-sample entropy filter, entropy-weighted
               loss, redundancy filter on a moving-average of output probabilities, and
               an optional Fisher (anti-forgetting) regulariser.
  SARTTA    -- SAR, Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild
               World", ICLR 2023. Entropy filter + Sharpness-Aware Minimisation + a
               model-recovery (reset) rule driven by a moving average of the loss.
  CoTTATTA  -- CoTTA, Wang et al., "Continual Test-Time Domain Adaptation", CVPR 2022.
               EMA teacher, augmentation-averaged teacher targets when the source
               model is unsure, and stochastic restoration of parameters to source.

Controls that isolate what the elastic trust region contributes (no citation: these
are the obvious simple alternatives a reviewer will ask about):

  TentRegularizedTTA -- TENT + L2 pull to the source parameters (L2-SP style, fixed
               strength) and/or hard periodic reset to source every N beats.
               "Is a constant pull / a periodic reset just as good as the elastic,
               divergence-driven pull?"

Run `python extra_baselines.py` for a synthetic self-test (no PhysioNet needed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from dataset import NUM_CLASSES
from streaming_eval import StepResult, TTAMethod, predictive_entropy, step_aux


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _sparse_params(model: nn.Module) -> List[nn.Parameter]:
    return model.configure_for_tta()          # dw_bn affine only, eval mode, BN stats frozen


def _source_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.tta_named_parameters()}


@torch.no_grad()
def _restore_to_source(model: nn.Module, src: Dict[str, torch.Tensor]) -> None:
    for n, p in model.tta_named_parameters():
        p.copy_(src[n].to(p.device))


def _teacher_logits(model: nn.Module, affine: Dict[str, torch.Tensor], x: torch.Tensor,
                    aux: Optional[torch.Tensor]) -> torch.Tensor:
    """Backbone forward with dw_bn affine params replaced by `affine` (functional_call,
    live params untouched), through the adaptable head -- same pattern as RoTTA/anchor."""
    params = dict(model.backbone.named_parameters())
    for full, val in affine.items():
        key = full[len("backbone."):] if full.startswith("backbone.") else full
        if key in params:
            params[key] = val.to(x.device)
    feat = functional_call(model.backbone, (params, dict(model.backbone.named_buffers())), (x,))
    if model.cfg.aux_feat_dim > 0:
        feat = torch.cat([feat, aux], dim=-1)
    return model.adaptable_head(feat)


# --------------------------------------------------------------------------- #
# EATA
# --------------------------------------------------------------------------- #
@dataclass
class EATAConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    e_margin_frac: float = 0.4     # E0 = e_margin_frac * ln(K)   (paper: 0.4 ln C)
    d_margin: float = 0.4          # redundancy threshold epsilon (see class docstring)
    ma_alpha: float = 0.9          # moving-average of output probabilities
    fisher_alpha: float = 0.0      # anti-forgetting weight; 0 = off (EATA w/o Fisher)
    fisher_n: int = 500            # source beats used to estimate the Fisher


class EATATTA(TTAMethod):
    """
    EATA (Niu et al., ICML 2022), N=1, sparse scope. Per beat:
      1. reliable filter: adapt only if entropy E(x) < E0 = 0.4 ln K.
      2. redundancy filter: adapt only if |cos(moving-avg probs, p(x))| < d_margin.
      3. entropy-weighted loss S(x)*E(x), S = 1/exp(E - E0); optional Fisher penalty
         fisher_alpha * sum F_i (theta_i - theta_src_i)^2 (Fisher from pseudo-labelled
         source beats via `set_fisher_data`, as in the authors' code).
    Deviations: (a) sparse dw_bn scope; (b) the official redundancy threshold 0.05 was
    designed for K=1000 where independent probability vectors are nearly orthogonal;
    with K=5 it filters almost every beat, so d_margin is exposed and tuned on DS1-VAL
    (default 0.4); (c) SGD+momentum for consistency with the other baselines.
    """

    def __init__(self, cfg: Optional[EATAConfig] = None):
        self.cfg = cfg or EATAConfig()
        tag = f"_fi{self.cfg.fisher_alpha:g}" if self.cfg.fisher_alpha > 0 else ""
        self.name = f"eata_sparse{tag}"
        self.opt = None
        self._ma: Optional[torch.Tensor] = None
        self._src: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._fisher_data = None
        self.e0 = self.cfg.e_margin_frac * math.log(NUM_CLASSES)

    def set_fisher_data(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> None:
        """Unlabelled SOURCE beats [n,C,L] (+ aux [n,A]) for the Fisher estimate."""
        self._fisher_data = (x[: self.cfg.fisher_n], None if aux is None else aux[: self.cfg.fisher_n])

    def reset(self, model: nn.Module) -> None:
        params = _sparse_params(model)
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self._ma = None
        self._src = _source_snapshot(model)
        self._fisher = {}
        if self.cfg.fisher_alpha > 0:
            if self._fisher_data is None:
                raise RuntimeError("fisher_alpha>0 needs set_fisher_data(...) before the stream")
            self._fisher = self._estimate_fisher(model)
        self.n_reliable = self.n_kept = 0

    def _estimate_fisher(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        xs, auxs = self._fisher_data
        fisher = {n: torch.zeros_like(p) for n, p in model.tta_named_parameters()}
        for i in range(len(xs)):
            xi = xs[i:i + 1]
            ai = None if auxs is None else auxs[i:i + 1]
            out = model(xi, aux=ai).logits
            loss = F.cross_entropy(out, out.argmax(-1).detach())      # pseudo-label, as in EATA code
            model.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.tta_named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        model.zero_grad(set_to_none=True)
        return {n: f / max(len(xs), 1) for n, f in fisher.items()}

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        out = model(x, aux=aux)                                       # live, WITH grad
        logits = out.logits
        pred_logits = logits.detach()
        with torch.no_grad():
            anchor = model.forward_anchor(x, aux=aux)
        ent = predictive_entropy(logits)                             # [1]
        probs = F.softmax(logits, dim=-1)

        reliable = bool((ent < self.e0).item())
        keep = reliable
        cos = float("nan")
        if reliable:
            self.n_reliable += 1
            if self._ma is not None:
                cos = float(F.cosine_similarity(self._ma.unsqueeze(0), probs.detach(), dim=1).item())
                keep = abs(cos) < self.cfg.d_margin
            if keep:                                                  # EATA updates the moving avg
                new = probs.detach().mean(0)                          # with probs of KEPT samples
                self._ma = new if self._ma is None else 0.9 * self._ma + (1 - 0.9) * new
        adapted = False
        loss_val = None
        if keep:
            self.n_kept += 1
            coeff = torch.exp(-(ent.detach() - self.e0))             # S(x) = 1/exp(E - E0)
            loss = (ent * coeff).mean()
            if self.cfg.fisher_alpha > 0:
                ewc = sum((self._fisher[n] * (p - self._src[n].to(p.device)) ** 2).sum()
                          for n, p in model.tta_named_parameters())
                loss = loss + self.cfg.fisher_alpha * ewc
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            adapted = True
            loss_val = float(loss.detach())
        return StepResult(logits=pred_logits, anchor_logits=anchor, adapted=adapted,
                          rejected=not adapted,
                          extra={"adapt_loss": loss_val if loss_val is not None else float("nan"),
                                 "entropy": float(ent.item()), "cos_ma": cos})


# --------------------------------------------------------------------------- #
# SAR
# --------------------------------------------------------------------------- #
@dataclass
class SARConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    rho: float = 0.05              # SAM neighbourhood radius (paper default)
    e_margin_frac: float = 0.4     # E0 = 0.4 ln K
    reset_ema: float = 0.2         # recover (reset) when EMA of second-pass loss < this
    ema_decay: float = 0.9


class SARTTA(TTAMethod):
    """
    SAR (Niu et al., ICLR 2023), N=1, sparse scope. Per beat:
      1. predict (reported logits, pre-update).
      2. entropy filter: skip beats with E >= E0 = 0.4 ln K (noisy, large gradients).
      3. SAM step: gradient of the filtered entropy, ascend to theta + rho g/||g||,
         recompute the (re-filtered) entropy there, restore theta, descend with that
         second gradient.
      4. model recovery: EMA of the second-pass loss; if it falls below `reset_ema`
         (collapse into trivial confident predictions), reset the adapted parameters to
         source and clear optimiser state.
    Deviation: sparse dw_bn scope instead of norm layers of a ResNet/ViT; thresholds are
    the paper's, scaled to K=5 through E0 = 0.4 ln K (reset_ema kept at the paper's 0.2).
    """

    def __init__(self, cfg: Optional[SARConfig] = None):
        self.cfg = cfg or SARConfig()
        self.name = "sar_sparse"
        self.opt = None
        self.params: List[nn.Parameter] = []
        self._src: Dict[str, torch.Tensor] = {}
        self._ema: Optional[float] = None
        self.e0 = self.cfg.e_margin_frac * math.log(NUM_CLASSES)

    def reset(self, model: nn.Module) -> None:
        self.params = _sparse_params(model)
        self.opt = torch.optim.SGD(self.params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self._src = _source_snapshot(model)
        self._ema = None
        self.n_resets = 0

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        logits = model(x, aux=aux).logits
        pred_logits = logits.detach()
        with torch.no_grad():
            anchor = model.forward_anchor(x, aux=aux)
        ent = predictive_entropy(logits)
        reset_flag = False
        adapted = False
        loss_val = float("nan")
        if bool((ent < self.e0).item()):
            self.opt.zero_grad(set_to_none=True)
            ent.mean().backward()
            grads = [p.grad for p in self.params if p.grad is not None]
            gnorm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) if grads else None
            eps: List[Optional[torch.Tensor]] = []
            if gnorm is not None and float(gnorm) > 0:
                scale = self.cfg.rho / (gnorm + 1e-12)
                with torch.no_grad():
                    for p in self.params:                              # first step: ascend
                        e = None if p.grad is None else p.grad * scale
                        eps.append(e)
                        if e is not None:
                            p.add_(e)
                self.opt.zero_grad(set_to_none=True)
                ent2 = predictive_entropy(model(x, aux=aux).logits)
                m2 = ent2 < self.e0
                if bool(m2.any().item()):
                    loss2 = ent2[m2].mean()
                    self._ema = float(loss2.item()) if self._ema is None else \
                        self.cfg.ema_decay * self._ema + (1 - self.cfg.ema_decay) * float(loss2.item())
                    loss2.backward()
                    loss_val = float(loss2.detach())
                    with torch.no_grad():                              # restore, then descend
                        for p, e in zip(self.params, eps):
                            if e is not None:
                                p.sub_(e)
                    self.opt.step()
                    adapted = True
                else:
                    with torch.no_grad():
                        for p, e in zip(self.params, eps):
                            if e is not None:
                                p.sub_(e)
            self.opt.zero_grad(set_to_none=True)
            if self._ema is not None and self._ema < self.cfg.reset_ema:
                _restore_to_source(model, self._src)
                self.opt = torch.optim.SGD(self.params, lr=self.cfg.lr, momentum=self.cfg.momentum)
                self._ema = None
                reset_flag = True
                self.n_resets += 1
        return StepResult(logits=pred_logits, anchor_logits=anchor, adapted=adapted,
                          rejected=not adapted, reset_triggered=reset_flag,
                          extra={"adapt_loss": loss_val, "entropy": float(ent.item())})


# --------------------------------------------------------------------------- #
# CoTTA
# --------------------------------------------------------------------------- #
@dataclass
class CoTTAConfig:
    lr: float = 1e-3
    ema_alpha: float = 0.999       # teacher EMA (paper 0.999)
    rst: float = 0.01              # stochastic-restore probability per parameter element
    ap: float = 0.92               # confidence threshold of the SOURCE model for using augs
    n_aug: int = 8                 # augmentations averaged (paper: 32; reduced for N=1 edge cost)
    aug_gain_std: float = 0.1
    aug_noise_std: float = 0.03
    aug_max_shift: int = 6


class CoTTATTA(TTAMethod):
    """
    CoTTA (Wang et al., CVPR 2022), N=1, sparse scope. Per beat:
      * teacher (EMA of the dw_bn affine) gives the reported prediction (as in the paper,
        the teacher's output is the output) -- computed BEFORE any update this step.
      * if the frozen source model's confidence on the beat is below `ap`, the teacher
        target is the mean over `n_aug` augmented views (gain, noise, small time shift);
        otherwise the plain teacher output.
      * student loss: symmetric soft cross-entropy between student output and the
        (augmentation-averaged) teacher target; Adam step on the student.
      * teacher EMA update; then stochastic restore: each parameter element is reset to
        its source value with probability `rst`.
    Deviations: sparse dw_bn scope; ECG-specific augmentations replace colour/affine
    image augmentations; n_aug=8 instead of 32; Adam as in the authors' CIFAR config.
    """

    def __init__(self, cfg: Optional[CoTTAConfig] = None):
        self.cfg = cfg or CoTTAConfig()
        self.name = "cotta_sparse"
        self.opt = None
        self._teacher: Dict[str, torch.Tensor] = {}
        self._src: Dict[str, torch.Tensor] = {}
        self.gen = torch.Generator().manual_seed(0)

    def reset(self, model: nn.Module) -> None:
        params = _sparse_params(model)
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr)
        self._src = _source_snapshot(model)
        self._teacher = {n: v.clone() for n, v in self._src.items()}
        self.gen = torch.Generator().manual_seed(0)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        g = 1.0 + c.aug_gain_std * torch.randn(1, generator=self.gen).item()
        shift = int(torch.randint(-c.aug_max_shift, c.aug_max_shift + 1, (1,), generator=self.gen).item())
        noise = c.aug_noise_std * torch.randn(x.shape, generator=self.gen).to(x.device)
        return torch.roll(x * g, shifts=shift, dims=-1) + noise

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        c = self.cfg
        with torch.no_grad():
            teacher_out = _teacher_logits(model, self._teacher, x, aux)      # reported (pre-update)
            anchor = model.forward_anchor(x, aux=aux)
            src_conf = float(F.softmax(anchor, -1).max().item())
            if src_conf < c.ap:
                outs = [_teacher_logits(model, self._teacher, self._augment(x), aux)
                        for _ in range(c.n_aug)]
                target = torch.stack(outs).mean(0)
            else:
                target = teacher_out

        student = model(x, aux=aux).logits                                    # WITH grad
        t_p, s_lp = F.softmax(target, -1), F.log_softmax(student, -1)
        s_p, t_lp = F.softmax(student, -1), F.log_softmax(target, -1)
        loss = (-0.5 * (t_p * s_lp).sum(-1) - 0.5 * (s_p * t_lp).sum(-1)).mean()   # symmetric CE
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for n, p in model.tta_named_parameters():
                self._teacher[n].mul_(c.ema_alpha).add_(p.detach(), alpha=1 - c.ema_alpha)
                mask = (torch.rand(p.shape, generator=self.gen) < c.rst).to(p.device)
                p.copy_(torch.where(mask, self._src[n].to(p.device), p))     # stochastic restore
        return StepResult(logits=teacher_out, anchor_logits=anchor, adapted=True, rejected=False,
                          extra={"adapt_loss": float(loss.detach()), "src_conf": src_conf,
                                 "used_aug": float(src_conf < c.ap)})


# --------------------------------------------------------------------------- #
# TENT + simple restoring controls
# --------------------------------------------------------------------------- #
@dataclass
class TentRegConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    l2_lambda: float = 0.0         # L2-SP pull: lambda * sum (theta - theta_src)^2 ; 0 = off
    reset_every: int = 0           # hard reset to source every N beats ; 0 = never


class TentRegularizedTTA(TTAMethod):
    """TENT (unmasked entropy, sparse scope) plus a *fixed-strength* restoring mechanism:
    an L2 pull to the source parameters (`l2_lambda`; keep lr*2*lambda < 1 or the SGD
    step on the quadratic term diverges) and/or periodic hard reset (`reset_every`). These are the obvious simpler alternatives to the elastic,
    divergence-driven trust region; if they matched it, the elastic part would not be a
    contribution. Name encodes the setting, e.g. `tent_l2_1e+01`, `tent_reset500`."""

    def __init__(self, cfg: Optional[TentRegConfig] = None):
        self.cfg = cfg or TentRegConfig()
        parts = ["tent"]
        if self.cfg.l2_lambda > 0:
            parts.append(f"l2_{self.cfg.l2_lambda:.0e}".replace("e+0", "e+").replace("e-0", "e-"))
        if self.cfg.reset_every > 0:
            parts.append(f"reset{self.cfg.reset_every}")
        self.name = "_".join(parts)
        self.opt = None
        self._src: Dict[str, torch.Tensor] = {}
        self._t = 0

    def reset(self, model: nn.Module) -> None:
        params = _sparse_params(model)
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self._src = _source_snapshot(model)
        self._t = 0

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        self._t += 1
        out = model(x, aux=aux)
        pred_logits = out.logits.detach()
        loss = predictive_entropy(out.logits).mean()
        if self.cfg.l2_lambda > 0:
            loss = loss + self.cfg.l2_lambda * sum(
                ((p - self._src[n].to(p.device)) ** 2).sum() for n, p in model.tta_named_parameters())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        reset = False
        if self.cfg.reset_every > 0 and self._t % self.cfg.reset_every == 0:
            _restore_to_source(model, self._src)
            self.opt = torch.optim.SGD([p for _, p in model.tta_named_parameters()],
                                       lr=self.cfg.lr, momentum=self.cfg.momentum)
            reset = True
        with torch.no_grad():
            anchor = model.forward_anchor(x, aux=aux)
        return StepResult(logits=pred_logits, anchor_logits=anchor, adapted=True, rejected=False,
                          reset_triggered=reset, extra={"adapt_loss": float(loss.detach())})


# --------------------------------------------------------------------------- #
# Self-test (synthetic; plumbing + invariants, not a scientific result)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import copy
    from base_cnn import CNNConfig, BeatCNN
    from baselines_tta import _SyntheticStreamDataset
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
    from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream

    print("[extra_baselines] self-test on synthetic data")
    torch.manual_seed(0)
    model = BeatCNN(CNNConfig(in_channels=1))
    ds = _SyntheticStreamDataset(n_records=2, beats_per_record=200, c=1, l=256, seed=0)
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()), lr=1e-3)
    model.train()
    for _ in range(10):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(model(ds.x[idx]).logits, ds.y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); model.freeze_anchor()

    def run(method, n=300, lr_check=False):
        m = copy.deepcopy(model)
        bn_before = {k: v.clone() for k, v in m.state_dict().items() if "running" in k}
        sim = SpaceShiftSimulator(SpaceShiftConfig(seed=0))
        ramp = linear_ramp(start_idx=30, n_ramp=60)
        stream = make_beat_stream(ds, n_beats=n, sim=sim, severity=lambda i: min(0.4, ramp(i)))
        ev = StreamingEvaluator(method, StreamingConfig(device="cpu", rolling_window=50, log_every=0))
        res = ev.run(m, stream)
        for k, v in bn_before.items():
            assert torch.equal(m.state_dict()[k], v), f"{method.name}: BN running stat {k} changed"
        assert not m.training, f"{method.name}: model left in train mode"
        drift0 = {n: (p.detach() - model.state_dict()[n]).abs().max().item()
                  for n, p in m.tta_named_parameters()}
        return res, max(drift0.values())

    ds_fisher = (ds.x[:60], None)
    methods = [
        EATATTA(EATAConfig()),
        EATATTA(EATAConfig(fisher_alpha=100.0, fisher_n=40)),
        SARTTA(SARConfig(lr=1e-2)),
        CoTTATTA(CoTTAConfig(n_aug=2)),
        TentRegularizedTTA(TentRegConfig(l2_lambda=10.0)),
        TentRegularizedTTA(TentRegConfig(reset_every=50)),
    ]
    for meth in methods:
        if isinstance(meth, EATATTA) and meth.cfg.fisher_alpha > 0:
            meth.set_fisher_data(*ds_fisher)
        res, maxdrift = run(meth)
        s = res.summary
        assert 0.0 <= s["accuracy"] <= 1.0 and math.isfinite(s["mean_js_div"])
        print(f"  {meth.name:<22} acc={s['accuracy']:.3f} D_JS={s['mean_js_div']:.4f} "
              f"adapt_rate={s['adaptation_rate']:.2f} max|d theta|={maxdrift:.4f}")

    # --- targeted invariants ------------------------------------------------ #
    # EATA: filters actually filter (not all beats adapted) and counters populated
    e = EATATTA(EATAConfig()); run(e)
    assert 0 < e.n_kept <= e.n_reliable, (e.n_kept, e.n_reliable)
    print(f"  EATA reliable={e.n_reliable} kept(non-redundant)={e.n_kept} of 300")
    # EATA Fisher: with huge fisher_alpha, parameters stay closer to source than w/o Fisher
    ef = EATATTA(EATAConfig(fisher_alpha=1e6, fisher_n=40)); ef.set_fisher_data(*ds_fisher)
    _, d_f = run(ef); _, d_0 = run(EATATTA(EATAConfig()))
    assert d_f <= d_0 + 1e-9, (d_f, d_0)
    # SAR: weights restored after the SAM perturbation (no permanent +rho*g drift on skipped beats)
    sr = SARTTA(SARConfig(lr=0.0)); _, d_sar0 = run(sr)
    assert d_sar0 < 1e-6, f"SAR with lr=0 must leave theta at source, got {d_sar0}"
    # CoTTA stochastic restore with rst=1 forces theta == source
    ct = CoTTATTA(CoTTAConfig(rst=1.0, n_aug=1)); _, d_ct = run(ct, n=60)
    assert d_ct < 1e-6, f"CoTTA rst=1 must pin theta to source, got {d_ct}"
    # TENT+reset: right after a reset the parameters equal source
    tr = TentRegularizedTTA(TentRegConfig(reset_every=50)); _, d_tr = run(tr, n=300)
    assert d_tr < 1e-6, f"reset at beat 300 (multiple of 50) must leave theta=source, got {d_tr}"
    # TENT+L2 (huge lambda) barely moves from source
    tl = TentRegularizedTTA(TentRegConfig(l2_lambda=100.0)); _, d_tl = run(tl)
    _, d_t0 = run(TentRegularizedTTA(TentRegConfig()))
    assert d_tl < d_t0, (d_tl, d_t0)
    print("  invariants OK: BN stats frozen, eval mode kept, EATA filters, Fisher/L2 restrain drift, "
          "SAR lr=0 / CoTTA rst=1 / periodic reset pin theta to source")
    print("Self-test passed.")
