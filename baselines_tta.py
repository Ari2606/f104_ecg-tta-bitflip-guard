"""
baselines_tta.py  --  this project, Stage 4 (Competing Baselines)

Three reference test-time-adaptation methods, all implementing `streaming_eval.TTAMethod`
so they run through the exact same `StreamingEvaluator` loop (N=1, predict-then-adapt,
no label ever touches `.step()`) that Stage 5's this project controller will use.

1. SourceOnlyTTA        -- zero adaptation, frozen weights. The "do nothing" floor.
2. TentTTA               -- standard entropy minimization (Wang et al., 2021), restricted
                            by default to the SAME sparse parameter set this project
                            is allowed (`model.tta_parameters()`, i.e. dw_bn affine only,
                            <5% of weights / sub-512KB SRAM) so the Stage-5 comparison
                            isolates the effect of masking + elastic reset, not a bigger
                            parameter budget. `scope="full_affine"` reproduces the
                            literature-standard TENT (all BatchNorm affine, including
                            pointwise BN) as a supplementary ablation -- flagged because
                            it exceeds this paper's edge-SRAM constraint.
                            TENT is deliberately given NO safety mechanism: it minimizes
                            entropy on every beat unconditionally, including corrupted
                            ones. That naivety is the point -- it is the failure mode
                            Stage 5 exists to fix.
3. SQIGatedTent         -- a scalar, whole-window temporal quality index (in-band vs.
                            out-of-band spectral energy, cutoff = dataset.py's own 40 Hz
                            bandpass edge) that completely gates OUT low-quality beats:
                            below threshold, the beat is still predicted but the entropy
                            update is skipped entirely. This is this paper's own minimal
                            scalar-SQI-gated TENT reference baseline. It is NOT a
                            reproduction of BeatRhythm-TTA (Jiang et al., arXiv:2608.23347),
                            which uses SQI-gated self-training plus beat-rhythm consistency
                            losses on record-level multi-label ECG; do not cite or name it
                            as that method. It
                            contrasts directly with Stage 5's soft, sub-window 2D
                            time-frequency mask: here a whole beat is kept or entirely
                            dropped, never partially trusted.
4. RoTTATTA              -- a REAL, NAMED, PUBLISHED method: RoTTA (Yuan, Xie, Li,
                            "Robust Test-Time Adaptation in Dynamic Scenarios",
                            CVPR 2023). Reproduces its two defining
                            mechanisms (CSTU category-balanced replay memory + EMA
                            teacher-student self-training), specifically designed for
                            temporally-correlated/non-i.i.d. streams -- the exact protocol
                            this project's N=1 chronological DS2 stream is. See
                            `RoTTAConfig`'s docstring below for the full mapping and for
                            why RoTTA's third mechanism (Robust BN) is a documented no-op
                            in this architecture rather than silently skipped.

All four methods report `anchor_logits` via `model.forward_anchor(x)` (backbone-state
anchor: current dw_bn affine vs. the frozen source snapshot, both through the shared
head -- see `base_cnn.py`'s module docstring), so their D_JS-vs-source columns in the
Stage-4/5 comparison table are genuine drift signals, not the identically-zero
same-features head comparison `model(x, use_anchor=True)` would give.

Run `python baselines_tta.py` with no arguments for a self-test sweep on a synthetic
space-shifted stream (no PhysioNet download needed); it prints the Stage-4 comparison
table produced by `streaming_eval.compare_methods`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from dataset import NUM_CLASSES
from streaming_eval import (StepResult, StreamingConfig, StreamingEvaluator, TTAMethod,
                            compare_methods, make_beat_stream, predictive_entropy, step_aux)


# --------------------------------------------------------------------------- #
# 1. Source-Only
# --------------------------------------------------------------------------- #
class SourceOnlyTTA(TTAMethod):
    """Zero adaptation: every beat is scored with the frozen Stage-1 weights."""
    name = "source_only"

    def reset(self, model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        with torch.no_grad():
            out = model(x, aux=aux)
            anchor_logits = model.forward_anchor(x, aux=aux)
        return StepResult(logits=out.logits, anchor_logits=anchor_logits,
                          adapted=False, rejected=False)


# --------------------------------------------------------------------------- #
# 2. Standard TENT (affine entropy minimization, no safety mechanism)
# --------------------------------------------------------------------------- #
@dataclass
class TentConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    scope: str = "sparse"          # "sparse" (dw_bn only, edge-budget-fair) | "full_affine" (literature TENT)


class TentTTA(TTAMethod):
    """
    Entropy minimization on EVERY beat, unconditionally. No SQI gate, no divergence
    check, no reset -- this is intentionally the naive baseline the paper's safety
    mechanisms are compared against.
    """

    def __init__(self, cfg: Optional[TentConfig] = None):
        self.cfg = cfg or TentConfig()
        self.name = f"tent_{self.cfg.scope}"
        self.opt: Optional[torch.optim.Optimizer] = None

    def reset(self, model: nn.Module) -> None:
        if self.cfg.scope == "sparse":
            params = model.configure_for_tta()                       # dw_bn affine only
        elif self.cfg.scope == "full_affine":
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            params = []
            for m in model.modules():                                 # only hits backbone BNs;
                if isinstance(m, nn.BatchNorm1d):                      # heads have no BatchNorm
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    params += [m.weight, m.bias]
        else:
            raise ValueError(f"unknown TENT scope '{self.cfg.scope}'")
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        out = model(x, aux=aux)                                        # adaptable head, WITH grad
        loss = predictive_entropy(out.logits).mean()                   # unmasked entropy, full window
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            anchor_logits = model.forward_anchor(x, aux=aux)            # source-dw_bn backbone state
        return StepResult(logits=out.logits.detach(), anchor_logits=anchor_logits,
                          adapted=True, rejected=False, extra={"adapt_loss": float(loss.detach())})


# --------------------------------------------------------------------------- #
# 3. SQI-gated TENT (scalar SQI gate)
# --------------------------------------------------------------------------- #
@dataclass
class SQIGatedTentConfig:
    fs: float = 360.0
    cutoff_hz: float = 40.0        # matches dataset.py's own diagnostic-band bandpass edge
    lr: float = 1e-3
    momentum: float = 0.9
    calib_beats: int = 200         # beats used to auto-calibrate the gate threshold online
    calib_quantile: float = 0.2    # gate = worst 20% of SQI seen during calibration
    threshold: Optional[float] = None   # override; None = online auto-calibration


class SQIGatedTent(TTAMethod):
    """
    scalar_sqi(x) = (in-band spectral power) / (total spectral power), in-band defined
    by `cutoff_hz`. Radiation bursts, EMG/motion artifact and ADC noise are broadband
    relative to the QRS-T rhythm, so they lower this ratio. Below threshold: predict
    with current weights but DROP the update entirely (no partial credit, unlike
    Stage 5's soft 2D mask). Threshold is auto-calibrated from the first
    `calib_beats` beats of the stream (their `calib_quantile` quantile) unless a
    fixed `threshold` is given -- no update is gated during calibration itself.
    """

    def __init__(self, cfg: Optional[SQIGatedTentConfig] = None):
        self.cfg = cfg or SQIGatedTentConfig()
        self.name = "sqi_gated_tent"
        self.opt: Optional[torch.optim.Optimizer] = None
        self._calib: list = []
        self._threshold: Optional[float] = None

    def reset(self, model: nn.Module) -> None:
        params = model.configure_for_tta()                            # dw_bn affine only
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self._calib = []
        self._threshold = self.cfg.threshold

    @staticmethod
    def scalar_sqi(x: torch.Tensor, fs: float, cutoff_hz: float) -> float:
        """x: [1, C, L] -> python float in [0, 1]. Higher = cleaner."""
        L = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)                                  # [1, C, L/2+1]
        freqs = torch.fft.rfftfreq(L, d=1.0 / fs).to(x.device)         # [L/2+1]
        power = (X.abs() ** 2).sum(dim=1)                              # sum leads -> [1, L/2+1]
        inband = power[:, freqs <= cutoff_hz].sum(-1)                  # [1]
        total = power.sum(-1).clamp_min(1e-12)                         # [1]
        return float((inband / total).clamp(0.0, 1.0).item())

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        sqi = self.scalar_sqi(x, self.cfg.fs, self.cfg.cutoff_hz)

        if self._threshold is None:
            self._calib.append(sqi)
            gate_open = True                                          # never gate during calibration
            if len(self._calib) >= self.cfg.calib_beats:
                self._threshold = float(np.quantile(self._calib, self.cfg.calib_quantile))
        else:
            gate_open = sqi >= self._threshold

        if gate_open:
            out = model(x, aux=aux)
            loss = predictive_entropy(out.logits).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                anchor_logits = model.forward_anchor(x, aux=aux)
            return StepResult(logits=out.logits.detach(), anchor_logits=anchor_logits,
                              adapted=True, rejected=False, sqi=sqi,
                              extra={"adapt_loss": float(loss.detach())})
        with torch.no_grad():
            out = model(x, aux=aux)
            anchor_logits = model.forward_anchor(x, aux=aux)
        return StepResult(logits=out.logits, anchor_logits=anchor_logits,
                          adapted=False, rejected=True, sqi=sqi)


# --------------------------------------------------------------------------- #
# 4. RoTTA (Robust Test-Time Adaptation in Dynamic Scenarios, Yuan et al., CVPR 2023)
# --------------------------------------------------------------------------- #
@dataclass
class RoTTAConfig:
    """
    Reproduces RoTTA's two mechanisms that specifically target TEMPORALLY-
    CORRELATED / non-i.i.d. streaming TTA -- Longhui Yuan, Binhui Xie, Shuang Li,
    "Robust Test-Time Adaptation in Dynamic Scenarios", CVPR 2023:

    1. CSTU memory (Category-balanced Sampling with Timeliness and Uncertainty,
       paper Sec 3.2): a fixed-capacity-per-class replay buffer. Every incoming
       beat is pseudo-labeled by an EMA teacher and offered to its predicted
       class's bucket; when a bucket is full, the incoming item replaces the
       stored item with the WORST (highest) heuristic score -- lambda_u *
       uncertainty + lambda_t * normalized_age -- if its own score is lower,
       otherwise it is dropped. Student updates are computed on a batch
       resampled CLASS-BALANCED from this memory every `update_every` beats,
       never on the raw incoming stream directly. This is RoTTA's answer to
       exactly the failure mode this project's own TentTTA baseline
       demonstrates with hard confusion-matrix evidence elsewhere in this
       project (unconstrained entropy minimization on a ~95%-class-N,
       autocorrelated stream collapsing to majority-class prediction).
    2. EMA teacher-student self-training: a teacher snapshot of the dw_bn
       affine parameters is updated as teacher <- ema_alpha*teacher +
       (1-ema_alpha)*student after every update step. The teacher (not the
       noisier single-beat student) supplies pseudo-labels for CSTU
       admission, and the replay loss is reweighted by each item's age via
       `timeliness_tau` so stale memory entries matter less.

    RoTTA's original paper also introduces "Robust BN" (batch stats computed
    from a running global estimate rather than the current, possibly
    degenerate, batch). This project's architecture ALREADY keeps BatchNorm
    running stats frozen during TTA for every method here (base_cnn.py's
    `configure_for_tta()`) and evaluates at batch size 1 -- exactly the
    condition Robust BN exists to fix. Robust BN is therefore a documented
    no-op in this codebase, not silently omitted.

    Scope: restricted to dw_bn affine ("sparse") by default, matching the
    SAME edge-SRAM fairness constraint already applied to TentTTA and
    SQIGatedTent in this file -- not RoTTA's original full-BN-affine
    scope. `scope="full_affine"` reproduces the literature-standard version
    as a supplementary ablation, flagged the same way TentConfig flags its
    own full-affine variant.
    """
    lr: float = 1e-3
    momentum: float = 0.9
    memory_per_class: int = 16      # CSTU per-category bucket capacity
    batch_size: int = 16            # class-balanced replay batch size
    update_every: int = 1           # take a student step every N beats (1 = every beat)
    ema_alpha: float = 0.999        # teacher EMA momentum
    lambda_t: float = 1.0           # timeliness weight in the CSTU admission score
    lambda_u: float = 1.0           # uncertainty weight in the CSTU admission score
    timeliness_tau: float = 1000.0  # beats; replay-loss age-decay scale (larger = slower decay)
    scope: str = "sparse"           # "sparse" (dw_bn affine, edge-budget-fair) | "full_affine"


class _CSTUMemory:
    """Category-balanced Sampling with Timeliness and Uncertainty (RoTTA, Sec 3.2).
    One fixed-capacity bucket per class; admission/eviction driven by a heuristic
    score = lambda_u*uncertainty + lambda_t*normalized_age (higher = more expendable).
    `sample_balanced` draws (as close to) an equal number of beats per NON-EMPTY
    class, so the student's replay batch is class-balanced even though the raw
    incoming stream is dominated by class N and temporally autocorrelated."""

    def __init__(self, num_classes: int, capacity_per_class: int, lambda_t: float, lambda_u: float):
        self.buckets: list = [[] for _ in range(num_classes)]
        self.capacity = capacity_per_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

    def _score(self, item: dict, now: int) -> float:
        age = max(0, now - item["ins_step"])
        norm_age = age / max(1, now)
        return self.lambda_u * item["uncertainty"] + self.lambda_t * norm_age

    def add(self, x: torch.Tensor, aux: Optional[torch.Tensor], label: int,
           uncertainty: float, now: int) -> bool:
        bucket = self.buckets[label]
        item = {"x": x, "aux": aux, "y": label, "uncertainty": uncertainty, "ins_step": now}
        if len(bucket) < self.capacity:
            bucket.append(item)
            return True
        scores = [self._score(b, now) for b in bucket]
        worst = int(np.argmax(scores))
        if self._score(item, now) <= scores[worst]:
            bucket[worst] = item
            return True
        return False

    def sample_balanced(self, batch_size: int) -> Optional[list]:
        nonempty = [b for b in self.buckets if b]
        if not nonempty:
            return None
        per_class = max(1, batch_size // len(nonempty))
        out = []
        for b in nonempty:
            k = min(per_class, len(b))
            idx = np.random.choice(len(b), size=k, replace=False)
            out += [b[i] for i in idx]
        return out

    def size(self) -> int:
        return sum(len(b) for b in self.buckets)


class RoTTATTA(TTAMethod):
    """RoTTA baseline (see `RoTTAConfig` docstring for the full mechanism mapping
    and citation). Predict-then-adapt per the shared protocol: this beat's reported
    logits always come from the live student BEFORE any update this step touches it."""

    def __init__(self, cfg: Optional[RoTTAConfig] = None):
        self.cfg = cfg or RoTTAConfig()
        self.name = f"rotta_{self.cfg.scope}"
        self.opt: Optional[torch.optim.Optimizer] = None
        self.mem: Optional[_CSTUMemory] = None
        self._teacher_affine: Dict[str, torch.Tensor] = {}
        self._step = 0

    def reset(self, model: nn.Module) -> None:
        if self.cfg.scope == "sparse":
            params = model.configure_for_tta()                       # dw_bn affine only
        elif self.cfg.scope == "full_affine":
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            params = []
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    params += [m.weight, m.bias]
        else:
            raise ValueError(f"unknown RoTTA scope '{self.cfg.scope}'")
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self.mem = _CSTUMemory(NUM_CLASSES, self.cfg.memory_per_class,
                               self.cfg.lambda_t, self.cfg.lambda_u)
        # teacher starts as an exact copy of the (source) dw_bn affine snapshot
        self._teacher_affine = {n: p.detach().clone() for n, p in model.tta_named_parameters()}
        self._step = 0

    @torch.no_grad()
    def _teacher_forward(self, model: nn.Module, x: torch.Tensor,
                         aux: Optional[torch.Tensor]) -> torch.Tensor:
        """Backbone forward with dw_bn affine substituted by the EMA teacher snapshot
        (mirrors base_cnn.py's `forward_anchor()` substitution pattern via
        `functional_call`, teacher snapshot in place of the fixed source snapshot),
        then through `adaptable_head` -- frozen in sparse scope, so this is exactly
        RoTTA's separate, more-stable teacher network."""
        src_params = dict(model.backbone.named_parameters())
        for full_name, val in self._teacher_affine.items():
            key = full_name[len("backbone."):] if full_name.startswith("backbone.") else full_name
            if key in src_params:
                src_params[key] = val.to(x.device)
        src_buffers = dict(model.backbone.named_buffers())
        feat = functional_call(model.backbone, (src_params, src_buffers), (x,))
        if model.cfg.aux_feat_dim > 0:
            feat = torch.cat([feat, aux], dim=-1)
        return model.adaptable_head(feat)

    @torch.no_grad()
    def _ema_update_teacher(self, model: nn.Module) -> None:
        a = self.cfg.ema_alpha
        for n, p in model.tta_named_parameters():
            if n in self._teacher_affine:
                self._teacher_affine[n].mul_(a).add_(p.detach(), alpha=1.0 - a)
            else:
                self._teacher_affine[n] = p.detach().clone()

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        self._step += 1

        # 1. causal prediction FIRST: this beat's reported logits use the live
        #    student weights as they stood before any update this step applies.
        with torch.no_grad():
            pred_logits = model(x, aux=aux).logits
            anchor_logits = model.forward_anchor(x, aux=aux)

        # 2. EMA teacher pseudo-labels this beat; CSTU offers it to its bucket.
        teacher_logits = self._teacher_forward(model, x, aux)
        pseudo_label = int(teacher_logits.argmax(dim=-1).item())
        uncertainty = float(predictive_entropy(teacher_logits).item())
        mem_admitted = self.mem.add(x.detach(), aux.detach() if aux is not None else None,
                                    pseudo_label, uncertainty, self._step)

        # 3. class-balanced replay update, every `update_every` beats. `adapted`/
        #    `rejected` follow this file's shared convention (a gradient update
        #    happened vs. one was due but withheld) -- NOT whether this beat's own
        #    sample was admitted into memory, which is reported separately in `extra`.
        adapted, rejected, loss_val = False, False, None
        if self._step % self.cfg.update_every == 0:
            batch = self.mem.sample_balanced(self.cfg.batch_size)
            if batch:
                xb = torch.cat([b["x"] for b in batch], dim=0)
                yb = torch.tensor([b["y"] for b in batch], dtype=torch.long, device=x.device)
                ages = torch.tensor([max(0, self._step - b["ins_step"]) for b in batch],
                                    dtype=torch.float32, device=x.device)
                w = torch.exp(-ages / self.cfg.timeliness_tau)     # recent items weighted higher
                aux_b = None
                if aux is not None:
                    aux_b = torch.cat([b["aux"] if b["aux"] is not None
                                       else torch.zeros_like(aux) for b in batch], dim=0)
                out = model(xb, aux=aux_b)
                per_sample = F.cross_entropy(out.logits, yb, reduction="none")
                loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self._ema_update_teacher(model)
                adapted = True
                loss_val = float(loss.detach())
            else:
                rejected = True            # update was due but memory had nothing to sample yet

        return StepResult(logits=pred_logits, anchor_logits=anchor_logits,
                          adapted=adapted, rejected=rejected,
                          extra={"adapt_loss": loss_val, "pseudo_label": pseudo_label,
                                 "mem_admitted": mem_admitted, "mem_size": self.mem.size()})


# --------------------------------------------------------------------------- #
# 5. BeatRhythm-TTA, beat-level adaptation
#    (Jiang, Deng, Zhou, Wang, Ou, Zheng, arXiv:2608.23347, 2026)
# --------------------------------------------------------------------------- #
@dataclass
class BeatRhythmTTAConfig:
    """
    Beat-level adaptation of BeatRhythm-TTA (Jiang et al., arXiv:2608.23347). Defaults
    follow the authors' released code (github.com/Vincent-Wenhan/BeatRhythm-TTA,
    src/method/beatrhythmtta.py + configs/method/beatrhythmtta.yaml) where the paper
    text is silent or differs; both are listed here so nothing is hidden.

    Kept as published
      * SQI = sigmoid(6*(conc-0.35)) * sigmoid(2*(sharp-1)) * sigmoid(6*(0.35-bwr))
        (paper Eq. 3 states plain sigmoids; the released code uses these shifted/sharpened
        ones, which is what makes the tau_q=0.05 gate able to fire at all).
      * EMA teacher (mu=0.999) pseudo-labels the weak view; student sees a strong view
        (gain N(1,0.06), noise std 0.006, one 4% mask); pseudo-label loss is masked by
        teacher confidence > pos_th (0.7) AND sample SQI >= tau_q (0.05).
      * Cosine consistency between weak and strong views on a mid-level feature map:
        global (mean-pooled) term weight 1.0, beat-centric term weight 0.5 with
        SQI weights floored at 0.05. Gradient flows through the weak view only
        (stop-gradient on the strong view), as in the released code; the paper text
        says the opposite (sg on the clean view).
      * Student BatchNorm uses batch statistics and never updates running stats;
        the teacher uses the frozen source running stats; the REPORTED prediction
        comes from the teacher.
      * Online/continual protocol: Adam, lr 1e-6, 5 steps per sample, no reset.

    Changed because our stream is one beat at a time (state these in the paper)
      * Rhythm level does not exist (one beat per sample): the 'global' term is the
        mean-pooled feature and the 'beat' term pools +-w_feat positions around the
        R peak. They overlap far more than in the 12-lead 10 s setting.
      * Single-label softmax (5 AAMI classes) instead of multi-label sigmoid: the
        code's own single-label branch is used (KL(teacher||student), mask = max
        teacher prob > pos_th).
      * Beat window is a fixed 256 samples at 360 Hz with the R peak at r_index; the
        released code's median-RR windows are replaced by a 0.4 s fallback RR
        (the code's own fallback is 200 samples at 500 Hz), scaled to 360 Hz.
      * `scope="sparse"` adapts only dw_bn affine (same edge budget as every other
        baseline here); the authors adapt all BatchNorm affine ("full_affine").
      * Predict-then-adapt: the reported logits are the teacher's output BEFORE this
        beat's update (same protocol as every other method). The authors report the
        teacher's output AFTER the update (`predict_after_update=True`).
    """
    lr: float = 1e-6
    steps: int = 5
    ema_m: float = 0.999
    lambda_beat: float = 0.5
    lambda_global: float = 1.0
    lambda_pl: float = 1.0
    pos_th: float = 0.7
    tau_q: float = 0.05
    sqi_floor: float = 0.05
    w_feat: int = 4
    s_noise_std: float = 0.006
    s_gain_std: float = 0.06
    s_mask_ratio: float = 0.04
    fs: float = 360.0
    r_index: int = 96
    fallback_rr_s: float = 0.4
    scope: str = "sparse"                 # "sparse" (dw_bn affine) | "full_affine" (as published)
    predict_after_update: bool = False
    feat_module: str = "backbone.proj"    # [B,64,32] feature map, the analogue of their 'layer3'


def beat_sqi(x: torch.Tensor, r_index: int, fs: float, fallback_rr_s: float = 0.4,
            eps: float = 1e-6) -> float:
    """BeatRhythm-TTA's per-beat SQI (released-code version) for one window.
    x: [1, C, L] -> float in [0, 1]. Higher = cleaner."""
    _, _, T = x.shape
    med_rr = fallback_rr_s * fs
    pre, post, cen = int(0.25 * med_rr), int(0.45 * med_rr), int(0.10 * med_rr)
    s, e = max(0, r_index - pre), min(T, r_index + post)
    seg = x[0, :, s:e]
    if seg.numel() < 16:
        return 0.0
    E = (seg ** 2).mean()
    cs, ce = max(0, r_index - cen - s), min(e - s, r_index + cen - s)
    center = seg[:, cs:ce] if ce > cs else seg
    conc = ((center ** 2).mean() / (E + eps)).clamp(0, 10)
    L = int(seg.shape[-1])
    k = int(max(5, min(L, int(0.8 * med_rr))))
    base = F.avg_pool1d(seg.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
    bwr = ((base ** 2).mean() / (E + eps)).clamp(0, 10)
    absd = (seg[:, 1:] - seg[:, :-1]).abs().reshape(-1)
    kk = max(1, min(absd.numel(), int(np.ceil(0.95 * absd.numel()))))
    sharp = absd.kthvalue(kk).values / (seg.pow(2).mean().sqrt() + eps)
    q = (torch.sigmoid(6.0 * (conc - 0.35)) * torch.sigmoid(2.0 * (sharp - 1.0))
         * torch.sigmoid(6.0 * (0.35 - bwr)))
    return float(q.clamp(0.0, 1.0))


class BeatRhythmTTA(TTAMethod):
    """Beat-level BeatRhythm-TTA baseline; see `BeatRhythmTTAConfig` for every deviation."""

    def __init__(self, cfg: Optional[BeatRhythmTTAConfig] = None):
        self.cfg = cfg or BeatRhythmTTAConfig()
        tag = f"{self.cfg.lr:.0e}".replace("e-0", "e-")
        self.name = f"beatrhythm_tta_lr{tag}"
        self.opt: Optional[torch.optim.Optimizer] = None
        self.teacher: Optional[nn.Module] = None
        self._params: list = []
        self._pairs: list = []
        self._hook = None
        self._feat: Optional[torch.Tensor] = None

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _bn_modules(model: nn.Module) -> list:
        return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    def reset(self, model: nn.Module) -> None:
        if self.cfg.scope == "sparse":
            params = model.configure_for_tta()
        elif self.cfg.scope == "full_affine":
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            params = []
            for m in self._bn_modules(model):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                params += [m.weight, m.bias]
        else:
            raise ValueError(f"unknown BeatRhythmTTA scope '{self.cfg.scope}'")
        self._params = params
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr)
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        t_named = dict(self.teacher.named_parameters())
        self._pairs = [(t_named[n], p) for n, p in model.named_parameters() if p.requires_grad]
        for m in self._bn_modules(model):          # train-mode BN uses batch stats and must
            m.track_running_stats = False          # never touch the (shared) running stats
        mod = dict(model.named_modules())[self.cfg.feat_module]
        if self._hook is not None:
            self._hook.remove()
        self._hook = mod.register_forward_hook(lambda _m, _i, out: setattr(self, "_feat", out))

    def _student_train_mode(self, model: nn.Module) -> None:
        model.eval()
        for m in self._bn_modules(model):
            m.train()

    def _strong_view(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        y = x * (1.0 + c.s_gain_std * torch.randn(x.shape[0], 1, 1, device=x.device))
        y = y + c.s_noise_std * torch.randn_like(y)
        L = max(1, int(x.shape[-1] * c.s_mask_ratio))
        s = int(torch.randint(0, max(1, x.shape[-1] - L + 1), (1,)))
        y = y.clone()
        y[..., s:s + L] = y.mean(dim=-1, keepdim=True)
        return y

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:        # stop-gradient on b
        return 1.0 - (F.normalize(a, dim=-1) * F.normalize(b.detach(), dim=-1)).sum(-1)

    @torch.no_grad()
    def _ema(self) -> None:
        m = self.cfg.ema_m
        for pt, ps in self._pairs:
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)

    # ---- one beat ---------------------------------------------------------- #
    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        c = self.cfg
        aux = step_aux(meta, x.device)
        sqi = beat_sqi(x, c.r_index, c.fs, c.fallback_rr_s)
        with torch.no_grad():
            logits_out = self.teacher(x, aux=aux).logits
        pl_open = sqi >= c.tau_q
        loss_val = None

        for _ in range(c.steps):
            x_s = self._strong_view(x)
            with torch.no_grad():
                logits_t = self.teacher(x, aux=aux).logits            # weak view == identity
            self._student_train_mode(model)
            model(x, aux=aux)                                          # weak view features
            f_w = self._feat
            logits_s = model(x_s, aux=aux).logits
            f_s = self._feat

            loss_global = self._cos(f_w.mean(dim=-1), f_s.mean(dim=-1)).mean()
            Tf = f_w.shape[-1]
            r = int(round(c.r_index / max(x.shape[-1] - 1, 1) * (Tf - 1)))
            lo, hi = max(0, r - c.w_feat), min(Tf, r + c.w_feat + 1)
            loss_beat = max(sqi, c.sqi_floor) * self._cos(
                f_w[..., lo:hi].mean(dim=-1), f_s[..., lo:hi].mean(dim=-1)).mean()

            pt = F.softmax(logits_t, dim=-1)
            confident = float(pt.max() > c.pos_th) * float(pl_open)
            kl = (pt * (torch.log(pt.clamp_min(1e-6)) - F.log_softmax(logits_s, dim=-1))).sum(-1)
            loss_pl = (kl * confident).sum()                            # 0 when masked out

            loss = c.lambda_pl * loss_pl + c.lambda_beat * loss_beat + c.lambda_global * loss_global
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self._ema()
            loss_val = float(loss.detach())
        model.eval()                                  # restore: forward_anchor/predictions use running stats

        if c.predict_after_update:
            with torch.no_grad():
                logits_out = self.teacher(x, aux=aux).logits
        with torch.no_grad():
            anchor_logits = model.forward_anchor(x, aux=aux)
        return StepResult(logits=logits_out, anchor_logits=anchor_logits, adapted=True,
                          rejected=not pl_open, sqi=sqi,
                          extra={"adapt_loss": loss_val, "pl_gate_open": pl_open})


# --------------------------------------------------------------------------- #
# Self-test: synthetic chronological stream, space-shifted with a mission-onset
# ramp, run through all three baselines with a fresh copy of the SAME source model.
# --------------------------------------------------------------------------- #
class _SyntheticStreamDataset:
    """Minimal MITBIHDataset duck-type: .x [N,C,L] .y [N] .record_ids [N] .sample_idx [N]."""

    def __init__(self, n_records: int = 3, beats_per_record: int = 200,
                c: int = 1, l: int = 256, seed: int = 0, start: int = 5000, step: int = 300):
        g = torch.Generator().manual_seed(seed)
        t = torch.linspace(-1, 1, l)
        templates = torch.stack([torch.sin((k + 1) * 3.0 * t) for k in range(NUM_CLASSES)])  # [K, L]
        xs, ys, rids, sidx = [], [], [], []
        for r in range(n_records):
            labels = torch.randint(0, NUM_CLASSES, (beats_per_record,), generator=g)
            for i in range(beats_per_record):
                k = int(labels[i])
                x = templates[k].view(1, l).repeat(c, 1) + 0.15 * torch.randn(c, l, generator=g)
                xs.append(x); ys.append(k); rids.append(r); sidx.append(start + i * step)
        self.x = torch.stack(xs)                                       # [N, C, L]
        self.y = torch.tensor(ys, dtype=torch.long)                    # [N]
        self.record_ids = torch.tensor(rids, dtype=torch.long)         # [N]
        self.sample_idx = torch.tensor(sidx, dtype=torch.long)         # [N]

    def __len__(self) -> int:
        return self.x.shape[0]


if __name__ == "__main__":
    from base_cnn import CNNConfig, BeatCNN
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp

    print("[baselines_tta] self-test: Source-Only vs. TENT(sparse) vs. SQI-gated TENT "
         "on a synthetic, space-shifted stream (no PhysioNet download needed).")
    torch.manual_seed(0)
    source_model = BeatCNN(CNNConfig(in_channels=1))

    ds = _SyntheticStreamDataset(n_records=3, beats_per_record=200, c=1, l=256, seed=0)

    # Brief supervised warm-up BEFORE freeze_anchor(), mirroring train_stage1.py's real
    # order (theta_f is a snapshot of a TRAINED model): a random-init model's near-uniform
    # outputs give near-zero entropy gradients almost everywhere, making TENT/SQI-gated TENT's
    # drift technically correct but numerically negligible in this smoke test otherwise.
    # Mini-batched clean beats only (no space shift) -- this stands in for Stage 1's DS1.
    opt = torch.optim.Adam(list(source_model.backbone.parameters())
                           + list(source_model.adaptable_head.parameters()), lr=1e-3)
    source_model.train()
    for _ in range(15):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(source_model(ds.x[idx]).logits, ds.y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    source_model.eval()
    source_model.freeze_anchor()                 # NOW lock the warmed-up state as theta_f
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=0))
    # calm for the first 100 beats (calibration window), then linearly ramps to severity=1
    severity_schedule = linear_ramp(start_idx=100, n_ramp=300)

    cfg = StreamingConfig(device="cpu", rolling_window=100, log_every=0)
    methods = [SourceOnlyTTA(), TentTTA(TentConfig(scope="sparse")), SQIGatedTent(),
              RoTTATTA(RoTTAConfig(memory_per_class=8, batch_size=8)),
              BeatRhythmTTA(BeatRhythmTTAConfig(lr=1e-4, steps=2))]
    results = {}
    for method in methods:
        m = copy.deepcopy(source_model)                                # identical starting weights
        stream = make_beat_stream(ds, n_beats=None, sim=sim, severity=severity_schedule)
        ev = StreamingEvaluator(method, cfg)
        res = ev.run(m, stream)
        results[method.name] = res
        res.save_csv(f"./results/_selftest_baselines_tta/{method.name}.csv")
        res.save_json(f"./results/_selftest_baselines_tta/{method.name}.json")

    print()
    print(compare_methods(results))

    assert results["source_only"].summary["adaptation_rate"] == 0.0
    assert results["source_only"].summary["mean_js_div"] < 1e-6, \
        "Source-Only must show ~0 divergence from its own anchor"
    assert results["tent_sparse"].summary["adaptation_rate"] == 1.0
    assert results["tent_sparse"].summary["mean_js_div"] > 1e-4, \
        "TENT adapting on a warmed-up (confident) model must show non-negligible drift"
    br = results["sqi_gated_tent"].summary
    assert 0.0 <= br["rejection_rate"] < 1.0, "gate must not reject every beat in this scenario"
    print(f"\nSQI gate: rejected {100*br['rejection_rate']:.1f}% of beats "
         f"(calibrated threshold after {SQIGatedTentConfig().calib_beats} beats).")

    rotta = results["rotta_sparse"].summary
    assert rotta["adaptation_rate"] > 0.0, "RoTTA must take at least one replay update"
    assert rotta["mean_js_div"] > 0.0, "RoTTA's dw_bn affine must move at all from source"
    print(f"\nRoTTA: adaptation_rate={100*rotta['adaptation_rate']:.1f}% "
         f"mean_js_div={rotta['mean_js_div']:.3e} "
         f"(class-balanced CSTU replay + EMA teacher, memory_per_class=8 in this self-test).")

    # ---- BeatRhythm-TTA specific checks ------------------------------------------------
    brt = results["beatrhythm_tta_lr1e-4"].summary
    assert brt["adaptation_rate"] == 1.0 and brt["mean_js_div"] > 0.0
    # (a) SQI: a clean QRS-centred beat must outscore the same beat swamped by wander + bursts
    t = torch.linspace(-1, 1, 256)
    clean = (torch.exp(-((t + 0.25) / 0.03) ** 2) * 3.0 + 0.05 * torch.randn(256)).view(1, 1, 256)
    bad = clean + 2.5 * torch.sin(torch.linspace(0, 3, 256)).view(1, 1, 256)
    bad[..., 150:158] += 6.0
    q_clean, q_bad = beat_sqi(clean, 96, 360.0), beat_sqi(bad, 96, 360.0)
    print(f"BeatRhythm-TTA SQI: clean={q_clean:.3f}  wander+burst={q_bad:.3f}")
    assert q_clean > q_bad, "SQI must rank a clean beat above a corrupted one"
    # (b) running stats must be untouched (student BN is train-mode, batch stats only) and
    #     the student must actually move while the source snapshot stays put
    m2 = copy.deepcopy(source_model)
    bn_before = {k: v.clone() for k, v in m2.state_dict().items() if "running" in k}
    meth = BeatRhythmTTA(BeatRhythmTTAConfig(lr=1e-3, steps=3))
    meth.reset(m2)
    p0 = [p.detach().clone() for p in m2.tta_parameters()]
    for i in range(10):
        meth.step(m2, ds.x[i:i + 1], {})
    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in bn_before.items()), \
        "BeatRhythm-TTA must never modify BatchNorm running statistics"
    assert any(not torch.equal(a, b) for a, b in zip(p0, m2.tta_parameters())), "student must adapt"
    assert not m2.training, "model must be left in eval mode after each step"
    # (c) lr=0 -> teacher == student == source: reported logits identical to Source-Only
    m3 = copy.deepcopy(source_model)
    meth0 = BeatRhythmTTA(BeatRhythmTTAConfig(lr=0.0, steps=2))
    meth0.reset(m3)
    r0 = meth0.step(m3, ds.x[:1], {})
    with torch.no_grad():
        ref = source_model.eval()(ds.x[:1]).logits
    assert torch.allclose(r0.logits, ref, atol=1e-5), "lr=0 must reproduce the source model"
    print("BeatRhythm-TTA checks: SQI ordering, BN running stats untouched, adapts, lr=0 == source -> OK")
    print("Self-test passed.")
