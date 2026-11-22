"""
ts_sqi_masker.py  --  this project, Stage 5 (Temporal-Spectral SQI Masking)

Computes a continuous 2D time-frequency quality mask M_SQI(t, f) in [0, 1] over an
ECG beat window via STFT, and uses it to build a SOFT, sub-window-masked
reconstruction of the beat -- as opposed to `baselines_tta.SQIGatedTent`,
which computes one SCALAR quality value for the WHOLE beat and either keeps or
completely drops it. Here, a beat that is 70% clean and 30% radiation-burst gets
its clean 70% preserved and its dirty 30% suppressed, in both time AND frequency,
rather than being thrown away wholesale.

What "masking the entropy gradient" means concretely
------------------------------------------------------
A classifier's entropy loss is a single scalar over the softmax output -- it has no
native per-(t,f) structure to mask. So the mask is applied to the INPUT, not to the
loss: `x_masked = ISTFT(M_SQI(t,f) [detached] * STFT(x))`. Because STFT/ISTFT are
linear and the mask is stop-gradient (it is a property of the raw signal, not of the
model), gradients of `entropy(model(x_masked).logits)` w.r.t. the adapting dw_bn
parameters are naturally dominated by the time-frequency cells the mask kept --
"backpropagating only through clean regions" falls out of ordinary backprop through
a denoised input, with no custom autograd needed.

M_SQI(t, f) is the product of two soft, differentiable-shaped (though itself
computed stop-gradient) factors:
  1. A band prior B(f): a soft passband indicator for `[low_hz, high_hz]`, matching
     dataset.py's own 0.5-40 Hz bandpass -- down-weights baseline wander (< 0.5 Hz)
     and broadband ADC/EMG/radiation noise (> 40 Hz), independent of t.
  2. A per-beat, per-frequency burst detector: this beat's own median magnitude
     across time at each frequency is used as a robust (median is insensitive to a
     short burst occupying a few frames) in-beat reference. A cell whose magnitude
     exceeds an effective threshold times its frequency's reference gets smoothly
     down-weighted -- this is what catches a LOCALIZED radiation burst or SEU spike
     even when it falls inside the passband, which a static band-prior alone cannot.

     Caveat this masker must account for: an ECG beat is ITSELF non-stationary --
     QRS energy is naturally concentrated in a handful of the ~33 STFT frames, which
     is the same statistical signature (rare, high-magnitude frames vs. a low
     per-frequency median) as a genuine radiation burst. A burst detector calibrated
     against a beat's OWN median without accounting for this would flag the QRS
     complex as noise and suppress exactly the content that matters (confirmed
     empirically: an earlier version of this file failed its own self-test this way).
     The fix is an `expected_activity(t)` soft prior -- two Gaussian bumps at the
     R-peak and T-wave sample positions (beats are always R-peak-aligned by
     `dataset.py`'s cut) -- that RAISES the burst threshold during physiologically
     expected transients and leaves it tight everywhere else (the isoelectric
     segments, where energy genuinely should be flat and any spike is suspect).

STFT settings (n_fft=32, hop=8, win=32, Hann) satisfy the constant-overlap-add (COLA)
condition (hop = n_fft/4 for Hann), so ISTFT reconstruction is near-exact when
M_SQI == 1 everywhere (verified in the self-test).

Shape legend: B = batch, C = leads, L = 256 samples, F = n_fft//2+1 freq bins,
T = number of STFT frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TSSQIConfig:
    fs: float = 360.0
    n_fft: int = 32
    hop_length: int = 8            # = n_fft/4 -> COLA-exact for a Hann window
    win_length: int = 32
    low_hz: float = 0.5            # matches dataset.py's bandpass lower edge
    high_hz: float = 40.0          # matches dataset.py's bandpass upper edge / SQI-gate cutoff
    # Separate transition widths: the low edge sits at only 0.5 Hz, so it needs a tight
    # sigmoid (a wide one would barely distinguish DC/baseline-wander from passband at
    # this scale); the high edge sits at 40 Hz with 140 Hz of headroom to Nyquist, so a
    # wider, gentler transition there is both fine and more robust to bin quantization.
    low_softness_hz: float = 0.2
    high_softness_hz: float = 5.0
    burst_ratio_thresh: float = 3.0  # cell/ref-median magnitude ratio where burst quality starts decaying
    burst_softness: float = 1.0      # sigmoid steepness in log-ratio units (smaller = sharper cutoff)

    # Expected-activity prior (see module docstring): where genuine QRS/T transients
    # are expected, `burst_ratio_thresh` is scaled up so real signal isn't flagged.
    pre_samples: int = 96            # R-peak sample offset -- matches dataset.py's `pre`
    qrs_sigma_s: float = 0.045       # slightly wider than space_shift_simulator's own (benefit of the doubt)
    t_center_s: float = 0.27
    t_sigma_s: float = 0.09
    expected_activity_boost: float = 6.0   # threshold multiplier at full expected-activity

    eps: float = 1e-8


class TSSQIMasker:
    """Stateless given its config (only `fs`/`n_fft` fix shapes); safe to share across beats."""

    def __init__(self, cfg: TSSQIConfig | None = None):
        self.cfg = cfg or TSSQIConfig()
        self._window_cache: Dict[torch.device, torch.Tensor] = {}
        self._freqs_cache: Dict[torch.device, torch.Tensor] = {}

    def _window(self, device: torch.device) -> torch.Tensor:
        if device not in self._window_cache:
            self._window_cache[device] = torch.hann_window(self.cfg.win_length, device=device)
        return self._window_cache[device]

    def _freqs(self, device: torch.device) -> torch.Tensor:
        if device not in self._freqs_cache:
            self._freqs_cache[device] = torch.fft.rfftfreq(
                self.cfg.n_fft, d=1.0 / self.cfg.fs).to(device)
        return self._freqs_cache[device]

    def _expected_activity(self, n_frames: int, device: torch.device) -> torch.Tensor:
        """
        [T] in [0,1]: soft prior for "a genuine QRS/T transient is expected here",
        evaluated at each STFT frame's approximate center sample (frame_idx * hop,
        which is accurate up to the analysis window's half-width -- adequate for a
        soft prior, not a precise event detector).
        """
        cfg = self.cfg
        frame_samples = torch.arange(n_frames, dtype=torch.float32, device=device) * cfg.hop_length
        qrs = torch.exp(-0.5 * ((frame_samples - cfg.pre_samples) / (cfg.qrs_sigma_s * cfg.fs)) ** 2)
        t_wave = torch.exp(-0.5 * ((frame_samples - (cfg.pre_samples + cfg.t_center_s * cfg.fs))
                                   / (cfg.t_sigma_s * cfg.fs)) ** 2)
        return torch.maximum(qrs, t_wave)                                 # [T]

    # ------------------------------------------------------------------ #
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] -> complex STFT [B, C, F, T]."""
        B, C, L = x.shape
        X = torch.stft(x.reshape(B * C, L), n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
                       win_length=self.cfg.win_length, window=self._window(x.device),
                       center=True, return_complex=True)                  # [B*C, F, T]
        return X.reshape(B, C, *X.shape[-2:])                             # [B, C, F, T]

    @torch.no_grad()
    def compute_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, C, L] -> (mask [B, F, T] in [0,1], info dict with diagnostic tensors).
        Channels are combined by averaging magnitude (a burst/wander event is treated
        as a property of the beat, shared across leads, which is a simplification --
        see the module docstring). The burst ratio deliberately uses MAGNITUDE, not
        power: a QRS complex with, say, 10x the amplitude of the baseline noise floor
        is normal and expected, but that is a 100x POWER ratio -- tuning a power-domain
        threshold to tolerate genuine QRS peaks would make it nearly blind to real
        bursts of comparable or lesser amplitude. Magnitude keeps the ratio's dynamic
        range interpretable and the threshold/boost parameters meaningful.
        """
        cfg = self.cfg
        X = self._stft(x)                                                 # [B, C, F, T] complex
        mag = X.abs().mean(dim=1)                                         # [B, F, T]
        freqs = self._freqs(x.device)                                     # [F]

        lo = torch.sigmoid((freqs - cfg.low_hz) / cfg.low_softness_hz)    # [F]
        hi = torch.sigmoid((cfg.high_hz - freqs) / cfg.high_softness_hz)  # [F]
        band = (lo * hi).view(1, -1, 1)                                   # [1, F, 1]

        ref = mag.median(dim=-1, keepdim=True).values                    # [B, F, 1] robust per-freq reference
        ratio = mag / (ref + cfg.eps)                                     # [B, F, T]
        log_ratio = torch.log(ratio.clamp_min(cfg.eps))

        activity = self._expected_activity(mag.shape[-1], x.device)       # [T]
        eff_thresh = cfg.burst_ratio_thresh * (1.0 + cfg.expected_activity_boost * activity)  # [T]
        log_eff_thresh = torch.log(eff_thresh).view(1, 1, -1)             # [1, 1, T]
        burst_q = torch.sigmoid(-(log_ratio - log_eff_thresh) / cfg.burst_softness)  # [B, F, T]

        mask = (band * burst_q).clamp(0.0, 1.0)                           # [B, F, T]
        return mask, {"magnitude": mag, "freqs": freqs, "ref": ref, "expected_activity": activity}

    def apply(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C, L] -> (x_masked [B, C, L], mask [B, F, T], coverage [B]).
        `coverage` = mean(mask), a per-beat scalar diagnostic in [0,1] ("what fraction
        of this beat's time-frequency content was judged trustworthy").
        """
        cfg = self.cfg
        B, C, L = x.shape
        X = self._stft(x)                                                 # [B, C, F, T] complex, WITH grad if x does
        mask, info = self.compute_mask(x)                                 # [B, F, T], stop-gradient by construction
        Xm = X * mask.unsqueeze(1).detach().to(X.dtype)                   # [B, C, F, T]
        x_masked = torch.istft(Xm.reshape(B * C, *Xm.shape[-2:]),
                               n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
                               window=self._window(x.device), center=True, length=L)   # [B*C, L]
        x_masked = x_masked.reshape(B, C, L)
        coverage = mask.mean(dim=(-2, -1))                                # [B]
        return x_masked, mask, coverage


# --------------------------------------------------------------------------- #
# Self-test: 2D mask computation -- band prior, localized burst suppression,
# near-exact reconstruction when clean, and integration with a real backward pass.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import torch.nn.functional as F2

    torch.manual_seed(0)
    cfg = TSSQIConfig()
    masker = TSSQIMasker(cfg)
    fs, L = cfg.fs, 256

    # QRS at t = pre_samples/fs and T-wave at + t_center_s, matching the SAME
    # R-peak-aligned convention TSSQIConfig's expected-activity prior assumes
    # (== dataset.py's `pre` cut convention) -- a mismatch here would make this a
    # self-test bug, not a masker bug. A small continuous noise floor is added
    # because a perfectly-silent-outside-the-bumps signal is unrealistic (real,
    # z-scored ECG always has SOME residual baseline energy) and creates a
    # pathological peak/median power ratio (~300x) that no reasonable burst
    # threshold should be tuned to accommodate, since that would also let real
    # bursts through.
    # QRS width matches cfg.qrs_sigma_s (the same convention used throughout this
    # project, e.g. space_shift_simulator.py's own qrs_sigma_s~=0.035-0.045s) --
    # an earlier, much narrower (0.012s) test pulse concentrated energy into
    # nearly one STFT frame, which genuinely resembles a burst; a real QRS
    # complex (~80-120 ms) does not.
    torch.manual_seed(1)
    t = torch.arange(L, dtype=torch.float32) / fs
    qrs_t0 = cfg.pre_samples / fs
    t_wave_t0 = qrs_t0 + cfg.t_center_s
    clean = (1.0 * torch.exp(-0.5 * ((t - qrs_t0) / cfg.qrs_sigma_s) ** 2)                # QRS-like pulse
            + 0.3 * torch.sin(2 * math.pi * 2.0 * t) * torch.exp(-0.5 * ((t - t_wave_t0) / cfg.t_sigma_s) ** 2)  # T-wave-like
            + 0.06 * torch.randn(L))                                                       # baseline noise floor
    x_clean = clean.view(1, 1, L)

    # ---- 1. band prior: pure out-of-band tones should be heavily masked --------
    lf = torch.sin(2 * math.pi * 0.1 * t).view(1, 1, L)                    # 0.1 Hz: below low_hz=0.5
    hf = torch.sin(2 * math.pi * 120.0 * t).view(1, 1, L)                  # 120 Hz: above high_hz=40
    mask_lf, _ = masker.compute_mask(lf)
    mask_hf, _ = masker.compute_mask(hf)
    freqs = masker._freqs(torch.device("cpu"))
    lf_bin = int((freqs - 0.1).abs().argmin())
    hf_bin = int((freqs - 120.0).abs().argmin())
    print(f"mean mask at ~0.1 Hz bin (below passband): {mask_lf[0, lf_bin].mean():.3f}  (expect low)")
    print(f"mean mask at ~120 Hz bin (above passband): {mask_hf[0, hf_bin].mean():.3f}  (expect low)")
    assert mask_lf[0, lf_bin].mean() < 0.15
    assert mask_hf[0, hf_bin].mean() < 0.15

    # ---- 2. clean, in-band signal: mask should be comfortably above the "reject"
    # range, NOT necessarily ~1.0 everywhere -- burst_softness=1.0 is a deliberately
    # gradual sigmoid (soft masking, not a hard cutoff), so even a genuinely clean
    # QRS complex settles around 0.7-0.9, not exactly 1.0. The property that matters
    # is CONTRAST against corrupted regions (checked in test 3 below), not absolute
    # near-unity on clean input.
    mask_clean, _ = masker.compute_mask(x_clean)
    in_band = (freqs >= cfg.low_hz + 2 * cfg.low_softness_hz) & (freqs <= cfg.high_hz - cfg.high_softness_hz)
    print(f"mean in-band mask on clean beat: {mask_clean[:, in_band].mean():.3f}  (expect comfortably high, ~0.7-0.9)")
    assert mask_clean[:, in_band].mean() > 0.6

    # COLA round-trip correctness, ISOLATED from the SQI mask's intended attenuation:
    # force mask==1 (identity) and verify ISTFT(STFT(x)) reconstructs x almost exactly.
    # This is the right way to test "is the STFT/ISTFT plumbing correct" separately
    # from "does the mask attenuate out-of-band/bursty content" (tested elsewhere) --
    # comparing the ACTUAL (masked, intentionally band-limited) reconstruction against
    # the unfiltered original would conflate the two and fail for the wrong reason
    # (14 of 17 frequency bins are outside the 0.5-40 Hz passband by design).
    X = masker._stft(x_clean)
    x_cola = torch.istft(X.reshape(1, *X.shape[-2:]), n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                         win_length=cfg.win_length, window=masker._window(x_clean.device),
                         center=True, length=L).reshape(1, 1, L)
    cola_err = float((x_cola - x_clean).pow(2).mean().sqrt() / x_clean.pow(2).mean().sqrt())
    print(f"COLA round-trip (mask forced to identity) relative RMS error: {cola_err:.5f}")
    assert cola_err < 0.02, "ISTFT(STFT(x)) must reconstruct x almost exactly (COLA-exact Hann window)"

    # Full-spectrum mask coverage is naturally low (most bins ARE legitimately out of
    # band for a real ECG signal) -- the meaningful number is the in-band one, above.
    _, _, cov = masker.apply(x_clean)
    print(f"full-spectrum coverage (includes correctly-suppressed out-of-band bins): {float(cov):.3f}")

    # ---- 3. localized burst: SUB-WINDOW masking, not whole-beat rejection ------
    # Placed well BEFORE the QRS onset (pre_samples=96, qrs_sigma_s=0.045s*fs~=16
    # samples -> activity ~0 by sample ~50), i.e. in a genuinely quiet isoelectric
    # region, so this test isn't confounded by the expected-activity boost.
    x_burst = x_clean.clone()
    burst_start, burst_len = 20, 15
    x_burst[..., burst_start:burst_start + burst_len] += 4.0 * torch.randn(burst_len)
    x_rec_b, mask_b, cov_b = masker.apply(x_burst)
    # error should be concentrated near the burst, small in the (untouched) QRS/T region
    err_at_burst = float((x_rec_b[..., burst_start:burst_start + burst_len]
                         - x_clean[..., burst_start:burst_start + burst_len]).abs().mean())
    err_away = float((x_rec_b[..., 90:210] - x_clean[..., 90:210]).abs().mean())
    print(f"coverage with localized burst: {float(cov_b):.3f} (expect < clean-signal coverage {float(cov):.3f})")
    print(f"|error| at burst region (suppressed by masking): {err_at_burst:.3f}  vs. away from it: {err_away:.3f}")
    assert float(cov_b) < float(cov), "a localized burst must lower overall coverage vs. the clean beat"
    assert err_away < 0.3, "regions far from the burst must be reconstructed cleanly (sub-window property)"

    # ---- 4. integration: masked input must support a real backward pass --------
    from base_cnn import CNNConfig, BeatCNN
    from streaming_eval import predictive_entropy

    model = BeatCNN(CNNConfig(in_channels=1))
    model.freeze_anchor()
    params = model.configure_for_tta()
    opt = torch.optim.SGD(params, lr=1e-2)
    x_masked, _, _ = masker.apply(x_burst)
    loss = predictive_entropy(model(x_masked).logits).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
    print(f"masked-input entropy loss={float(loss.detach()):.4f}  dw_bn grad_norm={grad_norm:.6f}")
    assert grad_norm > 0.0, "gradients must reach dw_bn affine params through the masked reconstruction"
    opt.step()

    print("Self-test passed.")
