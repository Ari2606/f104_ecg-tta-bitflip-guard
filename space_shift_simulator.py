"""
space_shift_simulator.py  --  this project, Stage 2 (Space Shift Simulator)

Mathematical simulation of the two dominant domain shifts an astronaut's
single-beat ECG window undergoes on a deep-space mission:

(1) MICROGRAVITY (cephalad fluid shift)
      * cardiac-vector axis tilt : rotation R(theta) of the 2-D lead-space vector
      * QRS amplitude attenuation, T-wave gain change, ST offset, QT shortening
(2) SLOW PHYSIOLOGICAL / MOTION ARTIFACT
      * baseline wander: respiration-band + sub-respiratory drift, phase-continuous in
        absolute record time (a beat sees only a slice of a slow sinusoid)
(3) COSMIC RADIATION ON THE ANALOG FRONT-END
      * total-ionizing-dose : gain drift + raised noise floor
      * single-event transients (SET): short saturated / high-amplitude bursts
      * ADC degradation: reduced effective resolution (quantization)
      * single-event upsets (SEU): random bit flips in ADC output codes
        (an MSB flip = a huge spike, an LSB flip = invisible)

Everything is governed by a scalar mission SEVERITY s in [0, 1] (s = 0 -> identity).

Determinism
-----------
* Subject-level parameters (tilt, gains, wander components...) are drawn from
  RNG(seed, record_id): a given astronaut/record has ONE persistent shift, as
  physiology would. Beat-level jitter/noise is drawn from RNG(seed, beat_key), so every
  method sees bit-identical corrupted data (fair TTA comparisons).

Ground truth for later stages
-----------------------------
`return_mask=True` yields a boolean [L] mask of samples hit by radiation events
(SET / significant SEU error). Stage 5 uses it to validate that the TS-SQI mask
really excludes corrupted regions.

Shape legend: C = leads (1 or 2), L = window length (256), B = batch.

Modelling caveat (state this in the paper): the 2-lead rotation treats the two
recorded leads as projections of one planar cardiac vector. MLII and V-leads are not
orthogonal frontal-plane leads, so R(theta) is a controlled *proxy* for axis change, not a
biophysically exact torso model. With C = 1 a pseudo-orthogonal channel (normalized
time-derivative) is used instead. All parameter ranges below are configurable
assumptions and should be calibrated against published spaceflight ECG data
before being quoted as physiological facts.

Parameter grounding (checked, Sept 2026 -- report this honestly in the paper's
limitations section; do not claim more precision than the literature has):

* RADIATION (seu_rate, set_rate, tid_gain_drift, noise_sigma) -- the ORDER OF
  MAGNITUDE is checkable against real operational data. Empirical single-event-
  upset rates measured on the JASON-2 satellite over 3 years of LEO operation
  are ~4.76e-7 bit-errors/bit/day, and the wider SEU literature for LEO
  spot-checks in the 1e-7 to 1e-6 /bit/day range (see e.g. arXiv:2407.11853,
  "A Case for Application-Aware Space Radiation Tolerance in Orbital
  Computing"). Converted to a single ~0.71 s / 256-sample beat window, that
  background rate is many orders of magnitude smaller than 1 event/beat.
  This simulator's seu_rate=2e-3 (per sample, per lead) AT SEVERITY=1 is
  THEREFORE NOT a "typical mission" rate -- it is a deliberate worst-case
  stress bound, chosen so the safety mechanism (TS-SQI masking, elastic
  anchor reset) can be evaluated under conditions far harsher than ordinary
  LEO background radiation (e.g. a solar particle event, deep-space transit
  outside Earth's magnetosphere, or an aged/degraded shielding scenario).
  State this distinction explicitly if asked: severity=1 = engineered stress
  test, not a forecast of typical mission telemetry corruption.

* MICROGRAVITY MORPHOLOGY (axis_tilt_deg, qrs_gain, t_gain, st_shift,
  qt_scale) -- UNLIKE the radiation numbers, these do NOT have a citable
  quantitative source, and a literature check found none exists publicly at
  this precision. What IS documented in human spaceflight ECG studies:
  QT/QTc intervals are slightly prolonged in some crewmembers after
  long-duration (but not short-duration) missions (D'Aunno et al. 2003,
  Am J Cardiol); ST-segment and T-wave alterations were observed during
  maximal exertion on Skylab; and QRS *duration* (not amplitude) is
  reported as essentially unchanged. No systematic study has published
  degree-level cardiac-axis-shift or percentage QRS/T-amplitude-change
  figures for spaceflight -- the field itself has not produced that dataset.
  (See "Cardiac rhythm problems during spaceflight" for a summary, and the
  npj Microgravity 2024 paper already cited in this project's related work,
  which names this same Earth-to-space data gap as an open problem.)
  The baseline-wander frequency band (resp_hz, drift_hz) is a partial
  exception: respiration rate during spaceflight HAS been measured
  quantitatively, though only under a specific hypoxic-challenge test
  protocol (PIo2=128 mmHg), not as a general resting-baseline figure --
  respiration rate rose from 8.6+-2.5 to 15.2+-4.4 breaths/min in-flight
  (0.14 Hz -> 0.25 Hz), i.e. it roughly doubled (Am J Aerosp Med/Hum
  Perform., Vol 88(6), p.527). This falls inside this simulator's
  resp_hz=(0.15, 0.40) Hz range, so that ONE parameter has a real,
  if narrow and condition-specific, anchor point -- state this distinction
  if asked (partial grounding for wander; no grounding for axis/QRS/T-gain).

  Because of this, axis_tilt_deg/qrs_gain/t_gain/st_shift/qt_scale are a
  DIRECTIONALLY-motivated but NUMERICALLY-unvalidated stress envelope: they
  encode the known qualitative direction of each effect (axis/morphology
  change from cephalad fluid shift, mild QT change) at a magnitude chosen to
  be a plausible, non-trivial test of TTA robustness -- not a claim that real
  astronaut ECGs shift by exactly this many degrees or this much gain. If a
  reviewer asks: say this directly. It is the same "space ECG ground truth
  does not exist yet" gap the closest related work (npj Microgravity 2024)
  also names as unsolved -- this project's contribution is a TTA/safety
  mechanism that is robust to a *range* of plausible shift magnitudes
  (that is what the severity sweep across mild/moderate/severe demonstrates),
  not a claim of a biophysically validated space-ECG simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SpaceShiftConfig:
    fs: float = 360.0
    r_index: int = 96                     # R-peak position inside window (== dataset `pre`)
    seed: int = 0

    # ---- switches (ablations) ---------------------------------------------- #
    enable_microgravity: bool = True      # master switch for (1)
    enable_axis_tilt: bool = True
    enable_morphology: bool = True        # QRS/T gain, ST offset, QT warp
    enable_wander: bool = True
    enable_radiation: bool = True

    # ---- (1) microgravity, ranges are the values reached at severity = 1 ----- #
    fixed_tilt_deg: Optional[float] = None            # signed override (tilt sweeps)
    axis_tilt_deg: Tuple[float, float] = (20.0, 45.0) # |theta|, sign random per subject
    beat_jitter_deg: float = 2.0                      # beat-to-beat tilt std
    qrs_gain: Tuple[float, float] = (0.70, 0.90)      # multiplicative QRS amplitude
    t_gain: Tuple[float, float] = (0.50, 1.40)        # multiplicative T amplitude
    st_shift: Tuple[float, float] = (-0.15, 0.15)     # additive ST offset (sigma units)
    qt_scale: Tuple[float, float] = (0.90, 1.00)      # <1 = shortened QT

    # ---- (2) baseline wander ------------------------------------------------- #
    wander_amp: Tuple[float, float] = (0.4, 1.0)      # total amplitude (sigma units)
    resp_hz: Tuple[float, float] = (0.15, 0.40)
    drift_hz: Tuple[float, float] = (0.03, 0.12)
    wander_weights: Tuple[float, float, float] = (0.45, 0.25, 0.30)  # resp1, resp2, drift

    # ---- (3) radiation -------------------------------------------------------- #
    adc_full_scale: float = 12.0          # +-FS in z-scored units
    adc_bits_clean: int = 12              # effective bits at s -> 0
    adc_bits_min: int = 7                 # effective bits at s = 1
    noise_sigma: float = 0.10             # extra white-noise floor at s = 1
    tid_gain_drift: float = 0.10          # +-fractional gain drift at s = 1
    seu_rate: float = 2e-3                # per-sample, per-lead bit-flip prob at s = 1
    set_rate: float = 1.5                 # Poisson mean #bursts / beat at s = 1
    set_len: Tuple[int, int] = (1, 8)     # burst length (samples)
    set_amp: Tuple[float, float] = (2.0, 6.0)
    mask_thresh: float = 0.5              # |error| (sigma) that flags a sample as corrupted


def make_config(name: str, **kw) -> Tuple[SpaceShiftConfig, float]:
    """Named mission profiles -> (config, severity)."""
    presets = {"earth": 0.0, "mild": 0.33, "moderate": 0.66, "severe": 1.0}
    if name not in presets:
        raise KeyError(f"unknown preset {name}; options: {list(presets)}")
    return SpaceShiftConfig(**kw), presets[name]


def make_quiet_leo_radiation_config(**kw) -> Tuple[SpaceShiftConfig, float]:
    """
    Radiation-ONLY sanity-check config, calibrated to a REAL measured LEO
    background single-event-upset rate -- not a point on the mild/moderate/
    severe stress sweep used for the main results table.

    Source: JASON-2 satellite, ~3 years of LEO operation, ~4.76e-7
    bit-errors/bit/day (wider literature range 1e-7 to 1e-6/bit/day).
    See arXiv:2407.11853 ("A Case for Application-Aware Space Radiation
    Tolerance in Orbital Computing"). Converted here to a per-SAMPLE
    probability (>=1 bit flips among `adc_bits_clean` bits, given the
    sample's dwell time 1/fs as a fraction of one day):

        p_bit  = seu_per_bit_per_day * (1/fs) / 86400
        p_samp = 1 - (1 - p_bit) ** adc_bits_clean

    At fs=360 Hz, adc_bits_clean=12 this works out to roughly 1.8e-13 per
    sample -- i.e. expected corrupted samples across an entire 10,000-beat
    streaming run (~2.56M samples) is on the order of 1e-6 (essentially
    zero). That is the point of this preset: it demonstrates the radiation
    branch of the simulator correctly reduces to a no-op at real quiet-LEO
    background levels, which is what justifies calling severity=1.0 in the
    main sweep a deliberate worst-case stress test rather than an arbitrary,
    uncalibrated number. Run this ALONGSIDE the main mild/moderate/severe
    sweep as a labelled sanity-check row/footnote, not merged into it.

    microgravity and wander are disabled here (enable_microgravity=False,
    enable_wander=False) to isolate the radiation effect alone; set_rate,
    noise_sigma and tid_gain_drift are zeroed for the same reason -- only
    seu_rate is backed by the cited figure, so only seu_rate is exercised.
    """
    fs = kw.pop("fs", 360.0)
    bits = kw.pop("adc_bits_clean", 12)
    seu_per_bit_per_day = 4.76e-7                      # JASON-2, arXiv:2407.11853
    day_seconds, sample_seconds = 86400.0, 1.0 / fs
    p_per_bit_per_sample = seu_per_bit_per_day * (sample_seconds / day_seconds)
    p_per_sample = 1.0 - (1.0 - p_per_bit_per_sample) ** bits
    cfg = SpaceShiftConfig(
        fs=fs, adc_bits_clean=bits,
        enable_microgravity=False, enable_wander=False, enable_radiation=True,
        seu_rate=p_per_sample, set_rate=0.0, noise_sigma=0.0, tid_gain_drift=0.0,
        **kw,
    )
    return cfg, 1.0


# --------------------------------------------------------------------------- #
# Subject-level (persistent) shift parameters
# --------------------------------------------------------------------------- #
@dataclass
class SubjectShift:
    tilt_rad: float
    qrs_gain: float
    t_gain: float
    st_shift: float
    qt_scale: float
    wander_amp: float
    wander_hz: torch.Tensor        # [3]     float64
    wander_phase: torch.Tensor     # [3, 2]  float64 (per lead)
    lead_scale: torch.Tensor       # [2]     float64
    tid_gain: float


def _draw_subject(cfg: SpaceShiftConfig, record_id: int) -> SubjectShift:
    rng = np.random.default_rng([cfg.seed, int(record_id) + 12345])
    u = lambda r: float(rng.uniform(*r))
    if cfg.fixed_tilt_deg is not None:
        tilt = math.radians(cfg.fixed_tilt_deg)
    else:
        tilt = math.radians(u(cfg.axis_tilt_deg)) * float(rng.choice([-1.0, 1.0]))
    hz = np.array([u(cfg.resp_hz), u(cfg.resp_hz), u(cfg.drift_hz)])
    return SubjectShift(
        tilt_rad=tilt,
        qrs_gain=u(cfg.qrs_gain), t_gain=u(cfg.t_gain),
        st_shift=u(cfg.st_shift), qt_scale=u(cfg.qt_scale),
        wander_amp=u(cfg.wander_amp),
        wander_hz=torch.from_numpy(hz),
        wander_phase=torch.from_numpy(rng.uniform(0, 2 * math.pi, size=(3, 2))),
        lead_scale=torch.from_numpy(np.array([1.0, rng.uniform(0.6, 1.4)])),
        tid_gain=float(rng.uniform(-1.0, 1.0)) * cfg.tid_gain_drift,
    )


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _linear_resample(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: [C, L], pos: [L] fractional source indices -> [C, L] (linear interpolation)."""
    L = x.shape[-1]
    p = pos.clamp(0, L - 1)
    i0 = p.floor().long()
    i1 = (i0 + 1).clamp(max=L - 1)
    frac = (p - i0.float())                              # [L]
    return x[:, i0] * (1 - frac) + x[:, i1] * frac       # [C, L]


def _rotate(x: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Rotate the cardiac vector by `theta` (rad).  x: [C, L] -> [C, L].
      C = 2 : [y0; y1] = R(theta) [x0; x1]
      C = 1 : y = cos(theta) x + sin(theta) q, q = normalized derivative (pseudo-orthogonal)
    """
    c, s = math.cos(theta), math.sin(theta)
    if x.shape[0] == 2:
        return torch.stack([c * x[0] - s * x[1], s * x[0] + c * x[1]], dim=0)
    q = torch.gradient(x, dim=-1)[0]                                     # [1, L]
    q = q * (x.std(dim=-1, keepdim=True) / (q.std(dim=-1, keepdim=True) + 1e-8))
    return c * x + s * q


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class SpaceShiftSimulator:
    """
    Callable: simulator(x [C, L], record_id=, beat_key=, t0_sec=, severity=) -> [C, L]
    (and optionally a bool [L] radiation-corruption mask).
    """

    def __init__(self, cfg: Optional[SpaceShiftConfig] = None):
        self.cfg = cfg or SpaceShiftConfig()
        self._subjects: Dict[int, SubjectShift] = {}
        self._win: Dict[int, Dict[str, torch.Tensor]] = {}

    # -- caches ------------------------------------------------------------- #
    def subject(self, record_id: int) -> SubjectShift:
        rid = int(record_id)
        if rid not in self._subjects:
            self._subjects[rid] = _draw_subject(self.cfg, rid)
        return self._subjects[rid]

    def _windows(self, L: int) -> Dict[str, torch.Tensor]:
        if L not in self._win:
            fs, r = self.cfg.fs, self.cfg.r_index
            t = torch.arange(L, dtype=torch.float32)
            g = lambda c, s: torch.exp(-0.5 * ((t - c) / s) ** 2)
            self._win[L] = {
                "t": t,
                "wq": g(r + 0.01 * fs, 0.035 * fs),     # QRS complex
                "wst": g(r + 0.13 * fs, 0.035 * fs),    # ST segment
                "wt": g(r + 0.27 * fs, 0.070 * fs),     # T wave
            }
        return self._win[L]

    # -- main --------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, x: torch.Tensor, *, record_id: int = 0, beat_key: int = 0,
                 t0_sec: Optional[float] = None, severity: float = 1.0,
                 return_mask: bool = False):
        cfg = self.cfg
        if x.dim() != 2 or x.shape[0] not in (1, 2):
            raise ValueError(f"expected [C in {{1,2}}, L], got {tuple(x.shape)}")
        C, L = x.shape
        s = float(min(max(severity, 0.0), 1.0))
        y = x.detach().to(torch.float32).clone()                         # [C, L]
        mask = torch.zeros(L, dtype=torch.bool)
        if s == 0.0:
            return (y, mask) if return_mask else y

        subj = self.subject(record_id)
        gen = torch.Generator().manual_seed(
            (cfg.seed * 1_000_003 + int(beat_key) * 7919 + 17) & 0x7FFFFFFF)
        w = self._windows(L)
        randn = lambda *sh: torch.randn(*sh, generator=gen)
        randint = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=gen))
        rand = lambda: float(torch.rand((), generator=gen))
        if t0_sec is None:                                               # i.i.d. mode
            t0_sec = rand() * 600.0

        # ---------------- (1) microgravity ---------------------------------- #
        if cfg.enable_microgravity:
            if cfg.enable_morphology:
                qt = 1.0 + s * (subj.qt_scale - 1.0)
                knee = cfg.r_index + 0.10 * cfg.fs
                pos = torch.where(w["t"] <= knee, w["t"], knee + (w["t"] - knee) / qt)
                y = _linear_resample(y, pos)                             # QT warp     [C, L]
                gq = 1.0 + s * (subj.qrs_gain - 1.0)
                gt = 1.0 + s * (subj.t_gain - 1.0)
                gain = (1.0 + (gq - 1.0) * w["wq"] + (gt - 1.0) * w["wt"]).clamp_min(0.05)  # [L]
                y = y * gain + (s * subj.st_shift) * w["wst"]            # [C, L]
            if cfg.enable_axis_tilt:
                theta = s * subj.tilt_rad + math.radians(cfg.beat_jitter_deg) * s * float(randn(()))
                y = _rotate(y, theta)                                    # [C, L]

        # ---------------- (2) baseline wander -------------------------------- #
        if cfg.enable_wander:
            t = t0_sec + torch.arange(L, dtype=torch.float64) / cfg.fs              # [L]
            wgt = torch.tensor(cfg.wander_weights, dtype=torch.float64)              # [3]
            arg = 2 * math.pi * subj.wander_hz[:, None] * t[None, :]                 # [3, L]
            comp = torch.sin(arg[:, None, :] + subj.wander_phase[:, :C, None])       # [3, C, L]
            wv = (wgt[:, None, None] * comp).sum(0) * subj.lead_scale[:C, None]      # [C, L]
            y = y + (s * subj.wander_amp) * wv.to(torch.float32)

        # ---------------- (3) cosmic radiation -------------------------------- #
        if cfg.enable_radiation:
            y = y * (1.0 + s * subj.tid_gain)                                        # TID gain drift
            ref = y.clone()                                                          # pre-event reference
            y = y + (s * cfg.noise_sigma) * randn(C, L)                              # noise floor
            n_set = int(torch.poisson(torch.tensor(cfg.set_rate * s), generator=gen))
            for _ in range(n_set):                                                   # SET bursts
                ln = randint(cfg.set_len[0], cfg.set_len[1] + 1)
                st = randint(0, max(L - ln, 0) + 1)
                ch = randint(0, C)
                amp = _lerp(*cfg.set_amp, rand()) * (1.0 if rand() < 0.5 else -1.0)
                y[ch, st:st + ln] += amp + 0.5 * randn(min(ln, L - st))
            bits = int(round(_lerp(cfg.adc_bits_clean, cfg.adc_bits_min, s)))
            y = self._adc(y, bits, cfg.seu_rate * s, gen)                            # quantize + SEU
            mask = ((y - ref).abs() > cfg.mask_thresh).any(dim=0)                    # [L] bool

        return (y, mask) if return_mask else y

    def _adc(self, y: torch.Tensor, bits: int, p_flip: float,
             gen: torch.Generator) -> torch.Tensor:
        """Uniform mid-tread ADC with random output-code bit flips.  y: [C, L]."""
        FS = self.cfg.adc_full_scale
        levels = 2 ** bits
        step = 2.0 * FS / (levels - 1)
        code = torch.round((y.clamp(-FS, FS) + FS) / step).long()                    # [C, L]
        if p_flip > 0:
            flip = torch.rand(y.shape, generator=gen) < p_flip                       # [C, L]
            if flip.any():
                bitpos = torch.randint(0, bits, y.shape, generator=gen)              # [C, L]
                flipped = torch.bitwise_xor(code, torch.bitwise_left_shift(torch.ones_like(bitpos), bitpos))
                code = torch.where(flip, flipped, code)
        return code.to(torch.float32) * step - FS

    def apply_batch(self, x: torch.Tensor, record_ids, beat_keys,
                    t0_sec=None, severity: float = 1.0, return_mask: bool = False):
        """x: [B, C, L] -> [B, C, L] (and mask [B, L])."""
        outs, masks = [], []
        for i in range(x.shape[0]):
            r = self(x[i], record_id=int(record_ids[i]), beat_key=int(beat_keys[i]),
                     t0_sec=None if t0_sec is None else float(t0_sec[i]),
                     severity=severity, return_mask=True)
            outs.append(r[0]); masks.append(r[1])
        y = torch.stack(outs)
        return (y, torch.stack(masks)) if return_mask else y


# --------------------------------------------------------------------------- #
# Dataset wrapper (duck-typed: needs .x [N,C,L], .y, .record_ids, .sample_idx)
# --------------------------------------------------------------------------- #
class SpaceShiftDataset(Dataset):
    """
    Applies the simulator on-the-fly, using each beat's absolute acquisition time so
    baseline wander is phase-continuous along a stream.

    severity : float, or callable(dataset_index) -> float  (mission-phase schedules)
    __getitem__ -> (x_shifted [C, L], y), or with RR features on the base dataset
    (x_shifted, rr [n_rr], y) -- matches MITBIHDataset's tuple ordering, see dataset.py --
    each optionally followed by rad_mask [L] as the final element when return_meta=True.
    RR features, if present, pass through UNCHANGED (self.sim only corrupts the waveform,
    not beat timing -- a real system would re-derive RR from its own, possibly also-
    corrupted R-peak detection, but that's a separate, not-yet-modeled failure mode;
    treating RR as clean here is the optimistic case).
    """

    def __init__(self, base, sim: SpaceShiftSimulator,
                 severity: Union[float, Callable[[int], float]] = 1.0,
                 return_meta: bool = False):
        self.base, self.sim = base, sim
        self.severity, self.return_meta = severity, return_meta
        self.y, self.record_ids, self.sample_idx = base.y, base.record_ids, base.sample_idx
        self.rr = base.rr if (getattr(base, "rr", None) is not None
                              and getattr(base, "cfg", None) is not None
                              and getattr(base.cfg, "use_rr_features", False)) else None

    def __len__(self) -> int:
        return len(self.base)

    def _sev(self, i: int) -> float:
        return float(self.severity(i)) if callable(self.severity) else float(self.severity)

    def __getitem__(self, i: int):
        cfg = self.sim.cfg
        t0 = (int(self.sample_idx[i]) - cfg.r_index) / cfg.fs            # window start (s)
        out = self.sim(self.base.x[i], record_id=int(self.record_ids[i]), beat_key=i,
                       t0_sec=t0, severity=self._sev(i), return_mask=self.return_meta)
        x_shifted, mask = (out[0], out[1]) if self.return_meta else (out, None)
        items = [x_shifted]
        if self.rr is not None:
            items.append(self.rr[i])
        items.append(self.y[i])
        if mask is not None:
            items.append(mask)
        return tuple(items)


def linear_ramp(start_idx: int, n_ramp: int) -> Callable[[int], float]:
    """Severity 0 before `start_idx`, then linear 0 -> 1 over `n_ramp` beats (mission onset)."""
    return lambda i: float(min(max((i - start_idx) / max(n_ramp, 1), 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Smoke test:  python space_shift_simulator.py
# --------------------------------------------------------------------------- #
def _synthetic_beat(L: int = 256, r: int = 96, fs: float = 360.0) -> torch.Tensor:
    t = torch.arange(L, dtype=torch.float32)
    g = lambda c, a, s: a * torch.exp(-0.5 * ((t - c) / (s * fs)) ** 2)
    lead1 = g(r - 0.16 * fs, 0.15, 0.025) - g(r - 0.02 * fs, 0.10, 0.008) + g(r, 1.0, 0.010) \
        - g(r + 0.025 * fs, 0.25, 0.010) + g(r + 0.27 * fs, 0.30, 0.045)
    lead2 = 0.5 * lead1 + g(r, 0.3, 0.012) - g(r + 0.27 * fs, 0.1, 0.045)
    return torch.stack([lead1, lead2]) * 3.0                              # [2, L]


if __name__ == "__main__":
    sim = SpaceShiftSimulator(SpaceShiftConfig())
    x = _synthetic_beat()
    for s in (0.0, 0.33, 0.66, 1.0):
        y, m = sim(x, record_id=100, beat_key=1, t0_sec=12.3, severity=s, return_mask=True)
        assert y.shape == x.shape and torch.isfinite(y).all()
        print(f"severity={s:4.2f}  out={tuple(y.shape)}  rms_err={float((y-x).pow(2).mean().sqrt()):.3f}"
              f"  corrupted_samples={int(m.sum())}")
    y1 = sim(x, record_id=100, beat_key=1, t0_sec=12.3)
    y2 = sim(x, record_id=100, beat_key=1, t0_sec=12.3)
    assert torch.equal(y1, y2), "simulator must be deterministic"
    print("determinism OK")

    # -- quiet-LEO radiation sanity check (real cited background SEU rate) -- #
    quiet_cfg, quiet_sev = make_quiet_leo_radiation_config()
    print(f"[quiet_leo] seu_rate (real, per-sample) = {quiet_cfg.seu_rate:.3e}  "
          f"(main sweep uses seu_rate=2e-3 at severity=1 -> "
          f"{2e-3 / quiet_cfg.seu_rate:,.0f}x higher than real LEO background)")
    quiet_sim = SpaceShiftSimulator(quiet_cfg)
    n_beats, n_corrupted = 2000, 0
    for i in range(n_beats):
        _, m = quiet_sim(x, record_id=100, beat_key=i, t0_sec=12.3, severity=quiet_sev, return_mask=True)
        n_corrupted += int(m.sum())
    print(f"[quiet_leo] {n_corrupted} corrupted samples across {n_beats} synthetic beats "
          f"({n_beats * x.shape[-1]} samples total) -- expect ~0, confirming the radiation "
          f"branch is a no-op at real quiet-LEO background, as documented in the module "
          f"docstring's 'Parameter grounding' section.")
    assert n_corrupted == 0 or n_corrupted / (n_beats * x.shape[-1]) < 1e-5, \
        "quiet-LEO preset should almost never trigger a corrupted sample"
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
        for k in range(2):
            ax[k].plot(x[k], label="Earth"); ax[k].plot(sim(x, record_id=100, beat_key=1,
                       t0_sec=12.3, severity=1.0)[k], label="Space (s=1)")
            ax[k].set_title(f"lead {k+1}")
        ax[0].legend(); plt.tight_layout(); plt.savefig("space_shift_demo.png", dpi=150)
        print("saved space_shift_demo.png")
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)
