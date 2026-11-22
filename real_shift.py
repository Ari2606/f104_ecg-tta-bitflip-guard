"""
real_shift.py  --  this project: evidence that does NOT depend on our own simulator.

Two kinds of real distribution shift, both on public PhysioNet data:

1. REAL NOISE on the held-out DS2 patients (`RealNoiseSimulator`)
   The MIT-BIH Noise Stress Test Database (NSTDB, Moody, Muldrow & Mark, 1984) provides
   three half-hour recordings of REAL ambulatory noise: `bw` (baseline wander), `em`
   (electrode motion artifact) and `ma` (muscle artifact). We add windows of these
   recordings to the DS2 beats at a controlled SNR, in the NSTDB layout (initial clean
   period, then alternating noisy / clean segments), so the noise has real temporal
   structure (bursts, drifts) that no parametric simulator of ours reproduces. We use the
   noise recordings with DS2 patients, NOT the NSTDB "118e*"/"119e*" records: those are
   records 118/119, which belong to DS1 (training) in the de Chazal split.
   SNR convention (explicit, because NSTDB leaves it to the user): beats are z-scored per
   record (unit signal variance, the model's own input scale), noise is scaled to a unit
   RMS reference over the whole noise record, then multiplied by 10^(-SNR_dB/20).
   Injection is post-preprocessing (on the model input), i.e. noise that the 0.5-40 Hz
   offline band-pass of the pipeline did NOT remove -- a deliberate stress, stated as such.

2. CROSS-DATABASE shift (`load_external`)
   Different hospital, device, population, sampling rate: the St Petersburg INCART
   12-lead database (75 records, 257 Hz, lead II used as the MLII proxy) and the MIT-BIH
   Supraventricular Arrhythmia Database (SVDB, 78 records, 128 Hz, first lead). Signals
   are resampled to 360 Hz, then follow the identical pipeline as MIT-BIH (0.5-40 Hz
   band-pass, record z-score, 96/160-sample window around the R-peak, RR features at
   360 Hz). Only beats with a mapped AAMI symbol are kept (same rule as dataset.py).
   Lead mismatch (lead II vs MLII; SVDB's lead is not guaranteed to be MLII-like) is a
   limitation to state in the paper, not hide.

Both produce objects that plug into `streaming_eval.make_beat_stream` unchanged
(`sim=` for the noise simulator; `ds=` for the external datasets).

Colab usage:
    from real_shift import download_nstdb, download_external
    download_nstdb("/content/drive/MyDrive/ttaguard_data/data/nstdb")
    download_external("incartdb", "/content/drive/MyDrive/ttaguard_data/data/incartdb")
Self-test:  python real_shift.py   (synthetic noise + synthetic wfdb records, no download)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly

from dataset import (AAMI_MAP, CLASS_NAMES, NUM_CLASSES, ECGDataConfig, _bandpass,
                     _normalize_record, _rr_interval_features)

FS_MODEL = 360.0
NOISE_KINDS = ("bw", "em", "ma")


# --------------------------------------------------------------------------- #
# 1. Real noise (NSTDB)
# --------------------------------------------------------------------------- #
def download_nstdb(data_dir: str) -> None:
    """Fetch the three noise recordings (bw, em, ma) only."""
    import wfdb
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("nstdb", dl_dir=data_dir, records=list(NOISE_KINDS))


def _resample(sig: np.ndarray, fs_in: float, fs_out: float = FS_MODEL) -> np.ndarray:
    """[T, C] -> [T', C] polyphase resample (exact rational ratio of the integer rates)."""
    if abs(fs_in - fs_out) < 1e-6:
        return sig
    fr = Fraction(int(round(fs_out)), int(round(fs_in))).limit_denominator(1000)
    return resample_poly(sig, fr.numerator, fr.denominator, axis=0)


class NoiseBank:
    """The three real noise recordings, resampled to 360 Hz and normalised to unit RMS
    (mean removed per channel) so that `snr_db` has a single, explicit meaning."""

    def __init__(self, arrays: Dict[str, np.ndarray]):
        assert set(arrays) >= set(NOISE_KINDS), f"need {NOISE_KINDS}, got {sorted(arrays)}"
        self.arr: Dict[str, np.ndarray] = {}
        for k in NOISE_KINDS:
            a = arrays[k].astype(np.float64)                       # [T, C]
            a = a - a.mean(axis=0, keepdims=True)
            a = a / (np.sqrt((a ** 2).mean(axis=0, keepdims=True)) + 1e-12)
            self.arr[k] = a.astype(np.float32)

    @classmethod
    def from_physionet(cls, data_dir: str) -> "NoiseBank":
        import wfdb
        out = {}
        for k in NOISE_KINDS:
            rec = wfdb.rdrecord(str(Path(data_dir) / k))
            sig = np.nan_to_num(rec.p_signal.astype(np.float64))
            out[k] = _resample(sig, float(rec.fs))
        return cls(out)

    @classmethod
    def synthetic(cls, seconds: float = 300.0, seed: int = 0) -> "NoiseBank":
        """Stand-in with the right qualitative character, for the self-test ONLY."""
        rng = np.random.default_rng(seed)
        n = int(seconds * FS_MODEL)
        t = np.arange(n) / FS_MODEL
        bw = np.stack([np.sin(2 * np.pi * 0.15 * t + p) + 0.5 * np.sin(2 * np.pi * 0.4 * t + 2 * p)
                       for p in (0.0, 1.0)], axis=1)
        em = np.cumsum(rng.standard_normal((n, 2)), axis=0)
        em = em - np.convolve(em[:, 0], np.ones(1000) / 1000, "same")[:, None]
        ma = rng.standard_normal((n, 2)) * (1 + 3 * (np.sin(2 * np.pi * 0.05 * t) > 0.5))[:, None]
        return cls({"bw": bw, "em": em, "ma": ma})

    def half(self, which: str) -> "NoiseBank":
        """'first' / 'second' half of every noise recording (re-normalised). Use the first half
        for TRAINING-time noise augmentation and the second half for evaluation, so a model
        trained on real noise is never tested on the same stretch of the same recording."""
        assert which in ("first", "second"), which
        out = {}
        for k, a in self.arr.items():
            m = len(a) // 2
            out[k] = a[:m] if which == "first" else a[m:]
        return NoiseBank(out)

    def __len__(self) -> int:
        return min(len(a) for a in self.arr.values())


@dataclass
class RealNoiseConfig:
    snr_db: float = 6.0
    kinds: Tuple[str, ...] = NOISE_KINDS      # a single kind, or all three (cycled per segment)
    clean_start_s: float = 300.0              # initial noise-free period (NSTDB layout)
    segment_s: float = 120.0                  # alternating noisy / clean segment length
    always_on: bool = False                   # True: noise throughout (after clean_start_s)
    seed: int = 0
    fs: float = FS_MODEL
    r_index: int = 96


class RealNoiseSimulator:
    """
    Duck-typed replacement for `SpaceShiftSimulator` in `make_beat_stream`:
        sim(x[C,L], record_id=, beat_key=, t0_sec=, severity=, return_mask=True)
            -> (x_noisy[C,L], corrupt_mask[L] bool)
    `severity` in [0,1] scales the noise amplitude (1 = the configured SNR); the benchmark
    passes 1.0. Noise is drawn from the REAL recording at the beat's own acquisition time
    (+ a fixed per-record offset that depends on `seed`), so consecutive beats see
    consecutive stretches of the same noise realisation -- temporal correlation is real.
    """

    def __init__(self, bank: NoiseBank, cfg: Optional[RealNoiseConfig] = None):
        self.bank, self.cfg = bank, cfg or RealNoiseConfig()
        for k in self.cfg.kinds:
            assert k in NOISE_KINDS, k
        self._amp = 10.0 ** (-self.cfg.snr_db / 20.0)

    def _offset(self, record_id: int) -> int:
        h = hashlib.sha256(f"{self.cfg.seed}:{record_id}".encode()).digest()
        return int.from_bytes(h[:4], "little") % len(self.bank)

    def noisy_at(self, t_sec: float) -> bool:
        c = self.cfg
        if t_sec < c.clean_start_s:
            return False
        if c.always_on:
            return True
        return int((t_sec - c.clean_start_s) // c.segment_s) % 2 == 0

    def kind_at(self, t_sec: float) -> str:
        c = self.cfg
        seg = int(max(t_sec - c.clean_start_s, 0.0) // (2 * c.segment_s if not c.always_on else c.segment_s))
        return c.kinds[seg % len(c.kinds)]

    def __call__(self, x: torch.Tensor, record_id: int = 0, beat_key: int = 0,
                 t0_sec: Optional[float] = None, severity: float = 1.0, return_mask: bool = False):
        C, L = x.shape
        t = float(t0_sec) if t0_sec is not None else float(beat_key) * 0.8   # fallback: ~75 bpm
        mask = torch.zeros(L, dtype=torch.bool)
        if severity > 0 and self.noisy_at(t):
            arr = self.bank.arr[self.kind_at(t)]                     # [T, 2]
            n = len(self.bank)
            start = int(round(t * self.cfg.fs)) + self._offset(record_id)
            idx = (start + np.arange(L)) % n
            seg = arr[idx]                                           # [L, 2]
            noise = torch.from_numpy(seg[:, [c % seg.shape[1] for c in range(C)]].T.copy())  # [C, L]
            x = x + float(severity) * self._amp * noise.to(x.dtype)
            mask[:] = True
        return (x, mask) if return_mask else x


# --------------------------------------------------------------------------- #
# 2. Cross-database datasets (INCART, SVDB)
# --------------------------------------------------------------------------- #
EXTERNAL_DBS: Dict[str, Dict] = {
    # lead_candidates in priority order; first one present in the record is used
    "incartdb": {"lead_candidates": ("II", "MLII", "ii"), "default_fs": 257.0},
    "svdb":     {"lead_candidates": ("ECG1", "MLII", "II"), "default_fs": 128.0},
}


def download_external(db: str, data_dir: str, records: Optional[Sequence[str]] = None) -> None:
    import wfdb
    assert db in EXTERNAL_DBS, f"unknown db {db}; known: {list(EXTERNAL_DBS)}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(db, dl_dir=data_dir, records=list(records) if records else "all")


def list_external_records(db: str, data_dir: str) -> List[str]:
    """Record names present on disk (those with a .hea AND an .atr file)."""
    d = Path(data_dir)
    return sorted(p.stem for p in d.glob("*.hea") if (d / f"{p.stem}.atr").exists())


def _cut_beats(sig360: np.ndarray, samples360: np.ndarray, symbols: Sequence[str],
               cfg: ECGDataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure helper: sig360 [T,C] (already filtered+normalised), beat R-peaks at 360 Hz."""
    T = sig360.shape[0]
    xs, ys, ss = [], [], []
    for s, sym in zip(samples360, symbols):
        cls = AAMI_MAP.get(sym)
        if cls is None:
            continue
        a, b = int(s) - cfg.pre, int(s) + cfg.post
        if a < 0 or b > T:
            continue
        xs.append(sig360[a:b].T.astype(np.float32))
        ys.append(cls)
        ss.append(int(s))
    if not xs:
        return (np.zeros((0, sig360.shape[1], cfg.window_len), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return np.stack(xs), np.asarray(ys, np.int64), np.asarray(ss, np.int64)


def load_external_record(db: str, record: str, data_dir: str, cfg: ECGDataConfig,
                        cache_dir: Optional[str] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One INCART/SVDB record -> (x [N,C,L], y [N], samples360 [N], rr [N,3])."""
    tag = (f"{db}_{record}_c{cfg.n_leads}_p{cfg.pre}_q{cfg.post}_{cfg.norm}_bp{int(cfg.bandpass)}"
           f"_{cfg.band[0]}-{cfg.band[1]}_rrL{cfg.n_rr_features if cfg.use_rr_features else 0}")
    cache_file = Path(cache_dir) / f"{tag}.npz" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        d = np.load(cache_file)
        return d["x"], d["y"], d["samples"], d["rr"]

    import wfdb
    spec = EXTERNAL_DBS[db]
    path = str(Path(data_dir) / record)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    names = [n.strip() for n in rec.sig_name]
    lead = next((names.index(c) for c in spec["lead_candidates"] if c in names), 0)
    idx = [lead]
    if cfg.n_leads == 2:
        idx.append(next(i for i in range(len(names)) if i != lead))
    sig = np.nan_to_num(rec.p_signal[:, idx].astype(np.float64))
    fs = float(rec.fs)
    sig = _resample(sig, fs, FS_MODEL)                                   # -> 360 Hz
    if cfg.bandpass:
        sig = _bandpass(sig, FS_MODEL, *cfg.band)
    if cfg.norm == "record":
        sig = _normalize_record(sig)
    samples360 = np.round(np.asarray(ann.sample, dtype=np.float64) * (FS_MODEL / fs)).astype(np.int64)
    x, y, s = _cut_beats(sig, samples360, ann.symbol, cfg)
    rr = (_rr_interval_features(s, FS_MODEL, n_features=cfg.n_rr_features) if (cfg.use_rr_features and len(s) > 0)
          else np.zeros((len(s), 0), np.float32))
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y, samples=s, rr=rr)
    return x, y, s, rr


class ExternalBeatDataset:
    """MITBIHDataset duck-type (x, y, rr, record_ids, sample_idx) for a non-MIT-BIH DB."""

    def __init__(self, db: str, data_dir: str, cfg: ECGDataConfig,
                 records: Optional[Sequence[str]] = None, cache_dir: Optional[str] = None,
                 max_beats_per_record: Optional[int] = None):
        self.db, self.cfg = db, cfg
        recs = list(records) if records else list_external_records(db, data_dir)
        if not recs:
            raise FileNotFoundError(f"no {db} records (.hea+.atr) under {data_dir}; "
                                    f"run real_shift.download_external('{db}', '{data_dir}')")
        xs, ys, sidx, rrs, rids, names = [], [], [], [], [], []
        for k, r in enumerate(recs):
            x, y, s, rr = load_external_record(db, r, data_dir, cfg, cache_dir)
            if max_beats_per_record is not None:
                x, y, s, rr = (x[:max_beats_per_record], y[:max_beats_per_record],
                               s[:max_beats_per_record], rr[:max_beats_per_record])
            if len(y) == 0:
                continue
            xs.append(x); ys.append(y); sidx.append(s); rrs.append(rr)
            rids.append(np.full(len(y), k, np.int64)); names.append(r)
        self.record_names = names                                         # id k -> record name
        self.records = list(range(len(names)))
        self.x = torch.from_numpy(np.concatenate(xs))
        self.y = torch.from_numpy(np.concatenate(ys))
        self.record_ids = torch.from_numpy(np.concatenate(rids))
        self.sample_idx = torch.from_numpy(np.concatenate(sidx))
        self.rr = torch.from_numpy(np.concatenate(rrs))

    def __len__(self) -> int:
        return self.x.shape[0]

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(self.y, minlength=NUM_CLASSES)


def load_external(db: str, data_dir: str, data_cfg: ECGDataConfig, cache_dir: Optional[str] = None,
                  records: Optional[Sequence[str]] = None, max_beats_per_record: Optional[int] = None
                  ) -> ExternalBeatDataset:
    """Entry point used by run_benchmark.py (and registrable in eval_cross_domain.EXTRA_DATASETS)."""
    return ExternalBeatDataset(db, data_dir, data_cfg, records, cache_dir, max_beats_per_record)


# --------------------------------------------------------------------------- #
# Self-test: synthetic noise + synthetic WFDB records (no download)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import wfdb

    print("[real_shift] self-test (synthetic stand-ins; real data needs download_*)")
    # ---- noise simulator ---------------------------------------------------- #
    bank = NoiseBank.synthetic(seconds=120)
    for k in NOISE_KINDS:
        rms = np.sqrt((bank.arr[k] ** 2).mean(axis=0))
        assert np.allclose(rms, 1.0, atol=1e-3), (k, rms)
    sim = RealNoiseSimulator(bank, RealNoiseConfig(snr_db=6.0, kinds=("em",), clean_start_s=10,
                                                   segment_s=5, seed=1))
    x = torch.zeros(1, 256)
    x_clean, m_clean = sim(x, record_id=7, beat_key=0, t0_sec=2.0, severity=1.0, return_mask=True)
    assert torch.equal(x_clean, x) and not m_clean.any(), "no noise before clean_start_s"
    x_n, m_n = sim(x, record_id=7, beat_key=0, t0_sec=11.0, severity=1.0, return_mask=True)
    assert m_n.all() and x_n.abs().sum() > 0
    x_g, m_g = sim(x, record_id=7, beat_key=0, t0_sec=16.0, severity=1.0, return_mask=True)
    assert not m_g.any(), "clean segment after the first noisy one"
    # SNR check: noise std over many windows ~= 10^(-SNR/20)
    stds = []
    for t in np.arange(10.0, 14.9, 0.05):
        xn, _ = sim(torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=float(t), severity=1.0, return_mask=True)
        stds.append(xn.numpy().ravel())
    got = float(np.concatenate(stds).std())
    want = 10 ** (-6.0 / 20)
    print(f"  noise std at SNR 6 dB: got {got:.3f}, nominal {want:.3f} (window-local vs record RMS)")
    assert 0.3 * want < got < 3.0 * want
    xs, _ = sim(torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=12.0, severity=0.5, return_mask=True)
    xf, _ = sim(torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=12.0, severity=1.0, return_mask=True)
    assert torch.allclose(xs * 2, xf, atol=1e-5), "severity scales amplitude linearly"
    a, _ = sim(torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=12.0, severity=1.0, return_mask=True)
    b, _ = sim(torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=12.0, severity=1.0, return_mask=True)
    assert torch.equal(a, b), "deterministic given (seed, record, time)"
    c, _ = RealNoiseSimulator(bank, RealNoiseConfig(snr_db=6, kinds=("em",), clean_start_s=10,
                                                    segment_s=5, seed=2))(
        torch.zeros(1, 256), record_id=3, beat_key=0, t0_sec=12.0, severity=1.0, return_mask=True)
    assert not torch.equal(a, c[0] if isinstance(c, tuple) else c), "seed changes the draw"
    # plugs into make_beat_stream with the (record, time) metadata a real stream has
    from baselines_tta import _SyntheticStreamDataset
    from streaming_eval import make_beat_stream
    ds = _SyntheticStreamDataset(n_records=2, beats_per_record=60, c=1, l=256, seed=0)
    sim_ds = RealNoiseSimulator(bank, RealNoiseConfig(snr_db=0, clean_start_s=0, segment_s=1000, always_on=True))
    sim_ds.cfg.fs, sim_ds.cfg.r_index = 360.0, 96
    got = list(make_beat_stream(ds, n_beats=20, sim=sim_ds, severity=1.0))
    assert len(got) == 20 and all(g[2]["corrupt_mask"].all() for g in got)
    print("  RealNoiseSimulator: layout, SNR scaling, determinism, stream plug-in OK")

    # ---- external DB loader: write real WFDB files and read them back -------- #
    with tempfile.TemporaryDirectory() as td:
        fs_in, secs = 257, 40
        t = np.arange(fs_in * secs) / fs_in
        beat_t = np.arange(1.0, secs - 1, 0.8)
        sig = np.zeros((len(t), 3))
        for bt in beat_t:
            sig[:, 1] += np.exp(-((t - bt) ** 2) / (2 * 0.012 ** 2))      # "lead II" with R spikes
        sig[:, 0] = 0.2 * np.sin(2 * np.pi * 0.3 * t)
        samples = np.round(beat_t * fs_in).astype(int)
        symbols = ["N"] * len(samples)
        symbols[3], symbols[5] = "V", "A"
        symbols[7] = "+"                                                    # non-beat -> dropped
        wfdb.wrsamp("I01", fs=fs_in, units=["mV"] * 3, sig_name=["I", "II", "III"],
                    p_signal=sig, fmt=["16"] * 3, write_dir=td)
        wfdb.wrann("I01", "atr", samples, symbols, write_dir=td)
        ds_ext = load_external("incartdb", td, ECGDataConfig(n_leads=1, use_rr_features=True),
                               cache_dir=td + "/cache")
        n_expect = len(samples) - 1         # minus the non-beat '+' annotation (all beats fit in the record)
        assert len(ds_ext) == n_expect, (len(ds_ext), n_expect)
        assert ds_ext.x.shape[1:] == (1, 256) and ds_ext.rr.shape[1] == 3
        cc = ds_ext.class_counts().tolist()
        assert cc[2] == 1 and cc[1] == 1, cc                                # one V, one S(A)
        r_pos = ds_ext.x[:, 0, :].abs().argmax(-1).float().mean().item()
        assert abs(r_pos - 96) <= 6, f"R-peak should sit near index 96 after 257->360 Hz resample, got {r_pos}"
        d2 = load_external("incartdb", td, ECGDataConfig(n_leads=1, use_rr_features=True),
                           cache_dir=td + "/cache")                         # cache path
        assert torch.equal(ds_ext.x, d2.x)
        pre = ds_ext.rr[:, 0].mean().item()
        assert abs((pre * 0.3 + 0.8) - 0.8) < 0.05, f"RR feature should decode to ~0.8 s, got {pre*0.3+0.8:.3f}"
    print(f"  ExternalBeatDataset: wfdb read, 257->360 Hz resample, R-peak at ~96 ({r_pos:.1f}), "
          f"RR~0.8 s, AAMI map, cache OK")
    print("Self-test passed.")
