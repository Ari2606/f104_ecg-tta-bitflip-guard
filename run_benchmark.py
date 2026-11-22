"""
run_benchmark.py  --  this project: the paper's evaluation driver.

One resumable job grid:   scenarios  x  checkpoints  x  seeds  x  methods
each job -> one compact per-beat log  <out_dir>/<scenario>/<ckpt_tag>/seed<k>/<method>.npz
(analyze_results.py turns these into tables with record-level bootstrap CIs + paired tests).

Scenarios (all on frozen-checkpoint, N=1 chronological streams, predict-then-adapt):
  clean                      DS2 unmodified            -> "do no harm" check
  leo                        DS2 + our parametric space-shift simulator (ramp, cap --leo_cap)
  noise_{bw|em|ma|mixed}_snr{dB}
                             DS2 + REAL NSTDB noise at a stated SNR (real_shift.py)
  incart | svdb              CROSS-DATABASE: other hospital / device / sampling rate
Seeds change: the simulator draw (leo), the noise-recording offsets (noise_*), torch's RNG.
Checkpoints (comma-separated, --checkpoint a.pt,b.pt,...) give independent source models.

Resumable: a job whose .npz exists is skipped, so a Colab disconnect only costs the job in
flight. Shard across notebooks with disjoint --seeds / --methods / --scenarios.

Smoke mode (no --checkpoint or no data): runs the same code path on synthetic data and
labels every output SYNTHETIC; never use those numbers for anything but plumbing checks.

Example (Colab):
  !python run_benchmark.py --checkpoint /content/drive/MyDrive/ttaguard_out/checkpoint_stage1_narrow.pt \
      --data_dir .../mitdb --cache_dir .../cache --nstdb_dir .../nstdb \
      --incart_dir .../incartdb --svdb_dir .../svdb \
      --scenarios clean,leo,noise_mixed_snr6,incart,svdb --methods main,pg_gate \
      --seeds 0,1,2 --out_dir /content/drive/MyDrive/ttaguard_out/bench
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import method_zoo
from dataset import DS1_TRAIN_RECORDS, DS2_RECORDS, ECGDataConfig, MITBIHDataset, build_datasets
from real_shift import NoiseBank, RealNoiseConfig, RealNoiseSimulator, load_external
from run_experiments import load_source_model_and_stream, real_data_available
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream

LOG_COLS = ["step", "record_id", "y_true", "y_pred", "entropy", "js_div", "adapted",
            "severity", "corrupt_frac", "latency_ms", "extra_elastic_alpha", "extra_adapt_loss"]
_NOISE_RE = re.compile(r"^noise_(bw|em|ma|mixed)_snr(-?\d+(?:\.\d+)?)$")


def _col(log: List[dict], k: str) -> np.ndarray:
    vals = [(np.nan if r.get(k) is None else float(r.get(k))) for r in log]
    return np.asarray(vals, dtype=np.float32)


def save_compact(path: Path, result, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: _col(result.log, k) for k in LOG_COLS}
    for k in ("step", "record_id", "y_true", "y_pred"):
        cols[k] = np.nan_to_num(cols[k], nan=-1).astype(np.int32)
    cols["adapted"] = cols["adapted"].astype(np.int8)
    meta = {**meta, "summary": {k: (None if v is None else (float(v) if isinstance(v, (int, float, np.floating)) else v))
                                for k, v in result.summary.items()
                                if isinstance(v, (int, float, np.floating, type(None)))}}
    np.savez_compressed(path, meta=json.dumps(meta), **cols)


def load_compact(path: Path) -> Tuple[dict, Dict[str, np.ndarray]]:
    d = np.load(path, allow_pickle=False)
    return json.loads(str(d["meta"])), {k: d[k] for k in d.files if k != "meta"}


# --------------------------------------------------------------------------- #
# scenario preparation
# --------------------------------------------------------------------------- #
class Scenario:
    def __init__(self, name: str, ds, sim, severity, is_real: bool):
        self.name, self.ds, self.sim, self.severity, self.is_real = name, ds, sim, severity, is_real


def _smoke_ds(model_n_leads: int, seed: int, n_records: int, beats: int):
    from baselines_tta import _SyntheticStreamDataset
    return _SyntheticStreamDataset(n_records=n_records, beats_per_record=beats, c=model_n_leads, l=256, seed=seed)


def prepare(scn: str, args, model, data_cfg: Optional[ECGDataConfig], ds2, real_model: bool,
            seed: int, noise_bank: Optional[NoiseBank]) -> Scenario:
    ramp = linear_ramp(start_idx=args.leo_start_idx, n_ramp=args.leo_n_ramp)
    if scn == "clean":
        return Scenario(scn, ds2, None, None, real_model)
    if scn == "leo":
        sim = SpaceShiftSimulator(SpaceShiftConfig(seed=seed))
        cap = args.leo_cap
        sev = (lambda i: min(cap, ramp(i))) if cap < 1.0 else ramp
        return Scenario(scn, ds2, sim, sev, real_model)
    m = _NOISE_RE.match(scn)
    if m:
        kinds = ("bw", "em", "ma") if m.group(1) == "mixed" else (m.group(1),)
        sim = RealNoiseSimulator(noise_bank, RealNoiseConfig(
            snr_db=float(m.group(2)), kinds=kinds, clean_start_s=args.noise_clean_start_s,
            segment_s=args.noise_segment_s, seed=seed))
        return Scenario(scn, ds2, sim, 1.0, real_model and getattr(noise_bank, "is_real", False))
    if scn in ("incart", "svdb"):
        d = {"incart": args.incart_dir, "svdb": args.svdb_dir}[scn]
        db = {"incart": "incartdb", "svdb": "svdb"}[scn]
        if real_model and d and Path(d).exists():
            from real_shift import list_external_records
            recs = list_external_records(db, d)
            if args.ext_n_records and len(recs) > args.ext_n_records:      # evenly spaced, deterministic
                pick = np.linspace(0, len(recs) - 1, args.ext_n_records).round().astype(int)
                recs = [recs[i] for i in pick]
            ds = load_external(db, d, data_cfg, cache_dir=args.cache_dir, records=recs,
                               max_beats_per_record=args.ext_beats_per_record)
            return Scenario(scn, ds, None, None, True)
        print(f"[run_benchmark] WARNING: {scn} data not available -> SYNTHETIC stand-in (smoke only)",
              file=sys.stderr)
        return Scenario(scn, _smoke_ds(model.cfg.in_channels, 100 + seed, 3, 300), None, None, False)
    raise ValueError(f"unknown scenario '{scn}'")


def source_beats_for_fisher(args, data_cfg: Optional[ECGDataConfig], ds2, real_model: bool, n: int = 500):
    """Unlabelled DS1-TRAIN beats (design-time data) for EATA's Fisher estimate."""
    rng = np.random.default_rng(0)
    if real_model and data_cfg is not None:
        cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": args.data_dir, "cache_dir": args.cache_dir})
        tr = MITBIHDataset(DS1_TRAIN_RECORDS[:6], cfg)
        idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
        return tr.x[idx], (tr.rr[idx] if tr.rr.shape[-1] > 0 else None)
    idx = rng.choice(len(ds2), size=min(n, len(ds2)), replace=False)
    rr = getattr(ds2, "rr", None)
    return ds2.x[idx], (rr[idx] if rr is not None and rr.shape[-1] > 0 else None)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None, help="one path or comma-separated list")
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--nstdb_dir", default=None)
    p.add_argument("--incart_dir", default=None)
    p.add_argument("--svdb_dir", default=None)
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--scenarios", default="clean,leo")
    p.add_argument("--methods", default="main")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n_beats", type=int, default=0,
                   help="0 = the FULL stream (all DS2 beats, ~49.7k). Any positive cap truncates by record "
                        "order and silently drops the LAST records -- 20000 covers only ~10 of 22 DS2 records")
    p.add_argument("--leo_cap", type=float, default=0.4)
    p.add_argument("--leo_start_idx", type=int, default=500)
    p.add_argument("--leo_n_ramp", type=int, default=2000)
    p.add_argument("--noise_clean_start_s", type=float, default=300.0)
    p.add_argument("--noise_segment_s", type=float, default=120.0)
    p.add_argument("--ext_n_records", type=int, default=12)
    p.add_argument("--ext_beats_per_record", type=int, default=1700)
    p.add_argument("--rolling_window", type=int, default=500)
    p.add_argument("--hp", default="{}", help='JSON, e.g. {"rotta":{"lr":0.0003}}')
    p.add_argument("--out_dir", default="./results/benchmark")
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(argv)

    methods = method_zoo.expand(a.methods)
    scenarios = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    hp = json.loads(a.hp)
    ckpts: List[Optional[str]] = [c.strip() for c in a.checkpoint.split(",")] if a.checkpoint else [None]
    out_root = Path(a.out_dir)
    print(f"[run_benchmark] {len(scenarios)} scenarios x {len(ckpts)} ckpt x {len(seeds)} seeds x "
          f"{len(methods)} methods = {len(scenarios)*len(ckpts)*len(seeds)*len(methods)} jobs -> {out_root}")
    if a.dry_run:
        return 0

    from eval_cross_domain import load_checkpoint
    noise_bank: Optional[NoiseBank] = None
    n_done = n_skip = 0
    t_start = time.time()
    for ck in ckpts:
        use_real = ck is not None and real_data_available(a.data_dir, DS2_RECORDS)
        if use_real:
            model, _, data_cfg, trained = load_checkpoint(ck, "cpu", a.n_leads)
            use_real = trained
        if use_real:
            data_cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": a.data_dir,
                                        "cache_dir": a.cache_dir})
            ds2 = build_datasets(data_cfg)["test"]
        else:
            if ck is not None:                       # never silently benchmark a synthetic model in place of a real one
                raise SystemExit(
                    f"[run_benchmark] ERROR: cannot use real data/checkpoint for '{ck}'. Check that (1) Google Drive "
                    f"is mounted, (2) `pip install wfdb` was run in THIS runtime, (3) --data_dir '{a.data_dir}' "
                    f"contains the MIT-BIH .dat files, (4) the checkpoint file exists.")
            model, ds2, _ = load_source_model_and_stream(None, a.data_dir, a.cache_dir, a.n_leads, 3, 600, 0)
            data_cfg = None
        ck_tag = Path(ck).stem if ck else "synthetic"
        if any(_NOISE_RE.match(s) for s in scenarios) and noise_bank is None:
            if use_real and a.nstdb_dir and Path(a.nstdb_dir).exists():
                noise_bank = NoiseBank.from_physionet(a.nstdb_dir).half("second"); noise_bank.is_real = True   # 1st half is reserved for training-time noise augmentation
            else:
                print("[run_benchmark] WARNING: NSTDB not found -> SYNTHETIC noise (smoke only)", file=sys.stderr)
                noise_bank = NoiseBank.synthetic(); noise_bank.is_real = False
        fisher = None
        for scn in scenarios:
            for seed in seeds:
                scen = prepare(scn, a, model, data_cfg, ds2, use_real, seed, noise_bank)
                if seed == seeds[0]:
                    from dataset import chronological_stream_indices
                    ix = chronological_stream_indices(scen.ds, a.n_beats or None)
                    rid = scen.ds.record_ids[ix].numpy()
                    cc = np.bincount(scen.ds.y[ix].numpy(), minlength=5).tolist()
                    print(f"[run_benchmark] stream {scn}: {len(ix)} beats, {len(set(rid.tolist()))} of "
                          f"{len(set(scen.ds.record_ids.tolist()))} records, class counts N/S/V/F/Q={cc}", flush=True)
                for key in methods:
                    path = out_root / scn / ck_tag / f"seed{seed}" / f"{key}.npz"
                    if path.exists() and not a.overwrite:
                        n_skip += 1
                        continue
                    torch.manual_seed(seed)
                    meth = method_zoo.build_method(key, hp)
                    if method_zoo.needs_source_beats(key):
                        fisher = fisher or source_beats_for_fisher(a, data_cfg, ds2, use_real)
                        meth.set_fisher_data(*fisher)
                    m = copy.deepcopy(model).to(a.device)
                    sim = scen.sim
                    stream = make_beat_stream(scen.ds, n_beats=(a.n_beats or None), sim=sim,
                                              severity=scen.severity if scen.severity is not None else 1.0)
                    ev = StreamingEvaluator(meth, StreamingConfig(device=a.device, rolling_window=a.rolling_window,
                                                                  log_every=0))
                    t0 = time.time()
                    res = ev.run(m, stream)
                    cfg_dict = dataclasses.asdict(meth.cfg) if hasattr(meth, "cfg") and dataclasses.is_dataclass(meth.cfg) else {}
                    cfg_dict = {k: v for k, v in cfg_dict.items() if isinstance(v, (int, float, str, bool, type(None)))}
                    save_compact(path, res, {"scenario": scn, "ckpt": ck_tag, "seed": seed, "method": key,
                                             "is_real": bool(scen.is_real), "n_beats": len(res.log),
                                             "method_cfg": cfg_dict})
                    s = res.summary
                    n_done += 1
                    print(f"[{scn}|{ck_tag}|seed{seed}] {key:<16} acc={s['accuracy']:.3f} "
                          f"F1exFQ={s.get('macro_f1_ex_fq') or float('nan'):.3f} D_JS={s['mean_js_div']:.3f} "
                          f"({time.time()-t0:.0f}s){'' if scen.is_real else '  [SYNTHETIC]'}", flush=True)
    print(f"[run_benchmark] done: {n_done} run, {n_skip} skipped (already present), "
          f"{(time.time()-t_start)/60:.1f} min. Outputs under {out_root}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    import shutil
    out = "./results/_selftest_run_benchmark"
    shutil.rmtree(out, ignore_errors=True)
    print("[run_benchmark] self-test: every scenario type x 4 methods on synthetic data")
    args = ["--scenarios", "clean,leo,noise_mixed_snr6,noise_em_snr0,incart,svdb",
            "--methods", "source_only,pg,pg_gate,eata", "--seeds", "0,1", "--n_beats", "300",
            "--leo_start_idx", "30", "--leo_n_ramp", "60", "--noise_clean_start_s", "0",
            "--noise_segment_s", "20", "--rolling_window", "50", "--out_dir", out]
    assert main(args) == 0
    files = sorted(Path(out).rglob("*.npz"))
    assert len(files) == 6 * 1 * 2 * 4, len(files)
    meta, cols = load_compact(files[0])
    assert set(LOG_COLS) <= set(cols) and meta["is_real"] is False and len(cols["y_true"]) == meta["n_beats"]
    # resumability: second call skips everything
    before = {f: f.stat().st_mtime_ns for f in files}
    assert main(args) == 0
    assert all(f.stat().st_mtime_ns == before[f] for f in files), "resume must not recompute"
    # real-noise scenarios must actually corrupt something in smoke mode
    m2, c2 = load_compact(Path(out) / "noise_mixed_snr6" / "synthetic" / "seed0" / "source_only.npz")
    assert np.nanmax(c2["corrupt_frac"]) == 1.0
    m3, c3 = load_compact(Path(out) / "clean" / "synthetic" / "seed0" / "pg.npz")
    assert np.all(np.isnan(c3["corrupt_frac"])) or np.nanmax(c3["corrupt_frac"]) == 0
    print(f"Self-test passed ({len(files)} job files, resume verified).")
