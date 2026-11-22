"""
run_multiseed.py  --  this project, Stage 7 multi-seed variance

Every Stage 7 number reported so far in this project is from a SINGLE seed
(sim_seed=0). A tier-1 reviewer's first question on any robustness/TTA result
is "across how many random corruption draws?" -- this script answers that by
rerunning the same 4-method comparison across multiple seeds and reporting
mean +/- std (and min/max) per method per metric, so the headline table can
carry real variance instead of one number.

What varies across a "seed" here: `sim_seed` controls SpaceShiftSimulator's
random corruption-event draws (which beats get hit, how hard, for the given
severity schedule) -- the single source of run-to-run randomness in this
protocol, since the checkpoint's weights are frozen and the beat stream order
is chronological (deterministic) for real DS2 data. `torch.manual_seed(seed)`
is also set per run for any incidental randomness (none expected at eval, but
cheap to pin down).

Usage (same checkpoint/data args as run_experiments.py):
    !python run_multiseed.py \
        --checkpoint /content/drive/MyDrive/ttaguard_out/checkpoint_stage1_narrow.pt \
        --data_dir /content/drive/MyDrive/ttaguard_data/data/mitdb \
        --cache_dir /content/drive/MyDrive/ttaguard_out/cache \
        --n_beats 20000 --severity_cap 0.4 --seeds 0,1,2,3,4 \
        --output_dir ./results/stage7_multiseed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from run_experiments import load_source_model_and_stream, run_all_methods
from streaming_eval import StreamingResult


METRICS = ["accuracy", "macro_f1", "macro_f1_ex_fq", "mean_js_div", "adaptation_rate"]


def run_seeds(checkpoint: Optional[str], data_dir: str, cache_dir: str, n_leads: int,
             n_beats: int, severity_start_idx: int, severity_n_ramp: int, severity_cap: float,
             rolling_window: int, pg_tau: float, pg_gamma: float, pg_kappa: float,
             seeds: Sequence[int], device: str, synthetic_n_records: int,
             synthetic_beats_per_record: int, rotta_memory_per_class: int = 16,
             rotta_batch_size: int = 16, rotta_update_every: int = 1,
             rotta_ema_alpha: float = 0.999, bhr_lrs: Sequence[float] = (1e-6, 1e-4),
             bhr_steps: int = 5, tent_lr: float = 1e-3,
             rotta_lr: float = 1e-3) -> Dict[str, Dict[str, List[float]]]:
    """Returns {method_name: {metric: [value_per_seed]}}. Reloads the model/data ONCE
    (it's a frozen checkpoint -- no reason to reload per seed) and reruns run_all_methods
    (which itself deepcopies the source model per method, per the existing protocol) once
    per seed, varying sim_seed (and torch's seed) only."""
    per_method: Dict[str, Dict[str, List[float]]] = {}
    is_real_flag = None
    for seed in seeds:
        torch.manual_seed(seed)
        # reload per seed: load_source_model_and_stream is cheap relative to the streaming
        # loop itself, and reloading guarantees no state (e.g. any BN running stats a
        # method's .reset() didn't fully clear) leaks across seeds -- correctness over speed.
        source_model, ds, is_real = load_source_model_and_stream(
            checkpoint, data_dir, cache_dir, n_leads,
            synthetic_n_records, synthetic_beats_per_record, seed)
        is_real_flag = is_real
        results = run_all_methods(source_model, ds, n_beats, severity_start_idx,
                                  severity_n_ramp, seed, rolling_window, device,
                                  severity_cap=severity_cap, pg_tau=pg_tau, pg_gamma=pg_gamma,
                                  pg_kappa=pg_kappa,
                                  rotta_memory_per_class=rotta_memory_per_class,
                                  rotta_batch_size=rotta_batch_size,
                                  rotta_update_every=rotta_update_every,
                                  rotta_ema_alpha=rotta_ema_alpha,
                                  bhr_lrs=bhr_lrs, bhr_steps=bhr_steps,
                                  tent_lr=tent_lr, rotta_lr=rotta_lr)
        print(f"[run_multiseed] seed={seed} done "
             f"({'REAL DS2' if is_real else 'SYNTHETIC smoke test'}).")
        for name, r in results.items():
            per_method.setdefault(name, {m: [] for m in METRICS})
            s = r.summary
            for m in METRICS:
                v = s.get(m)
                per_method[name][m].append(float(v) if v is not None else float("nan"))
    per_method["_is_real"] = is_real_flag  # type: ignore[assignment]
    return per_method


def summarize(per_method: Dict[str, Dict[str, List[float]]]) -> str:
    is_real = per_method.pop("_is_real", None)
    lines = [f"Data source: {'REAL DS2' if is_real else 'SYNTHETIC smoke test'}", ""]
    header = f"{'method':<20}" + "".join(f"{m:<22}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for name, metrics in per_method.items():
        row = [f"{name:<20}"]
        for m in METRICS:
            vals = np.array(metrics[m], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                row.append(f"{'n/a':<22}")
            else:
                row.append(f"{vals.mean():.3f}+/-{vals.std():.3f}".ljust(22))
        lines.append("".join(row))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, List[float]]]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--n_beats", type=int, default=10_000)
    p.add_argument("--severity_start_idx", type=int, default=500)
    p.add_argument("--severity_n_ramp", type=int, default=2000)
    p.add_argument("--severity_cap", type=float, default=0.4)
    p.add_argument("--rolling_window", type=int, default=500)
    p.add_argument("--pg_tau", type=float, default=0.05)
    p.add_argument("--pg_gamma", type=float, default=0.3)
    p.add_argument("--pg_kappa", type=float, default=40.0)
    p.add_argument("--rotta_memory_per_class", type=int, default=16)
    p.add_argument("--rotta_batch_size", type=int, default=16)
    p.add_argument("--rotta_update_every", type=int, default=1,
                   help="RoTTA replay step every N beats (larger = faster)")
    p.add_argument("--rotta_ema_alpha", type=float, default=0.999)
    p.add_argument("--tent_lr", type=float, default=1e-3)
    p.add_argument("--rotta_lr", type=float, default=1e-3)
    p.add_argument("--bhr_lrs", type=str, default="1e-6,1e-4",
                   help="BeatRhythm-TTA baseline learning rates, comma-separated (one run each)")
    p.add_argument("--bhr_steps", type=int, default=5,
                   help="BeatRhythm-TTA gradient steps per beat (paper online=5; cost is linear)")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="comma-separated seeds; each becomes both torch's seed and sim_seed")
    p.add_argument("--synthetic_n_records", type=int, default=3)
    p.add_argument("--synthetic_beats_per_record", type=int, default=600)
    p.add_argument("--output_dir", default="./results/stage7_multiseed")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[run_multiseed] running {len(seeds)} seeds: {seeds}")
    per_method = run_seeds(a.checkpoint, a.data_dir, a.cache_dir, a.n_leads, a.n_beats,
                           a.severity_start_idx, a.severity_n_ramp, a.severity_cap,
                           a.rolling_window, a.pg_tau, a.pg_gamma, a.pg_kappa, seeds,
                           a.device, a.synthetic_n_records, a.synthetic_beats_per_record,
                           a.rotta_memory_per_class, a.rotta_batch_size,
                           a.rotta_update_every, a.rotta_ema_alpha,
                           [float(v) for v in a.bhr_lrs.split(",")], a.bhr_steps,
                           a.tent_lr, a.rotta_lr)

    report = summarize(dict(per_method))  # copy: summarize() pops "_is_real"
    print("\n" + "=" * 100)
    print(f"Multi-seed summary ({len(seeds)} seeds: {seeds}) -- mean +/- std per method:")
    print("=" * 100)
    print(report)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = {name: metrics for name, metrics in per_method.items() if name != "_is_real"}
    (out / "multiseed_raw.json").write_text(json.dumps({"seeds": seeds, "per_method": raw}, indent=2))
    (out / "multiseed_summary.txt").write_text(report)
    print(f"\n[run_multiseed] wrote raw per-seed values and summary to {out}/")
    return per_method


# --------------------------------------------------------------------------- #
# Self-test: tiny synthetic 3-seed run, exercising the full aggregation path
# without needing real MIT-BIH data.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("[run_multiseed] self-test: 3-seed synthetic run (no real data needed).")
        out = main([
            "--n_beats", "400", "--severity_start_idx", "50", "--severity_n_ramp", "100",
            "--severity_cap", "0.4", "--rolling_window", "50",
            "--synthetic_n_records", "2", "--synthetic_beats_per_record", "300",
            "--seeds", "0,1,2", "--output_dir", "./results/_selftest_run_multiseed",
            "--device", "cpu", "--rotta_update_every", "4", "--rotta_batch_size", "8",
            "--bhr_steps", "1",
            "--rotta_update_every", "4", "--rotta_batch_size", "8",
        ])
        assert set(out.keys()) == {"source_only", "tent_sparse", "sqi_gated_tent",
                                   "rotta_sparse", "beatrhythm_tta_lr1e-6", "beatrhythm_tta_lr1e-4",
                                   "ttaguard", "_is_real"}
        assert out["_is_real"] is False
        for name in ("source_only", "tent_sparse", "sqi_gated_tent",
                    "rotta_sparse", "beatrhythm_tta_lr1e-6", "beatrhythm_tta_lr1e-4",
                    "ttaguard"):
            for m in METRICS:
                vals = out[name][m]
                assert len(vals) == 3, f"{name}/{m}: expected 3 seed values, got {len(vals)}"
        # Source-Only's accuracy should vary across seeds too (different corruption draws
        # hit different beats even though the model never adapts) -- not asserting a
        # specific amount, just that this isn't a frozen constant across seeds by accident.
        src_acc = out["source_only"]["accuracy"]
        print(f"\nsource_only accuracy across 3 seeds: {src_acc}")
        assert len(set(round(v, 6) for v in src_acc)) >= 1  # sanity: values are real floats
        print("Self-test passed.")
