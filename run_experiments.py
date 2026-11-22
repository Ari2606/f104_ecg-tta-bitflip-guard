"""
run_experiments.py  --  this project, Stage 7 (Full Continuous Streaming Experiments)

Orchestrates the paper's headline result: Source-Only / TENT(sparse) / SQI-gated TENT
/ this project, all four run through the SAME `streaming_eval.StreamingEvaluator`
(N=1, predict-then-adapt) over the SAME long, chronologically-ordered, space-shifted DS2
beat stream, starting from the SAME `train_stage1.py` checkpoint's frozen weights. Produces
everything `safety_metrics.py` can build from a `StreamingResult`: per-beat CSV/JSON logs,
the HURR/BUAR safety table, the long-horizon drift-curve figure, and a paper-ready LaTeX
results table.

Real vs. synthetic data (read before interpreting a run's numbers)
---------------------------------------------------------------------------
This script's default behavior mirrors every other file in this project
(`train_stage1.py`, `eval_cross_domain.py`, ...): if a Stage-1 checkpoint is given AND
real MIT-BIH DS2 records are available under `--data_dir` (requires the `wfdb` package
and the PhysioNet `mitdb` files -- this sandbox has neither: no network route to
PhysioNet and `wfdb` is not installed), it evaluates on real DS2 patients. Otherwise it
prints an explicit WARNING and falls back to a synthetic multi-record stream + a
briefly-warmed-up (not really trained) synthetic model, so the full pipeline --
methods, metrics, plots, LaTeX -- can be exercised and self-tested end-to-end without
that dependency. Synthetic-mode output files are also written, but the run is clearly
labeled "SMOKE TEST (synthetic data)" everywhere it's reported, and the paper's real
numbers must come from a real-checkpoint, real-data run once both are available.

Shape legend: B = batch, C = leads, L = 256 samples, K = 5 classes.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from base_cnn import CNNConfig, BeatCNN
from baselines_tta import (BeatRhythmTTA, BeatRhythmTTAConfig, RoTTAConfig, RoTTATTA,
                           SQIGatedTent, SQIGatedTentConfig, SourceOnlyTTA, TentConfig, TentTTA)
from dataset import ECGDataConfig
from safety_controller import GuardedTTA, SafetyControllerConfig
from safety_metrics import calibrate_thresholds, compute_safety_metrics, save_all_artifacts
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import (StreamingConfig, StreamingEvaluator, StreamingResult,
                            compare_methods, make_beat_stream)


# --------------------------------------------------------------------------- #
# Real-data availability check (never attempts a network download -- see module docstring)
# --------------------------------------------------------------------------- #
def real_data_available(data_dir: str, records) -> bool:
    import importlib.util
    if importlib.util.find_spec("wfdb") is None:
        return False
    d = Path(data_dir)
    return any((d / f"{r}.dat").exists() for r in records)


# --------------------------------------------------------------------------- #
# Model / data loading (real checkpoint+DS2, or a labeled synthetic fallback)
# --------------------------------------------------------------------------- #
def load_source_model_and_stream(checkpoint_path: Optional[str], data_dir: str, cache_dir: str,
                                 n_leads: int, n_records: int, beats_per_record: int, seed: int
                                 ) -> "tuple[BeatCNN, object, bool]":
    """Returns (frozen source model with anchor set, a duck-typed beat dataset, is_real_data)."""
    from dataset import DS2_RECORDS

    use_real = checkpoint_path is not None and real_data_available(data_dir, DS2_RECORDS)
    if use_real:
        from dataset import build_datasets
        from eval_cross_domain import load_checkpoint
        model, cnn_cfg, data_cfg, is_trained = load_checkpoint(checkpoint_path, "cpu", n_leads)
        if not is_trained:
            use_real = False
        else:
            data_cfg = ECGDataConfig(**{**asdict(data_cfg), "data_dir": data_dir, "cache_dir": cache_dir})
            ds = build_datasets(data_cfg)["test"]                  # DS2, record-disjoint from DS1
            return model, ds, True

    print("[run_experiments] WARNING: no usable Stage-1 checkpoint + real DS2 data "
         "(needs --checkpoint, the `wfdb` package, and PhysioNet mitdb files under "
         "--data_dir -- none of that is available in this sandbox). Falling back to a "
         "SMOKE TEST on synthetic data; treat all reported numbers as a pipeline check, "
         "not a scientific result.", file=sys.stderr)
    from baselines_tta import _SyntheticStreamDataset
    torch.manual_seed(seed)
    model = BeatCNN(CNNConfig(in_channels=n_leads))
    ds = _SyntheticStreamDataset(n_records=n_records, beats_per_record=beats_per_record,
                                 c=n_leads, l=256, seed=seed)
    # Mini-batched supervised warm-up before freeze_anchor() -- establishes a non-trivial,
    # non-uniform-output source model to adapt from (see every other file's self-test for
    # the same rationale: a random-init model's near-uniform outputs give near-zero
    # entropy gradients almost everywhere, which is a degenerate smoke test).
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()), lr=1e-3)
    model.train()
    for _ in range(15):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(model(ds.x[idx]).logits, ds.y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    model.freeze_anchor()
    return model, ds, False


# --------------------------------------------------------------------------- #
# Main experiment runner
# --------------------------------------------------------------------------- #
def run_all_methods(source_model: BeatCNN, ds, n_beats: int, severity_start_idx: int,
                    severity_n_ramp: int, sim_seed: int, rolling_window: int,
                    device: str = "cpu", severity_cap: float = 1.0,
                    pg_tau: float = 0.05, pg_gamma: float = 0.3, pg_kappa: float = 40.0,
                    rotta_memory_per_class: int = 16, rotta_batch_size: int = 16,
                    rotta_update_every: int = 1, rotta_ema_alpha: float = 0.999,
                    bhr_lrs: Sequence[float] = (1e-6, 1e-4), bhr_steps: int = 5,
                    tent_lr: float = 1e-3, rotta_lr: float = 1e-3,
                    ) -> Dict[str, StreamingResult]:
    if n_beats is not None and 0 < n_beats < len(ds):
        print(f"[run_experiments] WARNING: n_beats={n_beats} < {len(ds)} beats in the dataset: the stream is "
              f"truncated by record order and the LAST records are never evaluated. Use run_benchmark.py "
              f"(full stream by default) for any reported number.", file=sys.stderr)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=sim_seed))
    _ramp = linear_ramp(start_idx=severity_start_idx, n_ramp=severity_n_ramp)
    # severity_cap<1.0 bounds the ramp's plateau below full severity -- e.g. a transient
    # sensor/radiation event that peaks at moderate severity rather than one that ramps to
    # the simulator's maximum and stays there forever. Default 1.0 reproduces the original
    # unbounded ramp exactly (no behavior change for existing commands/results).
    severity_schedule = (lambda i: min(severity_cap, _ramp(i))) if severity_cap < 1.0 else _ramp
    cfg = StreamingConfig(device=device, rolling_window=rolling_window, log_every=0)

    methods = [
        SourceOnlyTTA(),
        TentTTA(TentConfig(scope="sparse", lr=tent_lr)),
        SQIGatedTent(SQIGatedTentConfig()),
        # RoTTA (Yuan et al., CVPR 2023) -- a REAL, NAMED, PUBLISHED TTA method, unlike
        # SQIGatedTent above (this project's own design). Its CSTU category-
        # balanced replay memory + EMA teacher are specifically built for temporally-
        # correlated streams, i.e. exactly this project's N=1 chronological protocol.
        RoTTATTA(RoTTAConfig(lr=rotta_lr, memory_per_class=rotta_memory_per_class, batch_size=rotta_batch_size,
                             update_every=rotta_update_every, ema_alpha=rotta_ema_alpha)),
        # BeatRhythm-TTA (Jiang et al., arXiv:2608.23347), beat-level adaptation, defaults taken
        # from the authors' released code. One run per learning rate: 1e-6 is their online/
        # continual value, 1e-4 their larger one (see BeatRhythmTTAConfig for every deviation).
        *[BeatRhythmTTA(BeatRhythmTTAConfig(lr=lr, steps=bhr_steps)) for lr in bhr_lrs],
        # tau/gamma/kappa default to the ORIGINAL untuned values (0.05/0.3/40.0) unless
        # overridden -- see tune_safety_controller.py, which found tau~0.12/gamma~0.15
        # beats these defaults on DS1-VAL (never on this DS2 test stream itself).
        GuardedTTA(SafetyControllerConfig(tau=pg_tau, gamma=pg_gamma, kappa=pg_kappa)),
    ]
    results: Dict[str, StreamingResult] = {}
    for method in methods:
        m = copy.deepcopy(source_model).to(device)              # identical starting weights, every method
        stream = make_beat_stream(ds, n_beats=n_beats, sim=sim, severity=severity_schedule)
        ev = StreamingEvaluator(method, cfg)
        results[method.name] = ev.run(m, stream)
        print(f"[run_experiments] {method.name:<20} done: "
             f"acc={results[method.name].summary['accuracy']:.3f}  "
             f"meanD_JS={results[method.name].summary['mean_js_div']}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None, help="train_stage1.py checkpoint_stage1.pt (optional)")
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--n_beats", type=int, default=10_000)
    p.add_argument("--severity_start_idx", type=int, default=500,
                   help="beat index the space-shift severity ramp begins at (a calm "
                        "calibration window precedes it, e.g. for SQI-gate's gate)")
    p.add_argument("--severity_n_ramp", type=int, default=2000,
                   help="beats over which severity linearly ramps 0 -> 1")
    p.add_argument("--rolling_window", type=int, default=500)
    p.add_argument("--severity_cap", type=float, default=1.0,
                   help="caps the severity ramp's plateau below full severity (e.g. 0.4 "
                        "for a moderate, transient-event mission profile instead of a "
                        "permanent worst-case corruption floor). Default 1.0 = uncapped, "
                        "identical to every run before this flag existed.")
    p.add_argument("--pg_tau", type=float, default=0.05,
                   help="this project elastic trust region tau (nats). Default 0.05 is "
                        "the original, untuned value; see tune_safety_controller.py for a "
                        "DS1-VAL-based search (never tune this against this script's own "
                        "DS2 output -- that's the test set).")
    p.add_argument("--pg_gamma", type=float, default=0.3,
                   help="this project elastic trust region max pull strength. Default "
                        "0.3 is the original, untuned value.")
    p.add_argument("--pg_kappa", type=float, default=40.0,
                   help="this project elastic trust region sigmoid steepness.")
    p.add_argument("--rotta_memory_per_class", type=int, default=16,
                   help="RoTTA baseline: CSTU replay memory capacity per class.")
    p.add_argument("--rotta_batch_size", type=int, default=16,
                   help="RoTTA baseline: class-balanced replay batch size per update step.")
    p.add_argument("--rotta_update_every", type=int, default=1,
                   help="RoTTA baseline: take a student replay step every N beats.")
    p.add_argument("--rotta_ema_alpha", type=float, default=0.999,
                   help="RoTTA baseline: teacher EMA momentum.")
    p.add_argument("--tent_lr", type=float, default=1e-3,
                   help="TENT baseline learning rate (tune on DS1-VAL with tune_baselines.py).")
    p.add_argument("--rotta_lr", type=float, default=1e-3,
                   help="RoTTA baseline learning rate (tune on DS1-VAL with tune_baselines.py).")
    p.add_argument("--bhr_lrs", type=str, default="1e-6,1e-4",
                   help="BeatRhythm-TTA baseline: comma-separated learning rates, one run each.")
    p.add_argument("--bhr_steps", type=int, default=5,
                   help="BeatRhythm-TTA baseline: gradient steps per beat (paper's online setting "
                        "is 5; their released yaml default is 1). Cost is linear in this.")
    p.add_argument("--sim_seed", type=int, default=0)
    p.add_argument("--synthetic_n_records", type=int, default=6)
    p.add_argument("--synthetic_beats_per_record", type=int, default=2000)
    p.add_argument("--corrupt_thresh", type=float, default=None,
                   help="HURR corrupt-beat threshold; default None = auto-calibrate (recommended)")
    p.add_argument("--clean_thresh", type=float, default=None,
                   help="BUAR clean-beat threshold; default None = auto-calibrate (recommended)")
    p.add_argument("--output_dir", default="./results/stage7")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    source_model, ds, is_real = load_source_model_and_stream(
        a.checkpoint, a.data_dir, a.cache_dir, a.n_leads,
        a.synthetic_n_records, a.synthetic_beats_per_record, a.seed)
    print(f"[run_experiments] data source: {'REAL DS2' if is_real else 'SYNTHETIC (smoke test)'}  "
         f"| stream target length: {a.n_beats} beats")

    results = run_all_methods(source_model, ds, a.n_beats, a.severity_start_idx,
                              a.severity_n_ramp, a.sim_seed, a.rolling_window, a.device,
                              severity_cap=a.severity_cap, pg_tau=a.pg_tau, pg_gamma=a.pg_gamma,
                              pg_kappa=a.pg_kappa, rotta_memory_per_class=a.rotta_memory_per_class,
                              rotta_batch_size=a.rotta_batch_size, rotta_update_every=a.rotta_update_every,
                              rotta_ema_alpha=a.rotta_ema_alpha,
                              bhr_lrs=[float(v) for v in a.bhr_lrs.split(",")],
                              bhr_steps=a.bhr_steps, tent_lr=a.tent_lr, rotta_lr=a.rotta_lr)

    print()
    print(compare_methods(results))

    out_dir = Path(a.output_dir) / ("real" if is_real else "synthetic_smoke_test")
    for name, r in results.items():
        r.save_csv(str(out_dir / "per_beat_logs" / f"{name}.csv"))
        r.save_json(str(out_dir / "per_beat_logs" / f"{name}.json"))
    artifact_paths = save_all_artifacts(results, str(out_dir), corrupt_thresh=a.corrupt_thresh,
                                        clean_thresh=a.clean_thresh, rolling_window=a.rolling_window)

    calib = calibrate_thresholds(results)
    thresh = calib or {"corrupt_thresh": a.corrupt_thresh, "clean_thresh": a.clean_thresh}
    safety = {name: compute_safety_metrics(r, thresh["corrupt_thresh"], thresh["clean_thresh"])
             for name, r in results.items()}
    print(f"\nHURR/BUAR thresholds: corrupt>={thresh['corrupt_thresh']:.4f}  "
         f"clean<={thresh['clean_thresh']:.4f}")
    print("method               HURR(%)  BUAR(%)")
    for name, sm in safety.items():
        hurr = "n/a" if sm.hurr is None else f"{100*sm.hurr:.1f}"
        buar = "n/a" if sm.buar is None else f"{100*sm.buar:.1f}"
        print(f"{name:<20} {hurr:>7}  {buar:>7}")

    print(f"\n[run_experiments] wrote per-beat logs, LaTeX table, drift-curve PNG, and JSON "
         f"summary under {out_dir}/  {'' if is_real else '(SMOKE TEST -- synthetic data)'}")
    for k, v in artifact_paths.items():
        print(f"  {k}: {v}")

    return {"results": results, "safety": safety, "artifact_paths": artifact_paths, "is_real": is_real}


# --------------------------------------------------------------------------- #
# Entry point. `python run_experiments.py --checkpoint ... --data_dir ...` (any real
# CLI args) runs `main()` for real, exactly as a normal script would -- this is the
# ONLY way to get the real-DS2, real-checkpoint Stage 7 run. Bare `python
# run_experiments.py` (no args at all) instead runs the self-test below: a full
# end-to-end pass on the synthetic fallback path (small n_beats for speed),
# exercising every step `main()`'s real-checkpoint/real-data path would also take --
# data loading fallback, all 7 methods, safety metrics, and every artifact.
#
# BUG FIXED HERE: this used to unconditionally call `main([...hardcoded synthetic
# args...])` regardless of sys.argv, so passing --checkpoint/--data_dir on the real
# command line was silently ignored and every run fell back to the synthetic smoke
# test -- exactly what you just hit. Real CLI args now actually reach `main()`.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()                                    # real run: argv goes straight to argparse
    else:
        print("[run_experiments] no CLI args given -- running self-test: full Stage-7 pipeline "
             "(synthetic-data fallback path, small n_beats for speed). Pass --checkpoint/"
             "--data_dir/etc. for a real run.")

        out = main([
            "--n_beats", "1500",
            "--severity_start_idx", "150",
            "--severity_n_ramp", "600",
            "--rolling_window", "150",
            "--synthetic_n_records", "3",
            "--synthetic_beats_per_record", "600",
            "--output_dir", "./results/_selftest_run_experiments",
            "--device", "cpu",
            "--seed", "0",
            # RoTTA does a class-balanced replay forward/backward EVERY beat by default
            # (update_every=1, batch_size=16) -- fine for a real GPU run but too slow for
            # a CPU self-test at these beat counts, so this self-test only thins the
            # replay cadence/size, not the mechanism itself (still exercises CSTU
            # admission, EMA teacher update, and the replay step every 4th beat).
            "--rotta_memory_per_class", "8",
            "--rotta_batch_size", "8",
            "--rotta_update_every", "4",
            "--bhr_steps", "1",
        ])

        assert out["is_real"] is False, "this sandbox has no MITDB/wfdb -- expected the synthetic fallback"
        results = out["results"]
        assert set(results.keys()) == {"source_only", "tent_sparse", "sqi_gated_tent",
                                       "rotta_sparse", "beatrhythm_tta_lr1e-6",
                                       "beatrhythm_tta_lr1e-4", "ttaguard"}
        for name, r in results.items():
            assert r.summary["n_beats"] == 1500, f"{name}: expected 1500 beats, got {r.summary['n_beats']}"

        # ---- core Stage 4/5 relationship must still hold end-to-end through this orchestrator ----
        src, tent, pg = results["source_only"].summary, results["tent_sparse"].summary, results["ttaguard"].summary
        assert src["mean_js_div"] < 1e-6, "Source-Only must show ~0 divergence from its own anchor"
        assert pg["mean_js_div"] < 0.5 * tent["mean_js_div"], \
            "this project must show meaningfully lower drift than naive TENT"
        print(f"\nthis project mean D_JS={pg['mean_js_div']:.3e} vs. TENT's {tent['mean_js_div']:.3e} -> OK")

        rotta = results["rotta_sparse"].summary
        assert rotta["adaptation_rate"] > 0.0, "RoTTA must take at least one replay update"
        print(f"RoTTA mean D_JS={rotta['mean_js_div']:.3e} acc={rotta['accuracy']:.3f} -> OK")

        # ---- safety metrics were computed and are well-formed ----
        safety = out["safety"]
        for name, sm in safety.items():
            assert sm.hurr is not None and sm.buar is not None, f"{name}: HURR/BUAR must be computed"
            assert 0.0 <= sm.hurr <= 1.0 and 0.0 <= sm.buar <= 1.0
        print("HURR/BUAR computed and well-formed for all 7 methods -> OK")

        # ---- every artifact file was actually written and is non-empty ----
        for k, path in out["artifact_paths"].items():
            p = Path(path)
            assert p.exists() and p.stat().st_size > 0, f"{k} missing or empty: {path}"
        print("LaTeX table, drift-curve PNG, and JSON summary all written -> OK")

        # ---- per-beat CSV/JSON logs were written for every method ----
        log_dir = Path("./results/_selftest_run_experiments/synthetic_smoke_test/per_beat_logs")
        for name in results:
            for ext in ("csv", "json"):
                p = log_dir / f"{name}.{ext}"
                assert p.exists() and p.stat().st_size > 0, f"missing per-beat log: {p}"
        print(f"per-beat CSV/JSON logs written for all 7 methods under {log_dir}/ -> OK")

        print("\nSelf-test passed.")
