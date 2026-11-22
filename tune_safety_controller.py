"""
tune_safety_controller.py  --  this project, Stage 5 hyperparameter search

Grid-searches the elastic trust region's (tau, gamma) on DS1-VAL (the held-out,
record-disjoint validation split from train_stage1.py's own protocol), NEVER on
DS2 (the test set `run_experiments.py`'s headline numbers come from). Tuning on
the test set would invalidate every number this project has reported so far --
this script exists specifically so that doesn't happen.

Why tau/gamma, not kappa: tau sets WHERE the restoring force ramps up (in D_JS
nats), gamma sets its MAX strength -- these two jointly control the real
trade-off (how much beneficial adaptation survives vs. how tightly drift is
capped). kappa only controls how sharply alpha transitions around tau; it's a
secondary knob, left at its established default (40.0) for this search.

Usage (same checkpoint/data args as run_experiments.py):
    !python tune_safety_controller.py \
        --checkpoint /content/drive/MyDrive/ttaguard_out/checkpoint_stage1_narrow.pt \
        --data_dir /content/drive/MyDrive/ttaguard_data/data/mitdb \
        --cache_dir /content/drive/MyDrive/ttaguard_out/cache \
        --n_beats 3000 --severity_cap 0.4

Prints one row per (tau, gamma) combo (accuracy, macro_f1_ex_fq, mean_js_div),
sorted best-macro_f1_ex_fq-first, so you can read off a winner, then confirm it
with ONE final run_experiments.py pass on the real DS2 test stream.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from base_cnn import CNNConfig, BeatCNN
from dataset import ECGDataConfig
from safety_controller import GuardedTTA, SafetyControllerConfig
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream


def load_val_stream(checkpoint_path: str, data_dir: str, cache_dir: str, n_leads: int
                    ) -> Tuple[BeatCNN, object]:
    """Loads the real checkpoint + DS1-VAL (NOT DS2) -- the held-out split this
    project's training protocol already reserves for exactly this kind of model-
    selection/tuning decision, never touched by any reported Stage 3/7 test number."""
    from dataset import build_datasets
    from eval_cross_domain import load_checkpoint

    model, cnn_cfg, data_cfg, is_trained = load_checkpoint(checkpoint_path, "cpu", n_leads)
    if not is_trained:
        raise RuntimeError(f"checkpoint at {checkpoint_path} has no trained weights "
                           f"(is_trained=False) -- check the path.")
    data_cfg = ECGDataConfig(**{**asdict(data_cfg), "data_dir": data_dir, "cache_dir": cache_dir})
    ds = build_datasets(data_cfg)["val"]           # DS1_VAL_RECORDS = [115, 203, 209, 230]
    return model, ds


def run_one(source_model: BeatCNN, ds, tau: float, gamma: float, kappa: float,
           n_beats: int, severity_cap: float, severity_start_idx: int, severity_n_ramp: int,
           sim_seed: int, rolling_window: int, device: str) -> dict:
    m = copy.deepcopy(source_model).to(device)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=sim_seed))
    _ramp = linear_ramp(start_idx=severity_start_idx, n_ramp=severity_n_ramp)
    severity_schedule = (lambda i: min(severity_cap, _ramp(i))) if severity_cap < 1.0 else _ramp
    stream = make_beat_stream(ds, n_beats=n_beats, sim=sim, severity=severity_schedule)
    method = GuardedTTA(SafetyControllerConfig(tau=tau, gamma=gamma, kappa=kappa))
    ev = StreamingEvaluator(method, StreamingConfig(device=device, rolling_window=rolling_window, log_every=0))
    res = ev.run(m, stream)
    s = res.summary
    return {"tau": tau, "gamma": gamma, "kappa": kappa, "n_beats": s["n_beats"],
           "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
           "macro_f1_ex_fq": s.get("macro_f1_ex_fq"), "mean_js_div": s["mean_js_div"],
           "adaptation_rate": s["adaptation_rate"], "reset_rate": s["reset_rate"]}


def main(argv: Optional[Sequence[str]] = None) -> List[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--n_beats", type=int, default=3000,
                   help="beats per grid point, drawn from DS1-VAL (not DS2) -- kept small "
                        "since this runs once per (tau,gamma) combo")
    p.add_argument("--severity_start_idx", type=int, default=100)
    p.add_argument("--severity_n_ramp", type=int, default=500)
    p.add_argument("--severity_cap", type=float, default=0.4,
                   help="matches the 'typical mission' profile this project settled on "
                        "for the headline (non-worst-case) result")
    p.add_argument("--rolling_window", type=int, default=300)
    p.add_argument("--sim_seed", type=int, default=0)
    p.add_argument("--taus", type=str, default="0.02,0.05,0.08,0.12")
    p.add_argument("--gammas", type=str, default="0.15,0.3,0.5")
    p.add_argument("--kappa", type=float, default=40.0)
    p.add_argument("--device", default="cpu")   # grid search: many short runs -- GPU launch
                                                 # overhead per N=1 step usually loses to CPU here
    a = p.parse_args(argv)

    taus = [float(x) for x in a.taus.split(",")]
    gammas = [float(x) for x in a.gammas.split(",")]

    print(f"[tune_safety_controller] loading checkpoint + DS1-VAL (held-out, never used for "
         f"any reported test number)...")
    source_model, ds = load_val_stream(a.checkpoint, a.data_dir, a.cache_dir, a.n_leads)
    n_available = len(ds)
    n_beats = min(a.n_beats, n_available)
    print(f"[tune_safety_controller] DS1-VAL has {n_available} beats total; using {n_beats} "
         f"per grid point (severity_cap={a.severity_cap}).")

    if run_one(source_model, ds, taus[0], gammas[0], a.kappa, 1, a.severity_cap,
              a.severity_start_idx, a.severity_n_ramp, a.sim_seed, 1, a.device
              )["macro_f1_ex_fq"] is None:
        raise RuntimeError(
            "streaming_eval.classification_report summary has no 'macro_f1_ex_fq' key -- "
            "this is almost certainly a STALE streaming_eval.py in this Colab session (the "
            "macro_f1_ex_fq fix needs to be re-uploaded). Run `!grep -n \"macro_f1_ex_fq\" "
            "streaming_eval.py` to confirm it's present before re-running this grid search; "
            "aborting now instead of burning the whole grid on a result we can't rank.")

    rows = []
    for gamma in gammas:
        for tau in taus:
            r = run_one(source_model, ds, tau, gamma, a.kappa, n_beats, a.severity_cap,
                       a.severity_start_idx, a.severity_n_ramp, a.sim_seed, a.rolling_window, a.device)
            rows.append(r)
            exfq_s = "n/a" if r["macro_f1_ex_fq"] is None else f"{r['macro_f1_ex_fq']:.3f}"
            print(f"  tau={tau:<6} gamma={gamma:<5} -> acc={r['accuracy']:.3f}  "
                 f"macroF1={r['macro_f1']:.3f}  F1exFQ={exfq_s}  "
                 f"meanD_JS={r['mean_js_div']:.3e}  adapt%={100*r['adaptation_rate']:.1f}")

    rows.sort(key=lambda r: (r["macro_f1_ex_fq"] is not None, r["macro_f1_ex_fq"]), reverse=True)
    print("\n" + "=" * 90)
    print("Ranked by macro_f1_ex_fq (best first) -- all numbers from DS1-VAL, not DS2:")
    print("=" * 90)
    header = f"{'tau':<8}{'gamma':<8}{'acc':<8}{'macroF1':<10}{'F1exFQ':<10}{'meanD_JS':<12}{'adapt%':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['tau']:<8}{r['gamma']:<8}{r['accuracy']:<8.3f}{r['macro_f1']:<10.3f}"
             f"{r['macro_f1_ex_fq']:<10.3f}{r['mean_js_div']:<12.3e}{100*r['adaptation_rate']:<8.1f}")

    best = rows[0]
    print(f"\nBest on DS1-VAL: tau={best['tau']} gamma={best['gamma']} kappa={a.kappa} "
         f"(macro_f1_ex_fq={best['macro_f1_ex_fq']:.3f} vs. the current default tau=0.05/"
         f"gamma=0.3). CONFIRM this on the real DS2 test stream with run_experiments.py "
         f"before reporting it -- these numbers are from the tuning split, not the test split.")
    return rows


# --------------------------------------------------------------------------- #
# Self-test: tiny synthetic grid (3 combos, 300 beats each) exercising the full
# path -- run_one()'s plumbing, sorting, and the DS1-VAL-not-DS2 loading function's
# signature -- without needing real MIT-BIH data. load_val_stream() itself is NOT
# exercised here (it needs a real checkpoint+data); this only checks run_one()
# and the grid/ranking logic, which is what's actually novel in this file.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        import torch.nn.functional as F
        from baselines_tta import _SyntheticStreamDataset

        print("[tune_safety_controller] self-test: tiny synthetic (tau,gamma) grid (no real "
             "data needed) -- exercises run_one()/ranking only, not load_val_stream().")
        torch.manual_seed(0)
        model = BeatCNN(CNNConfig(in_channels=1))
        ds = _SyntheticStreamDataset(n_records=2, beats_per_record=200, c=1, l=256, seed=0)

        opt = torch.optim.Adam(list(model.backbone.parameters())
                               + list(model.adaptable_head.parameters()), lr=1e-3)
        model.train()
        for _ in range(10):
            perm = torch.randperm(len(ds.y))
            for i in range(0, len(ds.y), 32):
                idx = perm[i:i + 32]
                loss = F.cross_entropy(model(ds.x[idx]).logits, ds.y[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        model.freeze_anchor()

        rows = []
        for tau in (0.02, 0.05, 0.1):
            r = run_one(model, ds, tau=tau, gamma=0.3, kappa=40.0, n_beats=300,
                       severity_cap=0.4, severity_start_idx=50, severity_n_ramp=100,
                       sim_seed=0, rolling_window=50, device="cpu")
            rows.append(r)
            print(f"  tau={tau} -> acc={r['accuracy']:.3f} F1exFQ={r['macro_f1_ex_fq']:.3f} "
                 f"meanD_JS={r['mean_js_div']:.3e}")

        assert len(rows) == 3
        for r in rows:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert r["macro_f1_ex_fq"] is not None, "macro_f1_ex_fq must be populated (streaming_eval fix)"
            assert r["mean_js_div"] is not None and r["mean_js_div"] >= 0.0
        # sanity: a larger tau (restoring force ramps up later) should generally show
        # >= drift than a smaller tau at the same gamma, on average across this tiny grid
        # (not asserted strictly beat-for-beat -- D_JS is stochastic -- just directionally)
        by_tau = {r["tau"]: r["mean_js_div"] for r in rows}
        print(f"\nD_JS by tau: {by_tau}")
        print("Self-test passed (plumbing verified; real tuning needs --checkpoint/--data_dir/--cache_dir).")
