"""
tune_baselines.py  --  this project: fair tuning of the competing baselines on DS1-VAL

this project's (tau, gamma) were searched on DS1-VAL (tune_safety_controller.py).
Reviewers will ask for the same courtesy for the baselines, otherwise "your method is
tuned and theirs is not". This script grid-searches each baseline's learning rate (and
RoTTA's EMA momentum) on DS1-VAL -- the held-out, record-disjoint split, NEVER DS2 --
ranking by macro_f1_ex_fq (N/S/V macro-F1; F and Q have only 3 and 5 beats in DS2, so
they cannot be ranked on). It prints the winner per method plus the exact flags to pass
to run_experiments.py / run_multiseed.py, then you confirm ONCE on DS2.

Expect the honest outcome that a baseline's best validation setting may be "adapt as
little as possible" (i.e. close to source-only). That is a legitimate result and should
be reported as such.

Usage:
    !python tune_baselines.py \
        --checkpoint /content/drive/MyDrive/ttaguard_out/checkpoint_stage1_narrow.pt \
        --data_dir /content/drive/MyDrive/ttaguard_data/data/mitdb \
        --cache_dir /content/drive/MyDrive/ttaguard_out/cache \
        --n_beats 3000 --severity_cap 0.4 --methods tent,rotta,bhr
"""
from __future__ import annotations

import argparse
import copy
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from baselines_tta import (BeatRhythmTTA, BeatRhythmTTAConfig, RoTTAConfig, RoTTATTA,
                           TentConfig, TentTTA)
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream
from tune_safety_controller import load_val_stream


def run_method(source_model, ds, method, n_beats: int, severity_cap: float,
              severity_start_idx: int, severity_n_ramp: int, sim_seed: int,
              rolling_window: int, device: str) -> dict:
    m = copy.deepcopy(source_model).to(device)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=sim_seed))
    ramp = linear_ramp(start_idx=severity_start_idx, n_ramp=severity_n_ramp)
    sched = (lambda i: min(severity_cap, ramp(i))) if severity_cap < 1.0 else ramp
    stream = make_beat_stream(ds, n_beats=n_beats, sim=sim, severity=sched)
    ev = StreamingEvaluator(method, StreamingConfig(device=device, rolling_window=rolling_window,
                                                    log_every=0))
    s = ev.run(m, stream).summary
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
            "macro_f1_ex_fq": s.get("macro_f1_ex_fq"), "mean_js_div": s["mean_js_div"]}


def build_grid(args) -> Dict[str, List[Tuple[str, Callable[[], object], str]]]:
    """method -> [(label, factory, flags-to-pass-to-the-runners)]"""
    f = lambda s: [float(v) for v in s.split(",")]
    grid: Dict[str, list] = {}
    if "tent" in args.methods:
        grid["tent"] = [(f"lr={lr:g}", (lambda lr=lr: TentTTA(TentConfig(scope="sparse", lr=lr))),
                         f"--tent_lr {lr:g}") for lr in f(args.tent_lrs)]
    if "rotta" in args.methods:
        grid["rotta"] = [(f"lr={lr:g} ema={ema:g}",
                          (lambda lr=lr, ema=ema: RoTTATTA(RoTTAConfig(
                              lr=lr, ema_alpha=ema, update_every=args.rotta_update_every))),
                          f"--rotta_lr {lr:g} --rotta_ema_alpha {ema:g}")
                         for lr in f(args.rotta_lrs) for ema in f(args.rotta_emas)]
    if "bhr" in args.methods:
        grid["bhr"] = [(f"lr={lr:g} steps={args.bhr_steps}",
                        (lambda lr=lr: BeatRhythmTTA(BeatRhythmTTAConfig(lr=lr, steps=args.bhr_steps))),
                        f"--bhr_lrs {lr:g} --bhr_steps {args.bhr_steps}") for lr in f(args.bhr_lrs)]
    return grid


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--n_beats", type=int, default=3000)
    p.add_argument("--severity_start_idx", type=int, default=100)
    p.add_argument("--severity_n_ramp", type=int, default=500)
    p.add_argument("--severity_cap", type=float, default=0.4)
    p.add_argument("--rolling_window", type=int, default=300)
    p.add_argument("--sim_seed", type=int, default=0)
    p.add_argument("--methods", type=str, default="tent,rotta,bhr")
    p.add_argument("--tent_lrs", type=str, default="1e-5,1e-4,1e-3")
    p.add_argument("--rotta_lrs", type=str, default="1e-4,1e-3")
    p.add_argument("--rotta_emas", type=str, default="0.99,0.999")
    p.add_argument("--rotta_update_every", type=int, default=1)
    p.add_argument("--bhr_lrs", type=str, default="1e-5,1e-4,1e-3")
    p.add_argument("--bhr_steps", type=int, default=5)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    a.methods = [m.strip() for m in a.methods.split(",")]

    print("[tune_baselines] loading checkpoint + DS1-VAL (held-out; never DS2)...")
    model, ds = load_val_stream(a.checkpoint, a.data_dir, a.cache_dir, a.n_leads)
    n_beats = min(a.n_beats, len(ds))
    print(f"[tune_baselines] DS1-VAL has {len(ds)} beats; using {n_beats} per grid point "
         f"(severity_cap={a.severity_cap}).")

    out: Dict[str, List[dict]] = {}
    for name, combos in build_grid(a).items():
        rows = []
        for label, factory, flags in combos:
            r = run_method(model, ds, factory(), n_beats, a.severity_cap, a.severity_start_idx,
                          a.severity_n_ramp, a.sim_seed, a.rolling_window, a.device)
            r.update(label=label, flags=flags)
            rows.append(r)
            print(f"  [{name}] {label:<22} acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} "
                 f"F1exFQ={r['macro_f1_ex_fq']:.3f} meanD_JS={r['mean_js_div']:.3e}", flush=True)
        rows.sort(key=lambda r: r["macro_f1_ex_fq"], reverse=True)
        out[name] = rows
        print(f"\n[{name}] ranked by macro_f1_ex_fq (DS1-VAL):")
        for r in rows:
            print(f"    {r['label']:<22} F1exFQ={r['macro_f1_ex_fq']:.3f} acc={r['accuracy']:.3f} "
                 f"D_JS={r['mean_js_div']:.3e}")
        print(f"[{name}] BEST on DS1-VAL: {rows[0]['label']}  ->  flags: {rows[0]['flags']}\n")

    print("=" * 90)
    print("Flags to pass to run_experiments.py / run_multiseed.py (confirm ONCE on DS2):")
    for name, rows in out.items():
        print(f"  {name:<6} {rows[0]['flags']}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        import torch.nn.functional as F
        from base_cnn import CNNConfig, BeatCNN
        from baselines_tta import _SyntheticStreamDataset

        print("[tune_baselines] self-test: tiny synthetic grid (no real data) -- exercises "
             "run_method()/grid/ranking only, not load_val_stream().")
        torch.manual_seed(0)
        model = BeatCNN(CNNConfig(in_channels=1))
        ds = _SyntheticStreamDataset(n_records=2, beats_per_record=150, c=1, l=256, seed=0)
        opt = torch.optim.Adam(list(model.backbone.parameters())
                               + list(model.adaptable_head.parameters()), lr=1e-3)
        model.train()
        for _ in range(8):
            perm = torch.randperm(len(ds.y))
            for i in range(0, len(ds.y), 32):
                idx = perm[i:i + 32]
                loss = F.cross_entropy(model(ds.x[idx]).logits, ds.y[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval(); model.freeze_anchor()

        args = argparse.Namespace(methods=["tent", "rotta", "bhr"], tent_lrs="1e-4,1e-3",
                                  rotta_lrs="1e-3", rotta_emas="0.99,0.999", rotta_update_every=4,
                                  bhr_lrs="1e-4", bhr_steps=1)
        grid = build_grid(args)
        assert set(grid) == {"tent", "rotta", "bhr"}
        assert [len(v) for v in (grid["tent"], grid["rotta"], grid["bhr"])] == [2, 2, 1]
        for name, combos in grid.items():
            rows = []
            for label, factory, flags in combos:
                r = run_method(model, ds, factory(), n_beats=200, severity_cap=0.4,
                              severity_start_idx=30, severity_n_ramp=60, sim_seed=0,
                              rolling_window=50, device="cpu")
                assert r["macro_f1_ex_fq"] is not None and 0.0 <= r["accuracy"] <= 1.0
                assert flags.startswith("--"), flags
                rows.append(r)
                print(f"  [{name}] {label:<22} acc={r['accuracy']:.3f} F1exFQ={r['macro_f1_ex_fq']:.3f} "
                     f"D_JS={r['mean_js_div']:.3e}  ({flags})")
        print("Self-test passed (plumbing verified; real tuning needs --checkpoint/--data_dir/--cache_dir).")
