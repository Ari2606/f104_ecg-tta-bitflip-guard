"""diag_vrecall.py -- fast V-recall diagnostic: runs ONLY source_only and this project
(both cheap; no RoTTA/BeatRhythm), seed 0, same protocol as run_experiments.py, then prints
where V beats go, by severity and by stream third.
Usage: python diag_vrecall.py --checkpoint ... --data_dir ... --cache_dir ... --n_beats 20000 --severity_cap 0.4
"""
import argparse, copy
import numpy as np, pandas as pd, torch
from run_experiments import load_source_model_and_stream
from safety_controller import GuardedTTA, SafetyControllerConfig
from baselines_tta import SourceOnlyTTA
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream


def run(model0, ds, method, a):
    m = copy.deepcopy(model0)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=a.seed))
    ramp = linear_ramp(start_idx=a.severity_start_idx, n_ramp=a.severity_n_ramp)
    sched = lambda i: min(a.severity_cap, ramp(i))
    stream = make_beat_stream(ds, n_beats=(a.n_beats or None), sim=sim, severity=sched)
    ev = StreamingEvaluator(method, StreamingConfig(device="cpu", rolling_window=500, log_every=0))
    return ev.run(m, stream)


def report(name, d):
    v = d[d.y_true == 2]
    print(f"\n{name}: V beats={len(v)} recall={np.mean(v.y_pred == 2):.3f}")
    print(" V predicted as (0=N,1=S,2=V,3=F,4=Q):", v.y_pred.value_counts().sort_index().to_dict())
    if v.severity.notna().any():
        vb = v.assign(sev_bin=pd.cut(v.severity.astype(float), [-.01, .01, .2, .4]))
        print(vb.groupby("sev_bin", observed=True).y_pred.agg(
            n="size", recall=lambda s: (s == 2).mean()).round(3).to_string())
    print(" recall by stream third:", [round(float((c == 2).mean()), 3) for c in np.array_split(v.y_pred.to_numpy(), 3)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", required=True); p.add_argument("--n_leads", type=int, default=1)
    p.add_argument("--n_beats", type=int, default=0, help="0 = full stream (all records)")
    p.add_argument("--severity_start_idx", type=int, default=500)
    p.add_argument("--severity_n_ramp", type=int, default=2000)
    p.add_argument("--severity_cap", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["ds2", "val"], default="ds2",
                   help="val = DS1-VAL (use this to choose; DS2 only to confirm once)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    if a.split == "val":
        from tune_safety_controller import load_val_stream
        model, ds = load_val_stream(a.checkpoint, a.data_dir, a.cache_dir, a.n_leads)
        a.n_beats = min(a.n_beats, len(ds)) if a.n_beats else 0; print(f"DS1-VAL, {a.n_beats or len(ds)} beats")
    else:
        model, ds, real = load_source_model_and_stream(a.checkpoint, a.data_dir, a.cache_dir,
                                                       a.n_leads, 3, 600, a.seed)
        print("REAL DS2" if real else "SYNTHETIC")
    variants = [("source_only", SourceOnlyTTA()),
                ("pg_default", GuardedTTA(SafetyControllerConfig())),
                ("pg_skip_abnormal", GuardedTTA(SafetyControllerConfig(adapt_gate="skip_abnormal")))]
    for label, meth in variants:
        r = run(model, ds, meth, a)
        d = pd.DataFrame(r.log)
        d.to_csv(f"diag_{a.split}_{label}.csv", index=False)
        s = r.summary
        print(f"\n=== {label} [{a.split}] acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f} "
              f"F1exFQ={s.get('macro_f1_ex_fq', float('nan')):.3f} D_JS={s['mean_js_div']:.3f} "
              f"adapt_rate={s['adaptation_rate']:.2f}")
        report(label, d)
