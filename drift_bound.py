"""
drift_bound.py  --  a rigorous, checkable bound on this project's parameter drift.

Setting. GuardedTTA adapts the dw_bn affine parameters theta (sparse scope). Per beat:
    v_t   = mu * v_{t-1} + g_t                      (SGD with momentum)
    th'   = th_t - eta * v_t
    th_t+1 = (1 - a_t) * th' + a_t * th_src,   a_t = gamma * sigmoid(kappa * (D_JS,t - tau))
Let delta_t = || th_t - th_src ||_2.

Proposition (bounded drift). Suppose the entropy gradients satisfy ||g_t|| <= G for all t
(G is a property of the model/stream and is MEASURED below, not assumed), and let
    a_min = gamma * sigmoid(-kappa * tau)      (the pull strength at zero output divergence).
Then a_t >= a_min for every t, and for all t
    delta_t <= max( delta_0 , s * (1 - a_min) / a_min ),      s = eta * G / (1 - mu).
Proof. ||v_t|| <= G/(1-mu) by induction (||v_t|| <= mu||v_{t-1}|| + G), so
||th' - th_t|| <= s. Hence delta_{t+1} = (1-a_t)||th' - th_src|| <= (1-a_t)(delta_t + s)
<= (1-a_min)(delta_t + s). The fixed point of x -> (1-a_min)(x+s) is s(1-a_min)/a_min and the
map is a contraction, so delta_t stays below max(delta_0, that fixed point).  QED
Remarks. (i) With the default tau=0.05, kappa=40, gamma=0.3: a_min = 0.3*sigmoid(-2) = 0.0358
and the bound is ~26.9*s. (ii) The elastic part makes the *effective* pull larger than a_min
whenever outputs diverge (a_t -> gamma = 0.3), so the bound is loose in practice; this script
reports how loose. (iii) The bound is in PARAMETER space; output drift (D_JS) is what the
paper's tables report, and is measured, not bounded, here. (iv) A fixed-pull ablation
(fixed_alpha) has the same bound with a_min = fixed_alpha; TENT (a=0) has NO such bound.

  python drift_bound.py --checkpoint ... --data_dir ... --cache_dir ...   # real stream
"""
from __future__ import annotations

import argparse
import copy
import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from safety_controller import GuardedTTA, SafetyControllerConfig
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream


def a_min(cfg: SafetyControllerConfig) -> float:
    return cfg.gamma / (1.0 + math.exp(cfg.kappa * cfg.tau))          # gamma * sigmoid(-kappa*tau)


def measure(model, ds, sim, severity, cfg: Optional[SafetyControllerConfig] = None,
            n_beats: Optional[int] = None) -> Dict[str, float]:
    cfg = cfg or SafetyControllerConfig()
    pg = GuardedTTA(cfg)
    m = copy.deepcopy(model)
    deltas, gnorms = [], []
    orig = pg.step

    def probed(model_, x, meta):
        r = orig(model_, x, meta)
        src = model_._source_tta
        d2, g2 = 0.0, 0.0
        for n, p in model_.tta_named_parameters():
            d2 += float(((p.detach() - src[n].to(p.device)) ** 2).sum())
            if p.grad is not None:
                g2 += float((p.grad ** 2).sum())
        deltas.append(math.sqrt(d2)); gnorms.append(math.sqrt(g2))
        return r

    pg.step = probed
    stream = make_beat_stream(ds, n_beats=n_beats, sim=sim, severity=severity)
    res = StreamingEvaluator(pg, StreamingConfig(device="cpu", rolling_window=200, log_every=0)).run(m, stream)
    G = float(np.max(gnorms)) if gnorms else 0.0
    s = cfg.lr * G / (1.0 - cfg.momentum)
    am = a_min(cfg)
    bound = s * (1.0 - am) / am
    return {"n_beats": len(deltas), "delta_max_observed": float(np.max(deltas)),
            "delta_mean_observed": float(np.mean(deltas)), "G_max": G, "s": s, "a_min": am,
            "bound": bound, "ratio_observed_over_bound": float(np.max(deltas)) / max(bound, 1e-12),
            "mean_js_div": float(res.summary["mean_js_div"])}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    from run_experiments import load_source_model_and_stream
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_beats", type=int, default=0, help="0 = full stream")
    p.add_argument("--leo_cap", type=float, default=0.4)
    a = p.parse_args(argv)
    model, ds, real = load_source_model_and_stream(a.checkpoint, a.data_dir, a.cache_dir, 1, 3, 600, 0)
    ramp = linear_ramp(start_idx=500, n_ramp=2000)
    r = measure(model, ds, SpaceShiftSimulator(SpaceShiftConfig(seed=0)), lambda i: min(a.leo_cap, ramp(i)),
                n_beats=a.n_beats or None)
    print(("REAL DS2" if real else "SYNTHETIC smoke test"), r)
    print(f"max ||theta - theta_src|| observed {r['delta_max_observed']:.4f}  <=  bound {r['bound']:.4f} "
          f"(G_max={r['G_max']:.3f}, a_min={r['a_min']:.4f}); observed/bound = {r['ratio_observed_over_bound']:.3f}")
    return r


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(); sys.exit(0)
    from baselines_tta import _SyntheticStreamDataset
    import torch.nn.functional as F
    from base_cnn import CNNConfig, BeatCNN
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
    torch.manual_seed(0)
    model = BeatCNN(CNNConfig(in_channels=1))
    ds = _SyntheticStreamDataset(n_records=2, beats_per_record=250, c=1, l=256, seed=0)
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()), lr=1e-3)
    model.train()
    for _ in range(8):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            ix = perm[i:i + 32]
            loss = F.cross_entropy(model(ds.x[ix]).logits, ds.y[ix])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); model.freeze_anchor()
    ramp = linear_ramp(start_idx=30, n_ramp=60)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=0))
    for cfg in (SafetyControllerConfig(), SafetyControllerConfig(lr=1e-2)):
        r = measure(model, ds, sim, lambda i: min(0.4, ramp(i)), cfg, n_beats=400)
        print(f"  lr={cfg.lr:g}: observed max delta {r['delta_max_observed']:.5f} <= bound {r['bound']:.5f} "
              f"(G_max={r['G_max']:.4f}, ratio {r['ratio_observed_over_bound']:.3f})")
        assert r["delta_max_observed"] <= r["bound"] + 1e-9, "proposition violated -- proof or code is wrong"
        assert abs(r["a_min"] - cfg.gamma * (1 / (1 + math.exp(cfg.kappa * cfg.tau)))) < 1e-12
    assert abs(a_min(SafetyControllerConfig()) - 0.0358) < 1e-3
    print("Self-test passed (observed drift respects the proven bound).")
