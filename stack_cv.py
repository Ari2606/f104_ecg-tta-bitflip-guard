"""
stack_cv.py  --  is S limited by the CNN, or by the information in RR timing? Can a tiny tree stacker add a few points?

Uses the fold checkpoints cv_sweep.py already trained (each fold's model never saw that fold's records), so every
feature below is OUT-OF-FOLD for its records. For each held-out fold f, a gradient-boosted classifier is trained on the
OTHER folds' out-of-fold features and scored on fold f (nested, no leakage). Three feature sets, pooled over folds:
  cnn    : the CNN's own N/S/V logits, scored directly (no stacker) -- the baseline;
  rr     : the 7 RR features only (tree);
  stack  : RR features + CNN logits (tree).
3-class scoring (F/Q beats dropped, labelled).  Reports S / V average precision and Se/PPV/F1 at argmax.

  python stack_cv.py --variant base --ckpt_dir .../cv_sweep --data_dir ... --cache_dir ... [--k 4]
Needs scikit-learn.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def ap_and_prf(score_S, score_V, pred, y) -> Dict:
    from operating_points import average_precision
    out = {}
    for c, nm, sc in ((1, "S", score_S), (2, "V", score_V)):
        tp = int(((pred == c) & (y == c)).sum())
        se, ppv = tp / max((y == c).sum(), 1), tp / max((pred == c).sum(), 1)
        out[nm] = {"AP": average_precision((y == c).astype(int), sc), "Se": 100 * se, "PPV": 100 * ppv,
                   "F1": 100 * 2 * se * ppv / max(se + ppv, 1e-12)}
    return out


def stack_eval(feats: List[Dict], seed: int = 0) -> Dict[str, Dict]:
    """feats[i] = {'rr': [n,d], 'lg': [n,5], 'y': [n]} for fold i (out-of-fold). Nested leave-one-fold-out."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from operating_points import ovr_score
    res: Dict[str, List] = {"cnn": [], "rr": [], "stack": []}
    Y = []
    for i, te in enumerate(feats):
        keep_te = te["y"] <= 2
        y_te = te["y"][keep_te]
        Y.append(y_te)
        lg_te = te["lg"][keep_te][:, :3]
        res["cnn"].append((ovr_score(lg_te, 1), ovr_score(lg_te, 2), lg_te.argmax(1)))
        trn = [f for j, f in enumerate(feats) if j != i]
        ytr = np.concatenate([f["y"] for f in trn]); k = ytr <= 2
        for name in ("rr", "stack"):
            def X(f, m):
                a = f["rr"][m]
                return a if name == "rr" else np.concatenate([a, f["lg"][m][:, :3]], 1)
            Xtr = np.concatenate([X(f, f["y"] <= 2) for f in trn]); ytr3 = ytr[k]
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0,
                                                 random_state=seed, class_weight="balanced").fit(Xtr, ytr3)
            P = clf.predict_proba(X(te, keep_te))
            res[name].append((P[:, 1], P[:, 2], P.argmax(1)))
    y = np.concatenate(Y)
    return {k: ap_and_prf(np.concatenate([r[0] for r in v]), np.concatenate([r[1] for r in v]),
                          np.concatenate([r[2] for r in v]), y) for k, v in res.items()}


def render(res: Dict[str, Dict]) -> str:
    L = ["| features | S AP | S Se/PPV/F1 | V AP | V Se/PPV/F1 |", "|---|---|---|---|---|"]
    for k in ("cnn", "rr", "stack"):
        s, v = res[k]["S"], res[k]["V"]
        L.append(f"| {k} | {s['AP']:.3f} | {s['Se']:.1f}/{s['PPV']:.1f}/{s['F1']:.1f} | {v['AP']:.3f} | "
                 f"{v['Se']:.1f}/{v['PPV']:.1f}/{v['F1']:.1f} |")
    return "\n".join(L) + "\n\nOut-of-fold on DS1, 3-class (F/Q dropped), nested leave-one-fold-out stacker. DS2 untouched. " \
           "Differences below ~0.03 AP are noise.\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", default="base")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    from cv_folds import assign_folds, record_weights
    from dataset import ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    from operating_points import logits_of
    folds = assign_folds(record_weights(a.data_dir, a.cache_dir), a.k)
    feats = []
    for i, f in enumerate(folds):
        model, _, dcfg, ok = load_checkpoint(str(Path(a.ckpt_dir) / f"{a.variant}_fold{i}.pt"), "cpu", 1)
        assert ok
        cfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
        ds = build_datasets(cfg, val_records=f)["val"]
        assert ds.rr.shape[-1] > 0, "needs RR features"
        feats.append({"rr": ds.rr.numpy(), "lg": logits_of(model, ds), "y": ds.y.numpy()})
        print(f"[stack_cv] fold {i}: {len(ds)} beats", flush=True)
    md = render(stack_eval(feats))
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    rng = np.random.default_rng(0)                      # synthetic: RR carries S signal, CNN logits are noisy
    feats = []
    for _ in range(4):
        n = 3000
        y = rng.choice(5, n, p=[.8, .06, .1, .03, .01])
        rr = rng.normal(size=(n, 7)).astype(np.float32); rr[y == 1, 5] -= 1.5
        lg = rng.normal(size=(n, 5)); lg[np.arange(n), y] += 0.8
        feats.append({"rr": rr, "lg": lg, "y": y})
    r = stack_eval(feats)
    assert all(0 < r[k]["S"]["AP"] <= 1 for k in r)
    assert r["stack"]["S"]["AP"] > r["cnn"]["S"]["AP"]          # stacker must exploit the RR signal
    print(render(r)); print("Self-test passed.")
