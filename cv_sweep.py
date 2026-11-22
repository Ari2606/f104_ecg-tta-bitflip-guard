"""
cv_sweep.py  --  which training change actually helps S/V, judged ONLY on held-out DS1 patients (DS2 is never touched).

Why this exists. The model we have was selected on 4 validation records using a noisy 5-class macro-F1 and
trained on ~80% of DS1. Every candidate fix (long-window RR, drop class Q, noise/rare augmentation, 2 leads,
wider net, loss/sampler, learning rate) is tried here under ONE protocol:
  * K record-disjoint folds over the 22 DS1 records (rare-class-balanced, cv_folds.assign_folds);
  * each variant trains one model per fold on the other folds, FIXED epochs, last EMA weights kept
    (no early stopping, no best-epoch pick -> nothing is selected on the held-out records);
  * out-of-fold logits are pooled; we report S and V average precision (ranking quality), and
    Se / PPV / F1 at the default argmax.
Pick the winner here, then train it on all of DS1 and evaluate DS2 once.

Resumable: a finished fold checkpoint is skipped.  --dry prints the commands without running them.

  !python cv_sweep.py --data_dir .../mitdb --cache_dir .../cache --nstdb_dir .../nstdb \
        --ckpt_dir .../cv_sweep --out_dir .../results/cv_sweep [--variants base,q0,aug] [--k 4] [--epochs 30]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

COMMON = "--n_rr_features 7 --fixed_epochs --no_high_accuracy_checkpoint"
# name -> extra train_stage1 flags (later flags override COMMON). Q-weight is zeroed in all but `base`.
VARIANTS: Dict[str, str] = {
    "base":   "--narrow",
    "q0":     "--narrow --zero_q_weight",
    "aug":    "--narrow --zero_q_weight --corrupt_aug 0.5 --nstdb_dir {nstdb}",
    "rare":   "--narrow --zero_q_weight --use_rare_class_augment",
    "lead2":  "--narrow --zero_q_weight --n_leads 2",
    "wide":   "--zero_q_weight",                                   # SE + wide net (~52K params)
    "lr1e3":  "--narrow --zero_q_weight --lr 1e-3",
    "tmpl":   "--narrow --n_template 16",
    "tmpl_wide": "--n_template 16",
    "bsamp":  "--narrow --zero_q_weight --loss_type focal --balanced_sampling",
}


def pooled_metrics(logits: np.ndarray, y: np.ndarray, per_fold: List[tuple]) -> dict:
    from operating_points import average_precision, ovr_score
    pred = logits.argmax(1)
    out = {"n": int(len(y))}
    for c, nm in ((1, "S"), (2, "V")):
        tp = int(((pred == c) & (y == c)).sum())
        se = tp / max(int((y == c).sum()), 1)
        ppv = tp / max(int((pred == c).sum()), 1)
        out[nm] = {"AP": average_precision((y == c).astype(int), ovr_score(logits, c)),
                   "Se": 100 * se, "PPV": 100 * ppv,
                   "F1": 100 * 2 * se * ppv / max(se + ppv, 1e-12)}
        fold_ap = [average_precision((fy == c).astype(int), ovr_score(fl, c)) for fl, fy in per_fold if (fy == c).any()]
        out[nm]["AP_fold_mean"], out[nm]["AP_fold_sd"] = float(np.mean(fold_ap)), float(np.std(fold_ap))
    return out


def evaluate_variant(ckpts: List[str], folds: List[List[int]], data_dir: str, cache_dir: str) -> dict:
    from dataset import ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    from operating_points import logits_of
    L, Y, per_fold = [], [], []
    for ck, val_recs in zip(ckpts, folds):
        model, _, data_cfg, trained = load_checkpoint(ck, "cpu", 1)
        assert trained, ck
        cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": data_dir, "cache_dir": cache_dir})
        ds = build_datasets(cfg, val_records=val_recs)["val"]
        lg, yy = logits_of(model, ds), ds.y.numpy()
        L.append(lg); Y.append(yy); per_fold.append((lg, yy))
    return pooled_metrics(np.concatenate(L), np.concatenate(Y), per_fold)


def render(res: Dict[str, dict]) -> str:
    rows = ["| variant | S AP (pooled) | S AP fold mean±sd | S Se/PPV/F1 | V AP | V Se/PPV/F1 |", "|---|---|---|---|---|---|"]
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["S"]["AP"]):
        s, v = r["S"], r["V"]
        rows.append(f"| {name} | {s['AP']:.3f} | {s['AP_fold_mean']:.3f}±{s['AP_fold_sd']:.3f} | "
                    f"{s['Se']:.1f}/{s['PPV']:.1f}/{s['F1']:.1f} | {v['AP']:.3f} | {v['Se']:.1f}/{v['PPV']:.1f}/{v['F1']:.1f} |")
    return "\n".join(rows) + ("\n\nOut-of-fold on DS1 only (DS2 untouched). One seed per fold; fold-to-fold sd is large, so treat "
                              "differences below ~0.03 AP as noise.\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--nstdb_dir", default="")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)

    from cv_folds import assign_folds, record_weights
    folds = assign_folds(record_weights(a.data_dir, a.cache_dir), a.k)
    print(f"[cv_sweep] folds: {folds}", flush=True)
    ck_dir, out = Path(a.ckpt_dir), Path(a.out_dir)
    ck_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    names = [v.strip() for v in a.variants.split(",")]
    for nm in names:
        if nm not in VARIANTS:
            raise SystemExit(f"unknown variant {nm}; choose from {list(VARIANTS)}")
        if "{nstdb}" in VARIANTS[nm] and not a.nstdb_dir:
            raise SystemExit(f"variant {nm} needs --nstdb_dir")

    for nm in names:
        for i, f in enumerate(folds):
            ck = ck_dir / f"{nm}_fold{i}.pt"
            if ck.exists():
                continue
            flags = f"{COMMON} --epochs {a.epochs} " + VARIANTS[nm].format(nstdb=a.nstdb_dir)
            cmd = [sys.executable, "train_stage1.py", "--data_dir", a.data_dir, "--cache_dir", a.cache_dir,
                   *shlex.split(flags), "--seed", str(a.seed), "--val_records", ",".join(map(str, f)),
                   "--checkpoint_path", str(ck)]
            if a.dry:
                print(" ".join(cmd)); continue
            print(f"[cv_sweep] training {nm} fold {i} (held out {f})", flush=True)
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (out / f"{nm}_fold{i}.log").write_text(r.stdout)
            if r.returncode != 0 or not ck.exists():
                print(r.stdout[-1500:]); raise SystemExit(f"training failed: {nm} fold {i}")
    if a.dry:
        return 0
    res = {}
    for nm in names:
        ckpts = [str(ck_dir / f"{nm}_fold{i}.pt") for i in range(len(folds))]
        res[nm] = evaluate_variant(ckpts, folds, a.data_dir, a.cache_dir)
        print(f"[cv_sweep] evaluated {nm}", flush=True)
    (out / "cv_sweep.json").write_text(json.dumps(res, indent=1))
    md = render(res)
    (out / "cv_sweep.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    rng = np.random.default_rng(0)           # self-test of the metric code on synthetic pooled logits
    y = rng.choice(5, 4000, p=[.8, .06, .1, .03, .01])
    lg = rng.normal(size=(4000, 5)); lg[np.arange(4000), y] += 2.0
    pf = [(lg[:2000], y[:2000]), (lg[2000:], y[2000:])]
    m = pooled_metrics(lg, y, pf)
    assert 0.3 < m["S"]["AP"] <= 1 and 0 < m["V"]["F1"] <= 100
    assert "base" in render({"base": m}) and set(VARIANTS) >= {"base", "q0", "aug"}
    print("Self-test passed.")
