"""
leak_check.py  --  is the extra training database (SVDB) free of near-duplicates of DS2 beats?

For a random subset of DS2 beats (all S and V beats plus normals), find each beat's best normalised
cross-correlation (cosine similarity of mean-removed, unit-norm windows, channel 0) against
  (a) the extra database  (SVDB), and
  (b) DS1, which is patient-disjoint from DS2 by the protocol  (the control).
If SVDB shared patients/recordings with DS2, (a) would show many near-1.0 matches that (b) lacks.
Reports quantiles of the best-match similarity and the fraction above 0.99 / 0.995 for both.

  python leak_check.py --data_dir ... --cache_dir ... --extra_dir .../svdb --out_dir ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch


def unit(x: np.ndarray) -> torch.Tensor:
    x = x[:, 0, :].astype(np.float32)
    x = x - x.mean(1, keepdims=True)
    return torch.from_numpy(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8))


def best_match(q: torch.Tensor, ref: torch.Tensor, chunk: int = 2048) -> np.ndarray:
    out = []
    for i in range(0, len(q), chunk):
        out.append((q[i:i + chunk] @ ref.T).max(1).values)
    return torch.cat(out).numpy()


def summarise(s: np.ndarray) -> Dict:
    return {"median": float(np.median(s)), "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99)),
            "max": float(s.max()), "frac>0.99": float((s > 0.99).mean()), "frac>0.995": float((s > 0.995).mean())}


def compare(q: torch.Tensor, ref_extra: torch.Tensor, ref_ds1: torch.Tensor) -> Dict:
    return {"vs_extra": summarise(best_match(q, ref_extra)), "vs_DS1_control": summarise(best_match(q, ref_ds1))}


def render(r: Dict) -> str:
    L = ["| reference set | median | p95 | p99 | max | frac>0.99 | frac>0.995 |", "|---|---|---|---|---|---|---|"]
    for k, nm in (("vs_extra", "extra database (SVDB)"), ("vs_DS1_control", "DS1 (control, disjoint)")):
        s = r[k]
        L.append(f"| {nm} | {s['median']:.3f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['max']:.3f} | "
                 f"{100 * s['frac>0.99']:.2f}% | {100 * s['frac>0.995']:.2f}% |")
    return "\n".join(L) + "\n\nBest-match cosine similarity of DS2 beats to each reference set. If the extra database " \
           "does not match DS2 more closely than the disjoint DS1 control, there is no sign of shared patients.\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--extra_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_normal", type=int, default=4000)
    a = ap.parse_args(argv)
    from dataset import DS1_RECORDS, DS2_RECORDS, ECGDataConfig, MITBIHDataset
    ids = sorted(int(f.stem) for f in Path(a.extra_dir).glob("*.hea") if f.stem.isdigit())
    cfg = ECGDataConfig(data_dir=a.data_dir, cache_dir=a.cache_dir, n_rr_features=7, n_template=0,
                        extra_dir=a.extra_dir, extra_records=tuple(ids))
    ds2 = MITBIHDataset(DS2_RECORDS, cfg)
    ds1 = MITBIHDataset(DS1_RECORDS, cfg)
    ex = MITBIHDataset(ids, cfg)
    y = ds2.y.numpy()
    rng = np.random.default_rng(0)
    pick = np.concatenate([np.flatnonzero(y != 0), rng.choice(np.flatnonzero(y == 0), a.n_normal, replace=False)])
    q = unit(ds2.x.numpy()[pick])
    n_ref = min(len(ds1), len(ex))                               # same reference size for a fair control
    r1 = unit(ds1.x.numpy()[rng.choice(len(ds1), n_ref, replace=False)])
    r2 = unit(ex.x.numpy()[rng.choice(len(ex), n_ref, replace=False)])
    res = compare(q, r2, r1)
    res["n_query"], res["n_ref_each"] = int(len(pick)), int(n_ref)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "leak_check.json").write_text(json.dumps(res, indent=1))
    md = render(res) + f"\n{len(pick)} DS2 beats (all S/V/F/Q + normals) vs {n_ref} reference beats each.\n"
    (out / "leak_check.md").write_text(md); print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 1, 256)).astype(np.float32)
    ds1 = rng.normal(size=(1000, 1, 256)).astype(np.float32)
    leaky = np.concatenate([rng.normal(size=(500, 1, 256)).astype(np.float32), base + 0.01 * rng.normal(size=base.shape).astype(np.float32)])
    clean = rng.normal(size=(800, 1, 256)).astype(np.float32)
    q = unit(base)
    rl = compare(q, unit(leaky), unit(ds1)); rc = compare(q, unit(clean), unit(ds1))
    assert rl["vs_extra"]["frac>0.99"] > 0.9 and rl["vs_DS1_control"]["frac>0.99"] < 0.01     # leak is detected
    assert rc["vs_extra"]["frac>0.99"] < 0.01                                                  # no false alarm
    print(render(rl)); print("Self-test passed.")
