"""
tune_cv.py  --  choose hyper-parameters on the average over leave-record-group-out folds.

For each fold (folds.json from cv_folds.py): a source checkpoint that never saw the fold's
DS1 records, evaluated on those held-out records through the SAME streaming protocol as the
test benchmark (N=1, chronological, predict-then-adapt, LEO shift at cap --leo_cap).
Each grid point is scored by the MEAN over folds of macro-F1 (N/S/V) -- every DS1 record is
held out exactly once -- and the per-fold scores and their spread are printed, so a setting
that only wins on one fold is visible as such. DS2 is never read here.

The grid is a JSON dict {method_key: {hp_field: [values...]}} over method_zoo keys; the
cartesian product per method is run. The unmodified default is always included as the
reference point ("default"), so "tuning helped" is a measured statement, not an assumption.

  python tune_cv.py --folds_json folds.json --data_dir ... --cache_dir ... \
      --grid '{"pg":{"tau":[0.05,0.1],"gamma":[0.15,0.3]},"rotta":{"lr":[3e-4,1e-3]}}' \
      --max_beats_per_record 1500 --out_dir ./results/tune_cv
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import method_zoo
from analyze_results import metrics_from_cms
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream

K = 5


def cap_per_record(ds, max_beats: Optional[int]):
    """Duck-typed copy keeping only the first `max_beats` (chronological) beats per record,
    so every record contributes and the adaptation dynamics inside a record are preserved."""
    if not max_beats:
        return ds
    rid, smp = ds.record_ids.numpy(), ds.sample_idx.numpy()
    keep = []
    for r in np.unique(rid):
        ix = np.where(rid == r)[0]
        keep.append(ix[np.argsort(smp[ix])][:max_beats])
    keep = np.sort(np.concatenate(keep))
    rr = getattr(ds, "rr", None)
    return SimpleNamespace(x=ds.x[keep], y=ds.y[keep], record_ids=ds.record_ids[keep],
                           sample_idx=ds.sample_idx[keep],
                           rr=(rr[keep] if rr is not None else None))


def score_fold(model, ds, key: str, hp: dict, leo_cap: float, start_idx: int, n_ramp: int,
               seed: int = 0) -> dict:
    meth = method_zoo.build_method(key, {key: hp} if hp else {})
    if method_zoo.needs_source_beats(key):
        rr = getattr(ds, "rr", None)
        meth.set_fisher_data(ds.x[:300], rr[:300] if rr is not None and rr.shape[-1] > 0 else None)
    m = copy.deepcopy(model)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=seed))
    ramp = linear_ramp(start_idx=start_idx, n_ramp=n_ramp)
    stream = make_beat_stream(ds, n_beats=None, sim=sim, severity=lambda i: min(leo_cap, ramp(i)))
    res = StreamingEvaluator(meth, StreamingConfig(device="cpu", rolling_window=500, log_every=0)).run(m, stream)
    y = np.array([r["y_true"] for r in res.log]); p = np.array([r["y_pred"] for r in res.log])
    cm = np.zeros((K, K), dtype=np.int64); np.add.at(cm, (y, p), 1)
    mt = {k: float(v[0]) for k, v in metrics_from_cms(cm[None]).items()}
    mt["drift"] = float(res.summary["mean_js_div"])
    return mt


def expand_grid(grid: Dict[str, Dict[str, list]]) -> List[tuple]:
    """-> [(method_key, label, hp_dict)], including ('key','default',{}) for every method."""
    out = []
    for key, fields in grid.items():
        out.append((key, "default", {}))
        names = list(fields)
        for vals in itertools.product(*[fields[n] for n in names]):
            hp = dict(zip(names, vals))
            out.append((key, " ".join(f"{n}={v:g}" if isinstance(v, (int, float)) else f"{n}={v}"
                                      for n, v in hp.items()), hp))
    return out


def tune(folds: List[tuple], grid: Dict[str, Dict[str, list]], leo_cap: float, start_idx: int,
         n_ramp: int, verbose: bool = True) -> Dict[str, List[dict]]:
    """folds: [(model, ds)]. Returns {method: [rows sorted by mean macro_f1_ex_fq desc]}."""
    results: Dict[str, List[dict]] = {}
    for key, label, hp in expand_grid(grid):
        per = [score_fold(m, d, key, hp, leo_cap, start_idx, n_ramp) for m, d in folds]
        f1 = np.array([x["macro_f1_ex_fq"] for x in per])
        row = {"label": label, "hp": hp, "mean_f1_exfq": float(f1.mean()), "sd_folds": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
               "per_fold": f1.tolist(), "mean_recall_V": float(np.mean([x["recall_V"] for x in per])),
               "mean_recall_S": float(np.mean([x["recall_S"] for x in per])),
               "mean_drift": float(np.mean([x["drift"] for x in per])),
               "mean_acc": float(np.mean([x["accuracy"] for x in per]))}
        results.setdefault(key, []).append(row)
        if verbose:
            print(f"  [{key}] {label:<28} F1(N/S/V)={row['mean_f1_exfq']:.3f}+-{row['sd_folds']:.3f} "
                  f"V-rec={row['mean_recall_V']:.3f} drift={row['mean_drift']:.3f} folds={np.round(f1, 3).tolist()}", flush=True)
    for key in results:
        results[key].sort(key=lambda r: -r["mean_f1_exfq"])
    return results


def load_folds(folds_json: str, data_dir: str, cache_dir: str, max_beats_per_record: Optional[int]):
    import dataclasses
    from dataset import ECGDataConfig, MITBIHDataset
    from eval_cross_domain import load_checkpoint
    spec = json.loads(Path(folds_json).read_text())["folds"]
    out = []
    for fo in spec:
        model, _, dcfg, trained = load_checkpoint(fo["checkpoint"], "cpu", 1)
        assert trained, fo["checkpoint"]
        dcfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": data_dir, "cache_dir": cache_dir})
        ds = MITBIHDataset(fo["val_records"], dcfg)
        out.append((model, cap_per_record(ds, max_beats_per_record)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--folds_json", required=True)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--grid", required=True)
    p.add_argument("--leo_cap", type=float, default=0.4)
    p.add_argument("--leo_start_idx", type=int, default=200)
    p.add_argument("--leo_n_ramp", type=int, default=800)
    p.add_argument("--max_beats_per_record", type=int, default=1500)
    p.add_argument("--out_dir", default="./results/tune_cv")
    a = p.parse_args(argv)
    folds = load_folds(a.folds_json, a.data_dir, a.cache_dir, a.max_beats_per_record)
    print(f"[tune_cv] {len(folds)} folds, held-out beats per fold: {[len(d.y) for _, d in folds]}")
    res = tune(folds, json.loads(a.grid), a.leo_cap, a.leo_start_idx, a.leo_n_ramp)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "tune_cv.json").write_text(json.dumps(res, indent=1))
    print("\n=== best per method (mean over folds of macro-F1 N/S/V) ===")
    for key, rows in res.items():
        d = next(r for r in rows if r["label"] == "default")
        b = rows[0]
        gain = b["mean_f1_exfq"] - d["mean_f1_exfq"]
        verdict = "(default is best)" if b["label"] == "default" else f"gain {gain:+.3f}"
        print(f"{key:<14} best: {b['label']:<28} {b['mean_f1_exfq']:.3f}   default: {d['mean_f1_exfq']:.3f}   {verdict}")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(); sys.exit(0)
    from baselines_tta import _SyntheticStreamDataset
    import torch.nn.functional as F
    from base_cnn import CNNConfig, BeatCNN
    folds = []
    for s in range(2):                                              # two synthetic "folds"
        torch.manual_seed(s)
        model = BeatCNN(CNNConfig(in_channels=1))
        ds = _SyntheticStreamDataset(n_records=3, beats_per_record=150, c=1, l=256, seed=s)
        opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()), lr=1e-3)
        model.train()
        for _ in range(6):
            perm = torch.randperm(len(ds.y))
            for i in range(0, len(ds.y), 32):
                ix = perm[i:i + 32]
                loss = F.cross_entropy(model(ds.x[ix]).logits, ds.y[ix])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval(); model.freeze_anchor()
        folds.append((model, cap_per_record(ds, 100)))
    assert all(len(np.unique(d.record_ids.numpy())) == 3 and len(d.y) <= 300 for _, d in folds)
    g = expand_grid({"pg": {"tau": [0.05, 0.1], "gamma": [0.15]}, "tent": {"lr": [1e-3]}})
    assert [(k, l) for k, l, _ in g] == [("pg", "default"), ("pg", "tau=0.05 gamma=0.15"), ("pg", "tau=0.1 gamma=0.15"),
                                         ("tent", "default"), ("tent", "lr=0.001")]
    res = tune(folds, {"pg": {"tau": [0.05, 0.1]}, "tent": {"lr": [1e-3]}}, 0.4, 20, 40)
    assert set(res) == {"pg", "tent"} and len(res["pg"]) == 3
    assert all(len(r["per_fold"]) == 2 for rows in res.values() for r in rows)
    assert res["pg"][0]["mean_f1_exfq"] >= res["pg"][-1]["mean_f1_exfq"], "rows sorted best-first"
    print("Self-test passed.")
