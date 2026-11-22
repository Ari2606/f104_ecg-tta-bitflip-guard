"""
operating_points.py  --  is the S/V gap to the literature a MODEL gap or just an OPERATING-POINT gap?

The default argmax decision trades S sensitivity against S precision at one fixed point. Published
systems sit at other points (e.g. SVEB Se 91 / PPV 42). This script, for each checkpoint on the clean full DS2:
  * computes threshold-free average precision (AP) of S-vs-rest and V-vs-rest (the model's ranking quality);
  * picks a score threshold on the VALIDATION split (DS1 patients, never DS2) that reaches a target
    sensitivity, applies it unchanged to DS2, and reports test Se / PPV / F1. No test-set tuning.
Decision rule per beat: predict S (or V) if its one-vs-rest score exceeds the threshold, else the
argmax class. The rule only ADDS S/V predictions, so targets below the default sensitivity are not reachable.
Diagnostic only: any operating point we would REPORT must come from this validation rule.

--ensemble also evaluates the average of the listed checkpoints (probabilities averaged, then log), as one extra row.
Checkpoints must share the same data config (same val/test beats).

  !python operating_points.py --checkpoint a.pt,b.pt --data_dir .../mitdb --cache_dir .../cache --out_dir .../results/opts [--ensemble]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from dataset import ECGDataConfig, build_datasets
from eval_cross_domain import load_checkpoint

TARGET_SE = (0.76, 0.85, 0.91)
NAMES = {1: "S", 2: "V"}


@torch.no_grad()
def logits_of(model, ds, bs: int = 512) -> np.ndarray:
    model.eval()
    use_aux = model.cfg.aux_feat_dim > 0
    out = []
    for i in range(0, len(ds), bs):
        x = ds.x[i:i + bs]
        aux = ds.rr[i:i + bs] if use_aux else None
        out.append(model(x, aux=aux).logits.cpu().numpy())
    return np.concatenate(out)


def log_softmax(l: np.ndarray) -> np.ndarray:
    m = l.max(1, keepdims=True)
    return l - (m[:, 0] + np.log(np.exp(l - m).sum(1)))[:, None]


def ensemble_logits(list_of_logits) -> np.ndarray:
    p = np.mean([np.exp(log_softmax(l)) for l in list_of_logits], axis=0)
    return np.log(np.clip(p, 1e-12, None))


def ovr_score(logits: np.ndarray, c: int) -> np.ndarray:
    m = logits.max(1, keepdims=True)
    others = np.delete(logits, c, axis=1)
    mo = others.max(1, keepdims=True)
    lse_oth = mo[:, 0] + np.log(np.exp(others - mo).sum(1))
    return logits[:, c] - lse_oth


def average_precision(y_bin: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    yb = y_bin[order]
    tp = np.cumsum(yb)
    prec = tp / np.arange(1, len(yb) + 1)
    return float((prec * yb).sum() / max(yb.sum(), 1))


def threshold_for_se(y_bin: np.ndarray, score: np.ndarray, se: float) -> float:
    pos = np.sort(score[y_bin == 1])[::-1]
    k = max(int(np.ceil(se * len(pos))) - 1, 0)
    return float(pos[k])


def apply(logits: np.ndarray, c: int, thr: float, y: np.ndarray):
    pred = logits.argmax(1).copy()
    pred[ovr_score(logits, c) >= thr] = c
    tp = int(((pred == c) & (y == c)).sum())
    se = tp / max(int((y == c).sum()), 1)
    ppv = tp / max(int((pred == c).sum()), 1)
    f1 = 2 * se * ppv / (se + ppv) if se + ppv > 0 else 0.0
    return 100 * se, 100 * ppv, 100 * f1


def evaluate(name: str, lv: np.ndarray, lt: np.ndarray, yv: np.ndarray, yt: np.ndarray) -> dict:
    r = {"ckpt": name, "n_val": int(len(yv)), "n_test": int(len(yt))}
    pred = lt.argmax(1)
    for c, nm in NAMES.items():
        sv, st = ovr_score(lv, c), ovr_score(lt, c)
        r[f"AP_{nm}"] = average_precision((yt == c).astype(int), st)
        tp = int(((pred == c) & (yt == c)).sum())
        se0, ppv0 = tp / max(int((yt == c).sum()), 1), tp / max(int((pred == c).sum()), 1)
        r[f"default_{nm}"] = (100 * se0, 100 * ppv0, 100 * 2 * se0 * ppv0 / max(se0 + ppv0, 1e-12))
        for t in TARGET_SE:
            thr = threshold_for_se((yv == c).astype(int), sv, t)
            r[f"{nm}_se{int(t*100)}"] = apply(lt, c, thr, yt)
    return r


def render(allres, mean_rows: bool = True) -> str:
    f = lambda t: f"{t[0]:.1f}/{t[1]:.1f}/{t[2]:.1f}"
    L = ["| ckpt | class | AP | default Se/PPV/F1 | val-thr Se~76 -> test Se/PPV/F1 | ~85 | ~91 |", "|---|---|---|---|---|---|---|"]
    for r in allres:
        for nm in ("S", "V"):
            L.append(f"| {r['ckpt']} | {nm} | {r[f'AP_{nm}']:.3f} | {f(r[f'default_{nm}'])} | "
                     f"{f(r[f'{nm}_se76'])} | {f(r[f'{nm}_se85'])} | {f(r[f'{nm}_se91'])} |")
    singles = [r for r in allres if not r["ckpt"].startswith("ENSEMBLE")]
    if mean_rows and len(singles) > 1:
        for nm in ("S", "V"):
            m = lambda key: np.mean([r[key] for r in singles], 0)
            L.append(f"| **mean of single models** | {nm} | {np.mean([r[f'AP_{nm}'] for r in singles]):.3f} | {f(m(f'default_{nm}'))} | "
                     f"{f(m(f'{nm}_se76'))} | {f(m(f'{nm}_se85'))} | {f(m(f'{nm}_se91'))} |")
    L += ["", "Thresholds are chosen on the validation split (DS1 patients) for the stated target sensitivity and then "
          "applied unchanged to DS2, so test Se will differ from the target. Test sensitivity is dominated by a "
          "few records (e.g. record 232 holds ~75% of DS2 S beats)."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="./data/mitdb")
    ap.add_argument("--cache_dir", default="./data/cache")
    ap.add_argument("--out_dir", default="./results/opts")
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args(argv)
    allres, LV, LT, ref = [], [], [], None
    for ck in [c.strip() for c in a.checkpoint.split(",")]:
        model, _, data_cfg, trained = load_checkpoint(ck, "cpu", 1)
        assert trained, ck
        cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
        sets = build_datasets(cfg)
        lv, lt = logits_of(model, sets["val"]), logits_of(model, sets["test"])
        yv, yt = sets["val"].y.numpy(), sets["test"].y.numpy()
        if ref is None:
            ref = (yv, yt)
        elif not (np.array_equal(ref[0], yv) and np.array_equal(ref[1], yt)):
            raise SystemExit(f"{ck}: val/test labels differ from the first checkpoint; cannot ensemble")
        LV.append(lv)
        LT.append(lt)
        allres.append(evaluate(Path(ck).stem, lv, lt, yv, yt))
    if a.ensemble and len(LV) > 1:
        allres.append(evaluate(f"ENSEMBLE of {len(LV)}", ensemble_logits(LV), ensemble_logits(LT), ref[0], ref[1]))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operating_points.json").write_text(json.dumps(allres, indent=1))
    md = render(allres)
    (out / "operating_points.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
