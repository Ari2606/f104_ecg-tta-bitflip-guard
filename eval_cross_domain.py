"""
eval_cross_domain.py  --  this project, Stage 3 (Cross-Domain Shift Validation)

Zero-shot evaluation of a Stage-1 checkpoint on:
  (a) DS2 "clean" (Earth-domain, inter-patient, zero-shot in the de Chazal sense --
      the model has never seen these patients, but the physiology is still terrestrial), and
  (b) DS2 run through the Stage-2 `SpaceShiftSimulator` at a severity sweep, i.e. the
      *true* cross-domain shift this paper is about (terrestrial-trained model, deployed
      physiology + hardware distribution it never saw).

This script does NOT adapt anything (no TTA). It exists to quantify how badly a frozen
source model degrades as severity rises -- the "Source-Only" reference curve that every
method in `baselines_tta.py` / `safety_controller.py` (Stages 4-5) will be compared
against, and it is also how a `train_stage1.py` checkpoint gets its first honest report
card before any adaptation work is built.

Checkpoint contract (produced by `train_stage1.py`, Stage 1 -- not yet written)
--------------------------------------------------------------------------
torch.save({
    "model_state": model.state_dict(),        # BeatCNN, incl. anchor + snapshot
    "cnn_cfg":     dataclasses.asdict(cnn_cfg),# base_cnn.CNNConfig fields
    "data_cfg":    {...},                      # the subset of ECGDataConfig that fixes
                                                # tensor shape / preprocessing: n_leads,
                                                # pre, post, norm, bandpass, band,
                                                # use_rr_features, n_rr_features
    "epoch": int, "val_macro_f1": float,
}, path)
If no checkpoint is given, this script falls back to a freshly-initialized (untrained)
model so the rest of the pipeline (metrics, CSV schema, plots) can be smoke-tested before
Stage 1 exists -- results are meaningless in that mode and are labelled as such.

INCART note (Stage 3 spec mentions "INCART or DS2"): no INCART loader exists yet because
`dataset.py` only implements MIT-BIH. `--extra_dataset incart` is wired as an explicit
NotImplementedError rather than silently falling back to DS2, so a paper run never
mislabels DS2 as INCART. Add an INCART loader with the same (x[N,C,L], y[N], record_ids,
sample_idx) duck-type to `dataset.py` and register it in `EXTRA_DATASETS` below to enable it.

Shape legend: B = batch, C = leads, L = 256 samples, K = NUM_CLASSES = 5.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from base_cnn import CNNConfig, BeatCNN
from dataset import (CLASS_NAMES, DS2_RECORDS, NUM_CLASSES, ECGDataConfig,
                     MITBIHDataset, build_datasets)
from space_shift_simulator import SpaceShiftConfig, SpaceShiftDataset, SpaceShiftSimulator

# Fields of ECGDataConfig that fix tensor shape / preprocessing (must match the
# checkpoint that produced the weights; everything else is a path/runtime knob).
_DATA_CONTRACT_FIELDS = ("n_leads", "pre", "post", "norm", "bandpass", "band",
                         "use_rr_features", "n_rr_features", "n_template")

# Hook for future non-MIT-BIH cohorts sharing the MITBIHDataset duck-type
# (x [N,C,L] float32, y [N] int64, record_ids [N] int64, sample_idx [N] int64).
EXTRA_DATASETS: Dict[str, "callable"] = {}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    checkpoint: Optional[str] = None
    data_dir: str = "./data/mitdb"
    cache_dir: str = "./data/cache"
    out_dir: str = "./results/stage3_cross_domain"
    batch_size: int = 256
    device: str = "cpu"
    seed: int = 0
    # Severity sweep applied to DS2 via the Stage-2 simulator. 0.0 == Earth == clean DS2.
    severities: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    extra_dataset: Optional[str] = None      # e.g. "incart", once registered


# --------------------------------------------------------------------------- #
# Metrics (dependency-free: no sklearn requirement)
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    """y_true, y_pred: int64 [N] in [0,k) -> cm [k, k], rows=true, cols=pred."""
    idx = y_true.astype(np.int64) * k + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=k * k).reshape(k, k)


def classification_report(cm: np.ndarray, class_names: Sequence[str] = CLASS_NAMES) -> Dict:
    """cm [K,K] -> per-class precision/recall/f1/support + accuracy, macro-F1, weighted-F1."""
    k = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)          # true counts per class      [K]
    pred_tot = cm.sum(axis=0).astype(np.float64)          # predicted counts per class [K]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / np.clip(precision + recall, 1e-12, None), 0.0)
    accuracy = float(tp.sum() / max(cm.sum(), 1))
    present = support > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    weighted_f1 = float((f1 * support).sum() / max(support.sum(), 1))
    per_class = {class_names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                for i in range(k)}
    macro_precision = float(precision[present].mean()) if present.any() else 0.0
    macro_recall = float(recall[present].mean()) if present.any() else 0.0

    # Supplementary, NOT a replacement for the full 5-class macro_f1 above (always report
    # both). Several published baselines -- including Romdhane et al. PeerJ CS 2020, whose
    # own per-class F1 table (N=95.88, SVEB=74.29, VEB=92.40, F=1.77) averages to EXACTLY
    # their reported "macro-F1=66.09%" -- compute macro-F1 over N/S/V/F only, silently
    # excluding Q (DS1 has ~4 Q-beats total, so Q is often dropped from the table entirely
    # rather than stated as excluded). Comparing our FULL 5-class macro_f1 against their
    # 4-class number understates us by conflating two different metrics; macro_f1_ex_fq is
    # the actually-comparable one.
    def _ex(exclude: Sequence[str]) -> float:
        idx = [i for i, c in enumerate(class_names) if c not in exclude and support[i] > 0]
        return float(f1[idx].mean()) if idx else 0.0

    macro_f1_ex_f = _ex(["F"])
    macro_f1_ex_fq = _ex(["F", "Q"])

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "macro_precision": macro_precision, "macro_recall": macro_recall,
            "macro_f1_ex_f": macro_f1_ex_f, "macro_f1_ex_fq": macro_f1_ex_fq,
            "per_class": per_class, "confusion_matrix": cm.tolist()}


# --------------------------------------------------------------------------- #
# Checkpoint / model
# --------------------------------------------------------------------------- #
def load_checkpoint(path: Optional[str], device: str, n_leads_default: int = 1
                    ) -> Tuple[BeatCNN, CNNConfig, ECGDataConfig, bool]:
    """
    Returns (model in eval mode, cnn_cfg, data_cfg, is_trained).
    is_trained=False means no checkpoint was given -> random-init smoke-test model.
    """
    if path is None:
        print("[eval_cross_domain] WARNING: no --checkpoint given. Evaluating a randomly "
              "initialized model -- results are NOT meaningful, this is a pipeline smoke test.",
              file=sys.stderr)
        data_cfg = ECGDataConfig(n_leads=n_leads_default, use_rr_features=False)
        cnn_cfg = CNNConfig(in_channels=data_cfg.in_channels)
        model = BeatCNN(cnn_cfg).to(device)
        model.freeze_anchor()
        model.eval()
        return model, cnn_cfg, data_cfg, False

    ckpt = torch.load(path, map_location=device)
    for req in ("model_state", "cnn_cfg", "data_cfg"):
        if req not in ckpt:
            raise KeyError(f"checkpoint '{path}' is missing required key '{req}'. "
                           f"See the checkpoint contract in this file's module docstring.")
    cnn_cfg = CNNConfig(**ckpt["cnn_cfg"])
    data_cfg = ECGDataConfig(**{**asdict(ECGDataConfig()), **ckpt["data_cfg"]})
    model = BeatCNN(cnn_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[eval_cross_domain] WARNING: state_dict mismatch -- "
              f"missing={missing}, unexpected={unexpected}", file=sys.stderr)
    if not bool(model.anchor_frozen):
        model.freeze_anchor()
    model.eval()
    return model, cnn_cfg, data_cfg, True


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class InferenceResult:
    y_true: np.ndarray      # [N]
    y_pred: np.ndarray      # [N]
    entropy: np.ndarray     # [N]  predictive entropy (nats) of the adaptable head
    corrupt_frac: Optional[np.ndarray] = None   # [N]  fraction of window flagged corrupted


@torch.no_grad()
def run_inference(model: BeatCNN, loader: DataLoader, device: str,
                  has_meta: bool = False) -> InferenceResult:
    """
    Unpacks batches as (x, [aux], y, [mask]) -- aux (RR features) present iff
    model.cfg.aux_feat_dim > 0 (matches dataset.py's MITBIHDataset/SpaceShiftDataset
    tuple ordering), mask present iff `has_meta` (caller must pass True iff the
    loader's dataset was built with return_meta=True -- tuple length alone can't
    disambiguate "(x, rr, y)" from "(x, y, mask)", both length 3).
    """
    use_aux = model.cfg.aux_feat_dim > 0
    ys, ps, hs, cf = [], [], [], []
    for batch in loader:
        idx = 0
        x = batch[idx]; idx += 1                                   # [B,C,L]
        aux = None
        if use_aux:
            aux = batch[idx].to(device); idx += 1                  # [B, aux_feat_dim]
        y = batch[idx]; idx += 1                                   # [B]
        mask = batch[idx] if has_meta else None                    # [B,L] or None
        x = x.to(device)
        out = model(x, aux=aux, use_anchor=False)
        logits = out.logits                                       # [B, K]
        probs = F.softmax(logits, dim=-1)                          # [B, K]
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)  # [B]
        ys.append(y.numpy())
        ps.append(logits.argmax(-1).cpu().numpy())
        hs.append(ent.cpu().numpy())
        if mask is not None:
            cf.append(mask.float().mean(-1).numpy())               # [B]
    return InferenceResult(
        y_true=np.concatenate(ys), y_pred=np.concatenate(ps), entropy=np.concatenate(hs),
        corrupt_frac=np.concatenate(cf) if has_meta else None)


# --------------------------------------------------------------------------- #
# Evaluation passes
# --------------------------------------------------------------------------- #
def evaluate_clean(model: BeatCNN, ds_test: MITBIHDataset, cfg: EvalConfig) -> Dict:
    loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    res = run_inference(model, loader, cfg.device, has_meta=False)
    rep = classification_report(confusion_matrix(res.y_true, res.y_pred))
    rep.update(severity=0.0, scenario="earth_clean", mean_entropy=float(res.entropy.mean()),
              n_beats=int(len(res.y_true)))
    return rep


def evaluate_shift_sweep(model: BeatCNN, ds_test: MITBIHDataset, cfg: EvalConfig
                         ) -> List[Dict]:
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=cfg.seed))
    rows = []
    for s in cfg.severities:
        shifted = SpaceShiftDataset(ds_test, sim, severity=float(s), return_meta=True)
        loader = DataLoader(shifted, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        res = run_inference(model, loader, cfg.device, has_meta=True)
        rep = classification_report(confusion_matrix(res.y_true, res.y_pred))
        clean_ent = float(res.entropy[res.corrupt_frac == 0].mean()) if (res.corrupt_frac == 0).any() else float("nan")
        corrupt_ent = float(res.entropy[res.corrupt_frac > 0].mean()) if (res.corrupt_frac > 0).any() else float("nan")
        rep.update(severity=float(s), scenario="space_shift",
                  mean_entropy=float(res.entropy.mean()),
                  mean_entropy_clean_beats=clean_ent, mean_entropy_corrupted_beats=corrupt_ent,
                  mean_corrupt_frac=float(res.corrupt_frac.mean()), n_beats=int(len(res.y_true)))
        rows.append(rep)
    return rows


def load_extra_dataset(name: str, cfg: EvalConfig, data_cfg: ECGDataConfig) -> MITBIHDataset:
    if name not in EXTRA_DATASETS:
        raise NotImplementedError(
            f"'{name}' is not registered in EXTRA_DATASETS. dataset.py currently implements "
            f"MIT-BIH only; add an INCART loader that returns an object with the same "
            f"(x[N,C,L], y[N], record_ids[N], sample_idx[N]) duck-type as MITBIHDataset and "
            f"register it under EXTRA_DATASETS['{name}'] to enable this flag.")
    return EXTRA_DATASETS[name](cfg, data_cfg)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_row(r: Dict) -> str:
    return (f"{r['scenario']:>14s}  sev={r['severity']:.2f}  n={r['n_beats']:6d}  "
           f"acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
           f"macroF1_exFQ={r.get('macro_f1_ex_fq', float('nan')):.4f}  "
           f"macroPrecision={r.get('macro_precision', float('nan')):.4f}  "
           f"macroRecall={r.get('macro_recall', float('nan')):.4f}  "
           f"H={r.get('mean_entropy', float('nan')):.3f}")


def summarize(clean: Dict, sweep: List[Dict]) -> str:
    lines = ["", "=" * 78, "this project | Stage 3: Cross-Domain Shift Validation (DS2, zero-shot)",
             "=" * 78, _fmt_row(clean)]
    for r in sweep:
        lines.append(_fmt_row(r))
    lines += ["-" * 78,
             f"earth_clean is the TRUE held-out DS2 test set (never touched during training or "
             f"checkpoint selection) -- this row, not any DS1-val number, is what's directly "
             f"comparable to a paper's reported test accuracy/precision/sensitivity: "
             f"Romdhane et al. PeerJ CS 2020 (same DS1/DS2 protocol): acc=92.53%, "
             f"macroPrecision=64.68%, macroRecall(sensitivity)=68.55%; "
             f"MBE 2024 (inter-patient): acc=91.22%, macroPrecision=68.23%, "
             f"macroRecall(sensitivity)=89.91%.",
             f"IMPORTANT metric-matching note: Romdhane's own reported 'macro-F1=66.09%' is "
             f"NOT a full 5-class average -- their per-class F1 table (N=95.88, SVEB=74.29, "
             f"VEB=92.40, F=1.77, no Q row at all) averages to EXACTLY 66.09% over N/S/V/F "
             f"only. So macroF1_exFQ above (not macro_f1) is the number directly comparable "
             f"to their 66.09% -- comparing our full 5-class macro_f1 against their 4-class "
             f"number understates us."]
    base_acc = clean["accuracy"]
    worst = min(sweep, key=lambda r: r["accuracy"]) if sweep else None
    if worst is not None:
        lines += ["-" * 78,
                 f"Accuracy drop @ severity={worst['severity']:.2f}: "
                 f"{base_acc:.4f} -> {worst['accuracy']:.4f}  "
                 f"(Delta = {worst['accuracy'] - base_acc:+.4f}); "
                 f"this is the Source-Only curve later methods must beat.",
                 "=" * 78, ""]
    return "\n".join(lines)


def save_results(clean: Dict, sweep: List[Dict], out_dir: str, is_trained: bool) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = [clean] + sweep
    with open(out / "cross_domain_results.json", "w") as f:
        json.dump({"is_trained_checkpoint": is_trained, "results": all_rows}, f, indent=2)
    csv_cols = ["scenario", "severity", "n_beats", "accuracy", "macro_f1", "weighted_f1",
               "mean_entropy", "mean_entropy_clean_beats", "mean_entropy_corrupted_beats",
               "mean_corrupt_frac"]
    with open(out / "cross_domain_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[eval_cross_domain] wrote {out/'cross_domain_results.json'} and .csv")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None, help="Stage-1 checkpoint (.pt); omit for a smoke test")
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--out_dir", default="./results/stage3_cross_domain")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--extra_dataset", default=None, choices=[None, "incart"])
    a = p.parse_args(argv)

    cfg = EvalConfig(checkpoint=a.checkpoint, data_dir=a.data_dir, cache_dir=a.cache_dir,
                     out_dir=a.out_dir, batch_size=a.batch_size, device=a.device, seed=a.seed,
                     severities=tuple(float(s) for s in a.severities.split(",")),
                     extra_dataset=a.extra_dataset)

    torch.manual_seed(cfg.seed)
    model, cnn_cfg, data_cfg, is_trained = load_checkpoint(cfg.checkpoint, cfg.device)
    data_cfg = dataclasses.replace(data_cfg, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir)

    if cfg.extra_dataset is not None:
        ds_test = load_extra_dataset(cfg.extra_dataset, cfg, data_cfg)
    else:
        ds_test = build_datasets(data_cfg)["test"]         # DS2, patient-disjoint from DS1

    clean = evaluate_clean(model, ds_test, cfg)
    sweep = evaluate_shift_sweep(model, ds_test, cfg)
    print(summarize(clean, sweep))
    save_results(clean, sweep, cfg.out_dir, is_trained)
    return {"clean": clean, "sweep": sweep}


# --------------------------------------------------------------------------- #
# Self-test (no PhysioNet download required): synthetic duck-typed dataset that
# satisfies MITBIHDataset's public contract (x, y, record_ids, sample_idx).
# --------------------------------------------------------------------------- #
class _SyntheticECGDataset(Dataset):
    def __init__(self, n: int = 512, c: int = 1, l: int = 256, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, c, l, generator=g)
        self.y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        self.record_ids = torch.full((n,), 999)
        self.sample_idx = torch.arange(n) * 300 + 10_000

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("[eval_cross_domain] no CLI args given -> running self-test on a synthetic "
              "duck-typed dataset (no PhysioNet download needed).")
        cfg = EvalConfig(checkpoint=None, severities=(0.0, 0.5, 1.0))
        model, _, _, is_trained = load_checkpoint(None, cfg.device, n_leads_default=1)
        ds = _SyntheticECGDataset(n=512, c=1, l=256, seed=cfg.seed)
        clean = evaluate_clean(model, ds, cfg)
        sweep = evaluate_shift_sweep(model, ds, cfg)
        print(summarize(clean, sweep))
        assert clean["n_beats"] == 512 and len(sweep) == 3
        assert 0.0 <= clean["accuracy"] <= 1.0 and 0.0 <= clean["macro_f1"] <= 1.0
        assert sweep[0]["mean_corrupt_frac"] == 0.0, "severity=0 must be the identity (no corruption)"
        save_results(clean, sweep, "./results/_selftest_stage3", is_trained)
        print("Self-test passed.")
