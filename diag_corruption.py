"""
diag_corruption.py  --  WHICH corruption breaks the source model?  (no adaptation involved)

The source model loses most of its accuracy on our corrupted stream (clean DS2 ~90% ->
~35% under the simulator at severity cap 0.4). Before changing anything, find out what
causes it: each simulator component alone (axis tilt, morphology, baseline wander, radiation),
all together, and REAL NSTDB noise (bw/em/ma) at several SNRs, on the clean DS2 beats.
Fixing the dominant cause (e.g. training-time augmentation for it) is worth more than any
adaptation tweak.

  !python diag_corruption.py --checkpoint ... --data_dir ... --cache_dir ... --nstdb_dir ... \
      --stride 3 --out_dir ./results/diag_corruption
Use --stride k to evaluate every k-th beat (all records stay represented) for a faster pass.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from dataset import ECGDataConfig, build_datasets
from eval_cross_domain import classification_report, confusion_matrix, load_checkpoint
from real_shift import NOISE_KINDS, NoiseBank, RealNoiseConfig, RealNoiseSimulator
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator

COMPONENTS = {
    "axis_tilt":  dict(enable_microgravity=True, enable_axis_tilt=True, enable_morphology=False, enable_wander=False, enable_radiation=False),
    "morphology": dict(enable_microgravity=True, enable_axis_tilt=False, enable_morphology=True, enable_wander=False, enable_radiation=False),
    "wander":     dict(enable_microgravity=False, enable_wander=True, enable_radiation=False),
    "radiation":  dict(enable_microgravity=False, enable_wander=False, enable_radiation=True),
    "all":        dict(),
}


@torch.no_grad()
def predict_corrupted(model, ds, idx: np.ndarray, corrupt: Optional[Callable], batch: int = 512) -> np.ndarray:
    use_aux = model.cfg.aux_feat_dim > 0
    preds = []
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        xs = []
        for i in ii:
            x = ds.x[i]
            if corrupt is not None:
                x = corrupt(x, int(ds.record_ids[i]), int(i), float(ds.sample_idx[i]))
            xs.append(x)
        xb = torch.stack(xs)
        aux = ds.rr[ii] if use_aux else None
        preds.append(model(xb, aux=aux).logits.argmax(-1).numpy())
    return np.concatenate(preds)


def evaluate(model, ds, idx, corrupt) -> Dict[str, float]:
    p = predict_corrupted(model, ds, idx, corrupt)
    y = ds.y.numpy()[idx]
    rep = classification_report(confusion_matrix(y, p))
    return {"accuracy": rep["accuracy"], "macro_f1_ex_fq": rep["macro_f1_ex_fq"],
            "recall_N": rep["per_class"]["N"]["recall"], "recall_S": rep["per_class"]["S"]["recall"],
            "recall_V": rep["per_class"]["V"]["recall"], "n": int(len(idx))}


def run(model, ds, stride: int, severities: Sequence[float], snrs: Sequence[float],
        bank: Optional[NoiseBank], seed: int = 0, fs: float = 360.0, r_index: int = 96) -> List[dict]:
    idx = np.arange(0, len(ds), stride)
    rows = [{"scenario": "clean", "param": "-", **evaluate(model, ds, idx, None)}]
    for comp, kw in COMPONENTS.items():
        sim = SpaceShiftSimulator(SpaceShiftConfig(seed=seed, **kw))
        for sev in severities:
            fn = (lambda x, rid, i, smp, sev=sev, sim=sim:
                  sim(x, record_id=rid, beat_key=i, t0_sec=(smp - r_index) / fs, severity=sev, return_mask=True)[0])
            rows.append({"scenario": f"sim:{comp}", "param": f"sev={sev:g}", **evaluate(model, ds, idx, fn)})
    if bank is not None:
        for kind in NOISE_KINDS:
            for snr in snrs:
                nsim = RealNoiseSimulator(bank, RealNoiseConfig(snr_db=snr, kinds=(kind,), clean_start_s=0.0,
                                                                always_on=True, seed=seed))
                fn = (lambda x, rid, i, smp, nsim=nsim:
                      nsim(x, record_id=rid, beat_key=i, t0_sec=(smp - r_index) / fs, severity=1.0, return_mask=True)[0])
                rows.append({"scenario": f"real:{kind}", "param": f"snr={snr:g}dB", **evaluate(model, ds, idx, fn)})
    return rows


def render(rows: List[dict]) -> str:
    L = ["| scenario | param | acc | F1 (N/S/V) | recall N | recall S | recall V |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scenario']} | {r['param']} | {r['accuracy']:.3f} | {r['macro_f1_ex_fq']:.3f} | "
                 f"{r['recall_N']:.3f} | {r['recall_S']:.3f} | {r['recall_V']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> List[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--nstdb_dir", default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--severities", default="0.2,0.4")
    p.add_argument("--snrs", default="12,6,0")
    p.add_argument("--out_dir", default="./results/diag_corruption")
    a = p.parse_args(argv)
    model, _, dcfg, trained = load_checkpoint(a.checkpoint, "cpu", 1)
    assert trained
    dcfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
    ds = build_datasets(dcfg)["test"]
    bank = NoiseBank.from_physionet(a.nstdb_dir).half("second") if a.nstdb_dir else None
    rows = run(model, ds, a.stride, [float(s) for s in a.severities.split(",")],
               [float(s) for s in a.snrs.split(",")], bank)
    md = render(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "diag_corruption.md").write_text(md); (out / "diag_corruption.json").write_text(json.dumps(rows, indent=1))
    print(md)
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(); sys.exit(0)
    import torch.nn.functional as F
    from base_cnn import CNNConfig, BeatCNN
    from baselines_tta import _SyntheticStreamDataset
    torch.manual_seed(0)
    model = BeatCNN(CNNConfig(in_channels=1))
    ds = _SyntheticStreamDataset(n_records=3, beats_per_record=120, c=1, l=256, seed=0)
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()), lr=1e-3)
    model.train()
    for _ in range(10):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            ix = perm[i:i + 32]
            loss = F.cross_entropy(model(ds.x[ix]).logits, ds.y[ix]); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    rows = run(model, ds, stride=2, severities=[0.4], snrs=[0.0], bank=NoiseBank.synthetic(60))
    print(render(rows))
    by = {(r["scenario"], r["param"]): r for r in rows}
    assert by[("clean", "-")]["accuracy"] > 0.8, "warm-up model should be accurate on clean synthetic data"
    assert len(rows) == 1 + 5 + 3, len(rows)
    assert min(r["accuracy"] for r in rows if r["scenario"] != "clean") <= by[("clean", "-")]["accuracy"]
    print("Self-test passed.")
