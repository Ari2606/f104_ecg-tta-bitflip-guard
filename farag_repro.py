"""
farag_repro.py  --  Farag 2023 (Sensors 23:1365, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification")
re-implemented from the paper and trained/evaluated under OUR protocol, to test whether the recipe reaches its published
S precision when nothing is selected on DS2.

What is reproduced (from the paper's text):
  * MLII lead, raw (no filtering/normalisation), resampled 360 -> 128 Hz, 64-sample (0.5 s) window centred on the R peak,
    FIRST DERIVATIVE of the window as the input (their best variants);
  * RR features: pre/post RR interval each divided by the LOCAL mean (last 80 RR) and the GLOBAL mean (last 400 RR) -> 4 numbers;
  * network: Conv1D (11 filters, kernel NK=32|64, 'same') -> BatchNorm -> tanh -> GlobalMaxPool; RR branch Dense 32-16-8 (ReLU);
    concatenate -> softmax over N/S/V;  Adam 1e-3, batch 512, early stopping, class weights SET or NOT SET (both reported);
  * DS1 split for validation: records < 200 train, records >= 200 validate (as in the paper); DS2 is the test set.
What is NOT reproduced: matched-filter template initialisation (they report it makes little difference; Glorot used here).
Differences from the paper that make this STRICTER: the epoch is chosen on DS1 validation only, one pre-declared configuration per
run (no best-of-100 variants), 5 seeds, F/Q scored two ways.

Scoring (DS2, both with and without record 232):
  'F/Q removed'  : F and Q beats dropped from scoring (the paper's 3-class setting, as far as the text says);
  'F/Q kept'     : F and Q beats stay in the test set (they cannot be predicted, so they only add errors / false positives).

  python farag_repro.py --data_dir .../mitdb --cache_dir .../cache --out_dir .../results/farag --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

FS_IN, FS_OUT, HALF = 360, 128, 32          # 64-sample window at 128 Hz


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def rr_features(samples: np.ndarray, local: int = 80, glob: int = 400) -> np.ndarray:
    """[N,4] = (pre/local, post/local, pre/global, post/global). Local/global means use only intervals that END at or
    before the current beat (past only); the post interval is the next beat's RR (as in the paper). Missing history -> 1."""
    s = samples.astype(np.float64)
    n = len(s)
    d = np.diff(s)                                     # d[j] = s[j+1]-s[j]
    cs = np.concatenate([[0.0], np.cumsum(d)])
    out = np.ones((n, 4), dtype=np.float32)
    for i in range(n):
        pre = d[i - 1] if i >= 1 else (d[i] if n > 1 else 1.0)
        post = d[i] if i < n - 1 else pre
        for k, W in enumerate((local, glob)):
            lo = max(0, i - W)
            if i - lo >= 1:
                m = (cs[i] - cs[lo]) / (i - lo)
                out[i, 2 * k], out[i, 2 * k + 1] = pre / m, post / m
    return np.clip(out, 0.0, 5.0)


def load_record(record: int, data_dir: str, cache_dir: str):
    from dataset import AAMI_MAP
    cache = Path(cache_dir) / f"farag_{record}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"], d["rr"]
    import wfdb
    from scipy.signal import resample_poly
    rec = wfdb.rdrecord(str(Path(data_dir) / str(record)))
    ann = wfdb.rdann(str(Path(data_dir) / str(record)), "atr")
    names = [n.strip() for n in rec.sig_name]
    sig = rec.p_signal[:, names.index("MLII")].astype(np.float64)
    sig128 = resample_poly(sig, 16, 45)                # 360 -> 128 Hz (16/45)
    xs, ys, ss = [], [], []
    for s, sym in zip(ann.sample, ann.symbol):
        c = AAMI_MAP.get(sym)
        if c is None:
            continue
        i = int(round(s * FS_OUT / FS_IN))
        a, b = i - HALF, i + HALF
        if a < 0 or b > len(sig128):
            continue
        xs.append(np.gradient(sig128[a:b]).astype(np.float32))     # first derivative of the raw window
        ys.append(c); ss.append(int(s))
    x, y, ss = np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(ss)
    rr = rr_features(ss)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, rr=rr)
    return x, y, rr


def load_split(records: Sequence[int], data_dir: str, cache_dir: str):
    xs, ys, rrs, rid = [], [], [], []
    for r in records:
        x, y, rr = load_record(r, data_dir, cache_dir)
        xs.append(x); ys.append(y); rrs.append(rr); rid.append(np.full(len(y), r))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rrs), np.concatenate(rid)


# --------------------------------------------------------------------------- #
# Model + training
# --------------------------------------------------------------------------- #
class MFNet(nn.Module):
    def __init__(self, n_filters: int = 11, nk: int = 32, n_rr: int = 4, n_cls: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, nk, padding="same")
        self.bn = nn.BatchNorm1d(n_filters)
        self.rr = nn.Sequential(nn.Linear(n_rr, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.out = nn.Linear(n_filters + 8, n_cls)

    def forward(self, x, rr):                            # x [B,64], rr [B,4]
        h = torch.tanh(self.bn(self.conv(x.unsqueeze(1)))).amax(-1)
        return self.out(torch.cat([h, self.rr(rr)], 1))


def macro_f1_3(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum()
        pr, rc = tp / max((p == c).sum(), 1), tp / max((y == c).sum(), 1)
        f.append(2 * pr * rc / max(pr + rc, 1e-12))
    return float(np.mean(f))


def train_one(tr, va, seed: int, weights: str, nk: int, max_epochs: int = 200, patience: int = 15, device: str = "cpu",
              min_epochs: int = 30):
    torch.manual_seed(seed); np.random.seed(seed)
    xt, yt, rt, _ = tr
    xv, yv, rv, _ = va
    k = yt <= 2
    xt, yt, rt = xt[k], yt[k], rt[k]
    kv = yv <= 2
    xv, yv, rv = xv[kv], yv[kv], rv[kv]
    model = MFNet(nk=nk).to(device)
    cnt = np.bincount(yt, minlength=3).astype(np.float64)
    w = torch.tensor(len(yt) / (3 * cnt), dtype=torch.float32, device=device) if weights == "balanced" else None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    Xt, Yt, Rt = (torch.tensor(a, device=device) for a in (xt, yt, rt))
    Xv, Rv = torch.tensor(xv, device=device), torch.tensor(rv, device=device)
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Yt), device=device)
        for i in range(0, len(perm), 512):
            ix = perm[i:i + 512]
            opt.zero_grad(); crit(model(Xt[ix], Rt[ix]), Yt[ix]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv, Rv).argmax(1).cpu().numpy()
        f1 = macro_f1_3(yv, pv)
        sched.step(1 - f1)
        if f1 > best:
            best, bad, best_ep = f1, 0, ep
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                break
    model.load_state_dict(best_state)
    return model.eval(), best_ep, best


@torch.no_grad()
def predict(model, x, rr, device: str = "cpu") -> np.ndarray:
    return model(torch.tensor(x, device=device), torch.tensor(rr, device=device)).argmax(1).cpu().numpy()


def score(y: np.ndarray, p: np.ndarray, rid: np.ndarray) -> Dict:
    from protocol_eval import class_table
    out = {}
    for tag, mask in (("FQ_removed", y <= 2), ("FQ_kept", np.ones_like(y, bool))):
        for r232, m2 in (("with232", mask), ("no232", mask & (rid != 232))):
            out[f"{tag}_{r232}"] = class_table(y[m2], p[m2], (0, 1, 2))
    return out


def render(res: Dict[str, List[Dict]]) -> str:
    L = ["| config | scoring | acc % | N Se/PPV/F1 | S Se/PPV/F1 | V Se/PPV/F1 |", "|---|---|---|---|---|---|"]
    for cfg, runs in res.items():
        for key in ("FQ_removed_with232", "FQ_removed_no232", "FQ_kept_with232", "FQ_kept_no232"):
            rs = [r["score"][key] for r in runs]
            m = lambda c, q: float(np.mean([r[c][q] for r in rs]))
            row = lambda c: f"{m(c,'Se'):.1f}/{m(c,'PPV'):.1f}/{m(c,'F1'):.1f}"
            L.append(f"| {cfg} | {key} | {np.mean([r['acc'] for r in rs]):.1f} | {row(0)} | {row(1)} | {row(2)} |")
    L += ["", "Mean over seeds. Published (Farag, DS2, selected on DS2): model 1 S 85.3/58.1, V 96.3/90.2; model 6 S 81.6/82.7, V 95.0/95.6 "
          "(3-class). 'FQ_removed' is the paper's setting as far as its text says; 'FQ_kept' is the stricter AAMI scoring."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--weights", default="balanced,none")
    ap.add_argument("--nk", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    from dataset import DS1_RECORDS, DS2_RECORDS
    tr = load_split([r for r in DS1_RECORDS if r < 200], a.data_dir, a.cache_dir)
    va = load_split([r for r in DS1_RECORDS if r >= 200], a.data_dir, a.cache_dir)
    te = load_split(DS2_RECORDS, a.data_dir, a.cache_dir)
    print(f"[farag] train {len(tr[1])} val {len(va[1])} test {len(te[1])} beats; train class counts {np.bincount(tr[1], minlength=5).tolist()}", flush=True)
    res: Dict[str, List[Dict]] = {}
    for wt in a.weights.split(","):
        runs = []
        for s in [int(v) for v in a.seeds.split(",")]:
            model, ep, vf1 = train_one(tr, va, s, wt, a.nk, device=a.device)
            p = predict(model.to(a.device), te[0], te[2], a.device)
            sc = score(te[1], p, te[3])
            runs.append({"seed": s, "best_epoch": ep, "val_f1": vf1, "score": sc, "acc": sc["FQ_removed_with232"]["acc"]})
            print(f"[farag] weights={wt} seed={s} epoch={ep} valF1={vf1:.3f} S(FQ removed)="
                  f"{sc['FQ_removed_with232'][1]['Se']:.1f}/{sc['FQ_removed_with232'][1]['PPV']:.1f}", flush=True)
        res[f"weights={wt},NK={a.nk}"] = runs
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "farag_repro.json").write_text(json.dumps(res, indent=1, default=float))
    md = render(res)
    (out / "farag_repro.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    # self-test: RR features are causal and flag a premature beat; model/training/scoring run on synthetic data
    rng = np.random.default_rng(0)
    s = np.cumsum((360 * rng.normal(0.8, 0.02, 700)).astype(int)); s[500] -= 150
    f = rr_features(s)
    assert f.shape == (700, 4) and f[500, 0] < 0.8 and abs(np.median(f[100:400, 0]) - 1.0) < 0.1
    s2 = s.copy(); s2[650:] += 300
    assert np.allclose(rr_features(s2)[:649, [0, 2]], f[:649][:, [0, 2]])      # pre-ratios depend on the past only
    n = 12000
    y = rng.choice(5, n, p=[.8, .08, .1, .015, .005])
    x = rng.normal(size=(n, 64)).astype(np.float32)
    x[y == 2, 30:36] += 2.0                                                    # V: distinct waveform
    rr = np.ones((n, 4), np.float32); rr[y == 1, 0] = 0.6                      # S: premature
    rid = rng.choice([100, 232, 233], n)
    tr, va = (x[:8000], y[:8000], rr[:8000], rid[:8000]), (x[8000:10000], y[8000:10000], rr[8000:10000], rid[8000:10000])
    model, ep, vf1 = train_one(tr, va, 0, "balanced", 32, max_epochs=60, patience=10)
    p = predict(model, x[10000:], rr[10000:])
    sc = score(y[10000:], p, rid[10000:])
    assert vf1 > 0.7 and sc["FQ_removed_with232"][1]["Se"] > 50 and sc["FQ_kept_no232"]["acc"] > 0
    assert "FQ_kept_no232" in render({"c": [{"score": sc, "acc": sc["FQ_removed_with232"]["acc"]}]})
    print(f"val macro-F1 {vf1:.3f}, synthetic S Se {sc['FQ_removed_with232'][1]['Se']:.0f}%")
    print("Self-test passed.")
