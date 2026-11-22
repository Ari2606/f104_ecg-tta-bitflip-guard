"""
buffer_fault.py  --  single-event upsets in the STREAMING INPUT STATE (not the weights).

The patient-relative model keeps two small writable buffers on the device:
  * template buffer : the last k=16 raw beats (16 x 256 float32 = 16 KB); channel 1 = sample-wise median of it,
                      channel 2 = beat - median.
  * RR register     : the 7 RR features (a short history; modelled as a value that persists W=8 beats).
A bit flip in either one corrupts the model INPUT for the following beats. This script injects such flips as a
Poisson process (rate = flips per 1000 beats, applied to each buffer), re-derives the template channels from the
corrupted buffer exactly as dataset.add_template_channels does, and scores the frozen model on all of DS2.

  unguarded : the buffers are used as stored (a NaN/Inf in the buffer propagates through the median)
  guarded   : on every read, buffer values are nan_to_num'd and clamped to bounds measured on DS1 TRAIN
              (waveform: 1.5 x max |x|; RR: per-feature min/max). No gradient step, no learning.

Scope and honesty: stress test, not an orbit prediction. The 131,072-bit template buffer at the JASON-2 rate
(4.76e-7 upsets/bit/day) takes ~0.06 flips/day; the simulated rates are 10^3-10^5 times higher. RR persistence
(W=8) is a modelling assumption. Flips in weights, activations, code are out of scope here (see fault_injection.py).

  python buffer_fault.py --checkpoint ckpt.pt --data_dir ... --cache_dir ... --out_dir ... [--rates 1,10,100 --seeds 0,1]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
np.seterr(all="ignore")

K, W = 16, 8                                   # template buffer depth (beats); RR register persistence (beats)


def flip_f32(a: np.ndarray, idx, bit: int) -> None:
    """In-place flip of `bit` of float32 element a[idx] (a is a contiguous float32 array)."""
    v = a.view(np.uint32)
    v[idx] ^= np.uint32(1 << bit)


def sanitize(a: np.ndarray, lo, hi) -> np.ndarray:
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=hi if np.isscalar(hi) else 0.0,
                                 neginf=lo if np.isscalar(lo) else 0.0), lo, hi)


def run_record(x: np.ndarray, rr: np.ndarray, rate: float, rng: np.random.Generator,
               guard: bool, wave_b: float, rr_lo: np.ndarray, rr_hi: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """x [n, 3, L] chronological, channel 0 = raw beat (template channels are rebuilt).
    Returns (x_out, rr_out) with the corrupted template channels / RR register."""
    n, _, L = x.shape
    xo = x.copy()
    rro = rr.copy()
    ring = np.zeros((K, L), dtype=np.float32)
    cnt = 0
    for i in range(n):
        for _ in range(rng.poisson(rate / 1000.0)):                     # SEUs in the template buffer
            if cnt > 0:
                flip_f32(ring, (int(rng.integers(0, min(cnt, K))), int(rng.integers(0, L))), int(rng.integers(0, 32)))
        if i == 0:
            tmpl = x[0, 0, :].copy()
        else:
            buf = ring[:min(cnt, K)]
            if guard:
                buf = sanitize(buf, -wave_b, wave_b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmpl = np.median(buf, axis=0)
        xo[i, 1, :] = tmpl
        xo[i, 2, :] = x[i, 0, :] - tmpl
        ring[cnt % K] = x[i, 0, :]
        cnt += 1
        for _ in range(rng.poisson(rate / 1000.0)):                     # SEUs in the RR register
            j, b = int(rng.integers(0, rr.shape[1])), int(rng.integers(0, 32))
            for t in range(i, min(i + W, n)):
                flip_f32(rro[t], j, b)
    if guard:
        rro = sanitize(rro, rr_lo[None, :], rr_hi[None, :]).astype(np.float32)
    return xo, rro


def logits_xr(model, x: np.ndarray, rr: np.ndarray, bs: int = 1024) -> np.ndarray:
    import torch
    model.eval()
    use_aux = model.cfg.aux_feat_dim > 0
    out = []
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs]))
            ab = torch.from_numpy(np.ascontiguousarray(rr[i:i + bs])) if use_aux else None
            out.append(model(xb, aux=ab).logits.cpu().numpy())
    return np.concatenate(out)


def corrupt_stream(x, rr, rid, rate, seed, guard, wave_b, rr_lo, rr_hi):
    xs, rs = np.empty_like(x), np.empty_like(rr)
    rng = np.random.default_rng(seed)
    for r in np.unique(rid):                                            # buffers reset per record (per patient)
        m = np.flatnonzero(rid == r)
        xs[m], rs[m] = run_record(x[m], rr[m], rate, rng, guard, wave_b, rr_lo, rr_hi)
    return xs, rs


def row_metrics(lg: np.ndarray, y: np.ndarray, rid: np.ndarray) -> Dict:
    from protocol_eval import evaluate
    a = evaluate(lg, y, rid)["A_5class"]
    return {"acc": a["acc"], "S_F1": a[1]["F1"], "V_F1": a[2]["F1"], "macro_F1": a["macro_F1"] if "macro_F1" in a else
            float(np.mean([a[c]["F1"] for c in (0, 1, 2)])), "nan_rows": int((~np.isfinite(lg).all(1)).sum())}


def render(rows: List[Dict], base: Dict) -> str:
    L = ["| flips / 1000 beats (each buffer) | guard | acc | S F1 | V F1 | mean(N,S,V) F1 | beats with non-finite logits |",
         "|---|---|---|---|---|---|---|",
         f"| 0 (clean) | - | {base['acc']:.1f} | {base['S_F1']:.1f} | {base['V_F1']:.1f} | {base['macro_F1']:.1f} | {base['nan_rows']} |"]
    for r in rows:
        L.append(f"| {r['rate']:g} | {'clamp' if r['guard'] else 'none'} | {r['acc']:.1f} | {r['S_F1']:.1f} | {r['V_F1']:.1f} | "
                 f"{r['macro_F1']:.1f} | {r['nan_rows']:.0f} |")
    return "\n".join(L) + ("\n\nFrozen model, flips in the 16-beat template buffer and the RR register. Mean over seeds. "
                           "Stress test at rates 10^3-10^5 x above the JASON-2 upset rate; not an orbit prediction.\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rates", default="1,10,100")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    from dataset import ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    model, _, dcfg, ok = load_checkpoint(a.checkpoint, "cpu", 1)
    assert ok
    cfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
    d = build_datasets(cfg)
    tr, te = d["train"], d["test"]
    assert te.x.shape[1] == 3, "expects waveform + 2 template channels (n_template>0, 1 lead)"
    wave_b = float(1.5 * np.abs(tr.x[:, 0, :].numpy()).max())
    rr_lo, rr_hi = tr.rr.numpy().min(0).astype(np.float32), tr.rr.numpy().max(0).astype(np.float32)
    x, rr = te.x.numpy().astype(np.float32), te.rr.numpy().astype(np.float32)
    y, rid = te.y.numpy(), te.record_ids.numpy()
    print(f"[buffer_fault] DS2 beats={len(y)}  wave bound={wave_b:.2f}  RR bounds lo={rr_lo.round(2)} hi={rr_hi.round(2)}", flush=True)
    base = row_metrics(logits_xr(model, x, rr), y, rid)
    # sanity: rate 0 must reproduce the stored template channels
    x0, r0 = corrupt_stream(x[:3000], rr[:3000], rid[:3000], 0.0, 0, False, wave_b, rr_lo, rr_hi)
    assert np.allclose(x0, x[:3000], atol=1e-5), "template rebuild does not match dataset channels"
    rows = []
    for rate in [float(v) for v in a.rates.split(",")]:
        for guard in (False, True):
            ms = []
            for sd in [int(v) for v in a.seeds.split(",")]:
                xs, rs = corrupt_stream(x, rr, rid, rate, sd, guard, wave_b, rr_lo, rr_hi)
                ms.append(row_metrics(logits_xr(model, xs, rs), y, rid))
            r = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            r.update(rate=rate, guard=guard)
            rows.append(r)
            print(f"[buffer_fault] rate={rate:g} guard={guard}  S_F1={r['S_F1']:.1f} V_F1={r['V_F1']:.1f} "
                  f"nan_rows={r['nan_rows']:.0f}", flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    md = render(rows, base)
    (out / "buffer_fault.md").write_text(md); (out / "buffer_fault.json").write_text(json.dumps({"base": base, "rows": rows}, indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    from dataset import add_template_channels
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(600, 1, 64)).astype(np.float32)
    x3 = add_template_channels(raw, K)
    rr = rng.normal(size=(600, 7)).astype(np.float32)
    lo, hi = np.full(7, -3, np.float32), np.full(7, 3, np.float32)
    xo, ro = run_record(x3, rr, 0.0, np.random.default_rng(1), False, 10.0, lo, hi)
    assert np.allclose(xo, x3, atol=1e-6) and np.array_equal(ro, rr)                     # rate 0 = identity
    xu, ru = run_record(x3, rr, 2000.0, np.random.default_rng(2), False, 10.0, lo, hi)    # heavy flips, unguarded
    xg, rg = run_record(x3, rr, 2000.0, np.random.default_rng(2), True, 10.0, lo, hi)
    assert not np.isfinite(xu).all() or not np.isfinite(ru).all() or np.abs(ru).max() > 10     # damage happens
    assert np.isfinite(xg).all() and np.isfinite(rg).all() and np.abs(rg).max() <= 3.0 + 1e-6  # guard bounds it
    f = np.array([1.0], np.float32); flip_f32(f, 0, 31); assert f[0] == -1.0
    print("Self-test passed.")
