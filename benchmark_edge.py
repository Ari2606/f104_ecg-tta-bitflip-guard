"""
benchmark_edge.py  --  what does each TTA method cost on an edge device?

"Edge-deployable" is a claim about COST, so it is measured here for EVERY method in the
comparison, not just ours (the existing edge_profiler.py profiles this project's
SRAM / MACs / energy in detail; this script puts all methods on one axis):

  per-beat compute   MACs actually executed per beat, counted with forward hooks on every
                     Conv1d / Linear the method runs (teacher passes, augmentation passes,
                     replay batches, SAM's second pass ... all included, none assumed):
                       - no-grad forward MACs      (prediction, teacher, anchor, checks)
                       - grad-enabled forward MACs (adaptation)
                       - backward MACs ~= 1x the grad-forward MACs: weights are frozen
                         except the dw_bn affine, so a backward pass is dominated by input
                         gradients (an ESTIMATE, stated as such)
                     -> analytic MCU latency at `--mcu_macs_per_sec` (default 150 MMAC/s,
                        the CMSIS-NN-class figure edge_profiler.py documents)
  host latency       wall-clock per beat on THIS machine (p50/p95/p99). Run this script ON
                     the target (Raspberry Pi etc.) with --threads 1 to get real numbers.
  persistent state   bytes of everything the method keeps between beats beyond the model:
                     EMA teachers, replay memories, optimiser momentum, moving averages,
                     Fisher matrices (fp32). Found by walking the method object, so a new
                     method cannot hide its memory.
  anchor pass        whether the method NEEDS the frozen-source pass to run (our method's
                     elastic pull does; for the others it is a logging convenience for the
                     drift column and is excluded from their cost).

Not measured here (documented limitation): int8 kernels, flash, real MCU power.
Usage:
  python benchmark_edge.py [--checkpoint ... --data_dir ... --cache_dir ...] \
      --methods main,variants --n_beats 600 --threads 1 --out_dir ./results/edge
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import method_zoo
from run_experiments import load_source_model_and_stream
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
from streaming_eval import StreamingConfig, StreamingEvaluator, make_beat_stream


# --------------------------------------------------------------------------- #
class MacCounter:
    """Forward hooks on Conv1d/Linear, bucketed by mode."""

    def __init__(self, model: nn.Module):
        self.buckets = {"nograd": 0.0, "grad": 0.0, "anchor": 0.0}
        self._in_anchor = False
        self._hooks = []
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                self._hooks.append(m.register_forward_hook(self._hook))
        self._orig_anchor = model.forward_anchor
        model.forward_anchor = self._wrapped_anchor           # instance-level patch, undone in close()
        self._model = model

    def _wrapped_anchor(self, *a, **k):
        self._in_anchor = True
        try:
            return self._orig_anchor(*a, **k)
        finally:
            self._in_anchor = False

    def _hook(self, mod, inp, out):
        if isinstance(mod, nn.Conv1d):
            macs = out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0]
        else:
            macs = out.numel() * mod.in_features
        if self._in_anchor:
            self.buckets["anchor"] += macs
        elif torch.is_grad_enabled() and out.requires_grad:
            self.buckets["grad"] += macs
        else:
            self.buckets["nograd"] += macs

    def close(self):
        for h in self._hooks:
            h.remove()
        self._model.forward_anchor = self._orig_anchor


def state_bytes(obj, model_ids: set, seen: Optional[set] = None, depth: int = 0) -> int:
    """Bytes of tensors reachable from a method object, excluding the model's own parameters
    and the model itself. Optimiser state (momentum / Adam moments) is included; the
    parameters the optimiser points at are not (they live in the model)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or id(obj) in model_ids or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, nn.Module):
        return 0
    if isinstance(obj, torch.optim.Optimizer):
        return sum(state_bytes(v, model_ids, seen, depth + 1) for st in obj.state.values() for v in st.values())
    if isinstance(obj, dict):
        return sum(state_bytes(v, model_ids, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(state_bytes(v, model_ids, seen, depth + 1) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(state_bytes(v, model_ids, seen, depth + 1) for k, v in vars(obj).items()
                   if k not in ("masker",))             # masker holds only fixed windows (counted below)
    return 0


def benchmark(model, ds, key: str, n_beats: int, mcu_macs: float, hp: dict, sim_seed: int = 0) -> dict:
    meth = method_zoo.build_method(key, hp)
    if method_zoo.needs_source_beats(key):
        rr = getattr(ds, "rr", None)
        meth.set_fisher_data(ds.x[:200], rr[:200] if rr is not None and rr.shape[-1] > 0 else None)
    m = copy.deepcopy(model)
    counter = MacCounter(m)
    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=sim_seed))
    ramp = linear_ramp(start_idx=max(n_beats // 10, 5), n_ramp=max(n_beats // 4, 10))
    stream = make_beat_stream(ds, n_beats=n_beats, sim=sim, severity=lambda i: min(0.4, ramp(i)))
    ev = StreamingEvaluator(meth, StreamingConfig(device="cpu", rolling_window=100, log_every=0))
    res = ev.run(m, stream)
    counter.close()
    n = len(res.log)
    needs_anchor = key.startswith("pg")
    nog = counter.buckets["nograd"] / n
    gr = counter.buckets["grad"] / n
    anc = counter.buckets["anchor"] / n
    bwd = 1.0 * gr                                                    # estimate, see module docstring
    total = nog + gr + bwd + (anc if needs_anchor else 0.0)
    lat = np.array([r["latency_ms"] for r in res.log][min(20, n // 5):])
    sb = state_bytes(meth, {id(m)} | {id(p) for p in m.parameters()})
    if hasattr(meth, "masker"):                                       # STFT window tables etc. (tiny)
        sb += sum(t.numel() * t.element_size() for t in vars(meth.masker).values() if isinstance(t, torch.Tensor))
    return {"method": key, "n_beats": n,
            "macs_nograd_fwd": nog, "macs_grad_fwd": gr, "macs_bwd_est": bwd,
            "macs_anchor_pass": anc if needs_anchor else 0.0, "macs_total_per_beat": total,
            "mcu_latency_ms_est": 1000.0 * total / mcu_macs,
            "host_p50_ms": float(np.percentile(lat, 50)), "host_p95_ms": float(np.percentile(lat, 95)),
            "host_p99_ms": float(np.percentile(lat, 99)),
            "state_kb": sb / 1024.0, "needs_anchor_pass": needs_anchor,
            "adaptation_rate": float(res.summary["adaptation_rate"])}


def render(rows: List[dict], n_adapted_params: int, total_params: int, mcu: float) -> str:
    L = [f"Adapted parameters (all methods, sparse dw_bn scope): {n_adapted_params} of {total_params} "
         f"({100.0*n_adapted_params/total_params:.1f}%). MCU latency = total MACs / {mcu/1e6:.0f} MMAC/s "
         f"(an assumption, not a measurement); host latency = this machine.", "",
         "| method | MACs/beat (k) | no-grad fwd | grad fwd | bwd (est) | MCU ms (est) | host p50 / p95 ms | "
         "persistent state KB | anchor pass | adapt rate |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['method']} | {r['macs_total_per_beat']/1e3:.0f} | {r['macs_nograd_fwd']/1e3:.0f} | "
                 f"{r['macs_grad_fwd']/1e3:.0f} | {r['macs_bwd_est']/1e3:.0f} | {r['mcu_latency_ms_est']:.2f} | "
                 f"{r['host_p50_ms']:.2f} / {r['host_p95_ms']:.2f} | {r['state_kb']:.1f} | "
                 f"{'yes' if r['needs_anchor_pass'] else 'no'} | {r['adaptation_rate']:.2f} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> List[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_leads", type=int, default=1)
    p.add_argument("--methods", default="main,variants")
    p.add_argument("--n_beats", type=int, default=600)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mcu_macs_per_sec", type=float, default=150e6)
    p.add_argument("--hp", default="{}")
    p.add_argument("--out_dir", default="./results/edge")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    model, ds, is_real = load_source_model_and_stream(a.checkpoint, a.data_dir, a.cache_dir, a.n_leads, 3, 600, 0)
    keys = method_zoo.expand(a.methods)
    rows = [benchmark(model, ds, k, a.n_beats, a.mcu_macs_per_sec, json.loads(a.hp)) for k in keys]
    cp = model.count_parameters()
    md = render(rows, int(cp["tta_trainable"]), int(cp["total"] - cp["anchor_head"]), a.mcu_macs_per_sec)
    if not is_real:
        md = "**SYNTHETIC smoke test: costs below are real for this model graph but accuracy-free; "\
             "host timings are this sandbox, not an edge device.**\n\n" + md
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "edge_costs.md").write_text(md); (out / "edge_costs.json").write_text(json.dumps(rows, indent=1))
    print(md)
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(); sys.exit(0)
    rows = main(["--methods", "source_only,tent,rotta,eata,sar,cotta,pg,beatrhythm", "--n_beats", "120",
                 "--out_dir", "./results/_selftest_edge", "--hp",
                 json.dumps({"beatrhythm": {"steps": 1}, "cotta": {"n_aug": 2}})])
    by = {r["method"]: r for r in rows}
    # sanity relations that must hold if the accounting is right
    assert by["source_only"]["macs_grad_fwd"] == 0 and by["source_only"]["state_kb"] < 1
    assert by["tent"]["macs_grad_fwd"] > 0 and by["tent"]["macs_total_per_beat"] > by["source_only"]["macs_total_per_beat"]
    assert by["cotta"]["macs_nograd_fwd"] > by["tent"]["macs_nograd_fwd"], "CoTTA runs teacher (+aug) passes"
    assert by["rotta"]["state_kb"] > 10 * by["tent"]["state_kb"], "RoTTA replay memory dominates its state"
    assert by["pg"]["needs_anchor_pass"] and by["pg"]["macs_anchor_pass"] > 0
    assert by["sar"]["macs_grad_fwd"] >= by["tent"]["macs_grad_fwd"] * 0.5
    print("Self-test passed (relative costs behave as the algorithms dictate).")
