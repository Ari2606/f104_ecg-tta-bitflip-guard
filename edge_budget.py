"""
edge_budget.py  --  every edge-deployability number that needs NO hardware, stated for what it is.

Reports for one checkpoint (or --smoke):
  * parameters, fp32 size, int8 size (1 byte/weight + 4 B bias/scale overhead estimate) -- parameter-only figures;
  * MAC count per beat (conv / linear, counted with hooks) -- hardware independent;
  * peak activation memory per beat (largest sum of live tensors, fp32 and int8) -- an upper-bound style estimate;
  * runtime buffers the streaming pipeline needs besides the network (RR history, patient-template buffer);
  * writable adapted-state size (the dw_bn set TTA keeps in RAM) -- the memory exposed to upsets;
  * latency ESTIMATES = MACs / assumed sustained throughput, for a labelled range of device classes.
    These are scenarios, NOT measurements. Measured host latency (export_onnx.py + bench_pi.py on a laptop) is
    a different quantity and must be labelled 'host-measured, not on target'.

  python edge_budget.py --checkpoint ckpt.pt          |   python edge_budget.py --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

# sustained MAC/s scenarios (order-of-magnitude assumptions, NOT measured, NOT vendor figures)
SCENARIOS = {"small MCU class (~5 MMAC/s)": 5e6, "tiny Linux SBC class (~100 MMAC/s)": 1e8,
             "Pi-4 class single core (~500 MMAC/s)": 5e8}


def count_macs(model: nn.Module, x: torch.Tensor, aux: Optional[torch.Tensor]) -> Dict:
    macs, live, hooks = {"conv": 0, "linear": 0}, [], []

    def conv_hook(m, i, o):
        macs["conv"] += int(o.numel() * (m.in_channels // m.groups) * m.kernel_size[0])
        live.append(o.numel())

    def lin_hook(m, i, o):
        macs["linear"] += int(m.in_features * m.out_features)
        live.append(o.numel())

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model.eval()(x, aux=aux)
    for h in hooks:
        h.remove()
    return {"macs_conv": macs["conv"], "macs_linear": macs["linear"], "macs_total": macs["conv"] + macs["linear"],
            "largest_activation_elems": max(live) if live else 0, "sum_activation_elems": sum(live)}


def budget(model: nn.Module, in_channels: int, aux_dim: int, window: int, n_template: int = 0) -> Dict:
    params = sum(p.numel() for p in model.parameters())
    bufs = sum(b.numel() * b.element_size() for b in model.buffers() if b.dtype.is_floating_point)
    x = torch.zeros(1, in_channels, window)
    aux = torch.zeros(1, aux_dim) if aux_dim > 0 else None
    m = count_macs(model, x, aux)
    tta = sum(p.numel() for _, p in model.tta_named_parameters()) if hasattr(model, "tta_named_parameters") else 0
    r = {"params": params, "fp32_KB": (params * 4 + bufs) / 1024, "int8_param_only_KB": params / 1024,
         "int8_est_KB_with_scales": (params * 1 + params * 0.05 * 4) / 1024, **m,
         "peak_act_int8_KB_upper": m["sum_activation_elems"] / 1024, "peak_act_fp32_KB_upper": m["sum_activation_elems"] * 4 / 1024,
         "adapted_state_values": tta, "adapted_state_fp32_B": tta * 4,
         "rr_history_buffer_B": 400 * 4 if aux_dim >= 7 else 10 * 4,
         "template_buffer_fp32_B": n_template * window * 4 if n_template else 0,
         "estimated_latency_ms": {k: 1e3 * m["macs_total"] / v for k, v in SCENARIOS.items()}}
    return r


def render(r: Dict) -> str:
    L = ["| quantity | value |", "|---|---|",
         f"| parameters | {r['params']:,} |", f"| fp32 size (params+BN buffers) | {r['fp32_KB']:.1f} KB |",
         f"| int8, parameter-only | {r['int8_param_only_KB']:.1f} KB |",
         f"| int8 incl. ~5% scale/zero-point overhead (estimate) | {r['int8_est_KB_with_scales']:.1f} KB |",
         f"| MACs per beat | {r['macs_total']:,} (conv {r['macs_conv']:,}, linear {r['macs_linear']:,}) |",
         f"| activation memory per beat, summed upper bound | {r['peak_act_int8_KB_upper']:.1f} KB int8 / {r['peak_act_fp32_KB_upper']:.1f} KB fp32 |",
         f"| writable adapted state (exposed to upsets) | {r['adapted_state_values']} values = {r['adapted_state_fp32_B']} B |",
         f"| streaming buffers: RR history / patient template | {r['rr_history_buffer_B']} B / {r['template_buffer_fp32_B']} B |"]
    for k, v in r["estimated_latency_ms"].items():
        L.append(f"| ESTIMATED latency, {k} | {v:.2f} ms/beat (MACs / assumed throughput; not measured) |")
    return "\n".join(L) + "\n\nThe real int8 ONNX file is larger than the parameter-only figure (graph + quantization metadata); " \
           "report the measured file size from export_onnx.py. Latency rows are scenarios, not measurements.\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="edge_budget.json")
    a = ap.parse_args(argv)
    if a.smoke or not a.checkpoint:
        from base_cnn import CNNConfig, BeatCNN
        cfg = CNNConfig(in_channels=1, aux_feat_dim=7, stem_ch=16, b1_branch_ch=16, b2_branch_ch=32, feat_dim=64,
                        head_hidden=32, use_se=False)
        model = BeatCNN(cfg).eval(); model.freeze_anchor()
        r = budget(model, 1, 7, 256, 0)
    else:
        from eval_cross_domain import load_checkpoint
        model, cnn_cfg, data_cfg, ok = load_checkpoint(a.checkpoint, "cpu", 1)
        assert ok, "not a trained checkpoint"
        r = budget(model, data_cfg.in_channels, data_cfg.aux_feat_dim, data_cfg.window_len, data_cfg.n_template)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
