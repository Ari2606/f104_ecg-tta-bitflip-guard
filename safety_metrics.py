"""
safety_metrics.py  --  this project, Stage 7 (Safety & Robustness Metrics)

Turns the raw per-beat logs any `streaming_eval.StreamingResult` already produces
into the paper's headline safety metrics and publication artifacts:

  1. Harmful Update Rejection Ratio (HURR)  -- on beats that were genuinely
     corrupted, how much did the method pull its adaptation strength BACK?
  2. Benign Update Acceptance Ratio (BUAR)  -- the necessary complement: on
     beats that were genuinely clean, did it keep adapting, or over-gate?
     HURR alone can't distinguish a well-calibrated controller from one that
     is simply always off (Source-Only gets a trivial HURR=1.0 by refusing
     every update, clean or not -- BUAR=0.0 for the same reason exposes that).
  3. Long-horizon drift curves -- D_JS and rolling accuracy over a full
     continuous stream (10,000 beats by convention), to show WHETHER a
     method's divergence from the frozen source model re-stabilizes
     (this project, SQI-gate) or keeps climbing (TENT).
  4. Publication artifacts -- a LaTeX results table (headline metrics +
     HURR/BUAR) and a drift-curve figure (matplotlib PNG), built directly
     from `StreamingResult.summary` / `.log`, so a rerun regenerates both
     without hand-editing.

Ground-truth caveat (read before using HURR/BUAR on anything but this
simulated evaluation)
---------------------------------------------------------------------------
HURR/BUAR need to know, beat-by-beat, whether a beat was ACTUALLY corrupted.
That is available here only because `space_shift_simulator` stamps an oracle
`corrupt_mask` into each beat's `meta`, which `streaming_eval.StreamingEvaluator`
already reduces to a `corrupt_frac` column in every log row (originally added
there just as a diagnostic). On a real deployed stream there is no such
oracle. HURR/BUAR are therefore SIMULATION-ONLY evaluation metrics -- exactly
like needing ground-truth labels to report accuracy -- not a signal available
to the controller itself, which must decide using only its own unsupervised
TS-SQI / D_JS / entropy signals. They judge the controller from the outside.

"Adaptation weight" per beat
------------------------------
Different methods express "how much did I trust this beat" differently:
this project reports a continuous `extra_mask_coverage` in [0, 1]
(`safety_controller.GuardedTTA`'s per-beat SQI coverage); TENT and
SQI-gate only have the boolean `adapted` flag (SQI-gate's whole-beat
gate has no partial credit by design -- see `baselines_tta.py`). `_adapt_weight`
below uses the continuous signal when a method provides one and falls back to
the boolean otherwise, so HURR/BUAR are computed on the SAME footing across
every method without penalizing a boolean-gated one for lacking a signal it
was never designed to produce.

Shape legend: n/a (this module operates on already-logged scalar-per-beat data).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from streaming_eval import StreamingResult


# --------------------------------------------------------------------------- #
# 1. Harmful Update Rejection Ratio / Benign Update Acceptance Ratio
# --------------------------------------------------------------------------- #
def _adapt_weight(row: Dict) -> float:
    """
    [0,1] "how much this method trusted/updated on this beat", best signal available,
    preferred in this order:
      1. `extra_inband_mask_coverage` (this project) -- mask mean restricted to
         in-band STFT bins. Far more discriminative than the whole-spectrum mean
         (see `safety_controller.GuardedTTA.__init__`'s docstring for why: at
         n_fft=32/fs=360 Hz, most STFT bins sit outside the 0.5-40 Hz ECG passband
         by construction, so the whole-spectrum mean barely moves with corruption).
      2. `extra_mask_coverage` -- fallback if only the whole-spectrum figure was
         logged (e.g. an older run / a different masked-TTA method).
      3. the boolean `adapted` flag -- TENT, SQI-gate, Source-Only, whose gates
         (or lack of one) have no partial credit by design.
    """
    ic = row.get("extra_inband_mask_coverage")
    if ic is not None:
        return float(np.clip(ic, 0.0, 1.0))
    mc = row.get("extra_mask_coverage")
    if mc is not None:
        return float(np.clip(mc, 0.0, 1.0))
    return 1.0 if row.get("adapted") else 0.0


@dataclass
class SafetyMetrics:
    method_name: str
    hurr: Optional[float]                # None if no corrupt_frac was logged (no simulator used)
    buar: Optional[float]
    n_corrupt_beats: int
    n_clean_beats: int
    corrupt_thresh: float
    clean_thresh: float
    reset_count: int
    reset_rate: float
    max_js_div: Optional[float]


def calibrate_thresholds(results: "Dict[str, StreamingResult]", corrupt_quantile: float = 0.9,
                         clean_quantile: float = 0.3) -> Optional[Dict[str, float]]:
    """
    Data-driven HURR/BUAR thresholds, calibrated from the ACTUAL `corrupt_frac`
    distribution observed across every method's log (pooled, since `corrupt_frac`
    depends only on the shared simulator/severity schedule and beat position, not
    on any one method's predictions, so pooling is valid and gives a more stable
    quantile estimate than calibrating per-method).

    A fixed absolute threshold (e.g. "corrupt_frac >= 0.3") is NOT portable across
    simulator configurations: `space_shift_simulator`'s corruption events are
    localized within a 256-sample window rather than covering most of it, so
    `corrupt_frac` can realistically top out well under 0.3 for an entire stream
    -- a hardcoded 0.3 threshold silently matched ZERO beats the first time this
    was run (caught by this file's own self-test, not a hypothetical). Quantiles
    of the OBSERVED distribution always populate both buckets (barring a
    pathological all-identical stream), and adapt automatically to a harsher or
    gentler severity schedule.
    Returns None if no method logged a `corrupt_frac` (no simulator was used).
    """
    pooled = np.array([r["corrupt_frac"] for res in results.values() for r in res.log
                       if r.get("corrupt_frac") is not None], dtype=np.float64)
    if pooled.size == 0:
        return None
    corrupt_thresh = float(np.quantile(pooled, corrupt_quantile))
    clean_thresh = float(np.quantile(pooled, clean_quantile))
    if not (clean_thresh < corrupt_thresh):
        # degenerate distribution (e.g. almost every beat has the same corrupt_frac):
        # fall back to the median as a single separating threshold for both directions.
        med = float(np.median(pooled))
        clean_thresh = corrupt_thresh = med
    return {"corrupt_thresh": corrupt_thresh, "clean_thresh": clean_thresh}


def compute_safety_metrics(result: StreamingResult, corrupt_thresh: float,
                           clean_thresh: float) -> SafetyMetrics:
    """
    HURR = mean(1 - adapt_weight) over beats with corrupt_frac >= corrupt_thresh.
           1.0 = fully withheld adaptation on every genuinely-corrupted beat (ideal);
           0.0 = adapted at full strength regardless of corruption (TENT's failure mode).
    BUAR = mean(adapt_weight) over beats with corrupt_frac <= clean_thresh.
           1.0 = kept adapting at full strength on every genuinely-clean beat (ideal);
           0.0 = never adapts at all, even when perfectly safe to (over-cautious / useless).
    A method is only "safe AND useful" if both are high -- see module docstring.
    Thresholds are required (not silently defaulted): call `calibrate_thresholds` on
    the full set of methods' results first (see `save_all_artifacts`), since a fixed
    absolute default is not portable across simulator/severity configurations (see
    `calibrate_thresholds`'s docstring for why this file's own self-test caught that).
    """
    log = result.log
    rows_with_gt = [r for r in log if r.get("corrupt_frac") is not None]
    if not rows_with_gt:
        return SafetyMetrics(result.method_name, None, None, 0, 0, corrupt_thresh, clean_thresh,
                             result.summary["reset_count"], result.summary["reset_rate"],
                             result.summary["max_js_div"])
    corrupt_rows = [r for r in rows_with_gt if r["corrupt_frac"] >= corrupt_thresh]
    clean_rows = [r for r in rows_with_gt if r["corrupt_frac"] <= clean_thresh]
    hurr = float(np.mean([1.0 - _adapt_weight(r) for r in corrupt_rows])) if corrupt_rows else None
    buar = float(np.mean([_adapt_weight(r) for r in clean_rows])) if clean_rows else None
    return SafetyMetrics(result.method_name, hurr, buar, len(corrupt_rows), len(clean_rows),
                         corrupt_thresh, clean_thresh, result.summary["reset_count"],
                         result.summary["reset_rate"], result.summary["max_js_div"])


# --------------------------------------------------------------------------- #
# 2. Long-horizon drift curves
# --------------------------------------------------------------------------- #
@dataclass
class DriftCurve:
    method_name: str
    step: np.ndarray
    js_div: np.ndarray            # NaN where a method logged no anchor divergence
    roll_accuracy: np.ndarray
    cum_accuracy: np.ndarray
    severity: np.ndarray          # NaN where no simulator was used


def drift_curve(result: StreamingResult) -> DriftCurve:
    log = result.log
    step = np.array([r["step"] for r in log], dtype=np.int64)
    js = np.array([r["js_div"] if r["js_div"] is not None else np.nan for r in log], dtype=np.float64)
    roll_acc = np.array([r["roll_accuracy"] for r in log], dtype=np.float64)
    cum_acc = np.array([r["cum_accuracy"] for r in log], dtype=np.float64)
    sev = np.array([r["severity"] if r.get("severity") is not None else np.nan for r in log], dtype=np.float64)
    return DriftCurve(result.method_name, step, js, roll_acc, cum_acc, sev)


def plot_drift_curves(curves: Sequence[DriftCurve], out_path: str, rolling_window_label: str = "") -> str:
    """Two-panel figure: rolling accuracy (top) and D_JS, log-scale (bottom), one line per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_acc, ax_js) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    for c in curves:
        ax_acc.plot(c.step, c.roll_accuracy, label=c.method_name, linewidth=1.2)
    ax_acc.set_ylabel(f"rolling accuracy {rolling_window_label}".strip())
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.legend(loc="lower left", fontsize=8, ncol=2)
    ax_acc.grid(alpha=0.3)
    ax_acc.set_title("this project -- long-horizon streaming TTA under space-shift onset")

    eps = 1e-12
    for c in curves:
        ax_js.plot(c.step, np.clip(c.js_div, eps, None), label=c.method_name, linewidth=1.2)
    ax_js.set_yscale("log")
    ax_js.set_xlabel("beat (chronological stream position)")
    ax_js.set_ylabel(r"$D_{JS}$(adapted $\Vert$ frozen anchor), nats [log scale]")
    ax_js.grid(alpha=0.3, which="both")

    # mark the shift-onset ramp region if any curve carries a severity schedule
    for c in curves:
        finite = np.isfinite(c.severity)
        if finite.any() and np.nanmax(c.severity) > 0:
            onset = int(c.step[finite][np.argmax(c.severity[finite] > 0)])
            for ax in (ax_acc, ax_js):
                ax.axvline(onset, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_acc.text(onset, 1.0, "  shift onset", fontsize=7, color="gray", va="top")
            break

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 3. Publication artifacts: LaTeX results table
# --------------------------------------------------------------------------- #
def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _fmt_pct(x: Optional[float]) -> str:
    return f"{100 * x:.1f}" if x is not None else "--"


def latex_results_table(results: Dict[str, StreamingResult], safety: Dict[str, SafetyMetrics],
                        caption: str = "Streaming test-time adaptation under simulated spaceflight "
                                       "sensor/physiology shift (DS2, N=1 online protocol).",
                        label: str = "tab:ttaguard_results") -> str:
    """Renders the headline accuracy/drift/latency table AND the HURR/BUAR safety
    table used in the this project paper, as one `table*` LaTeX environment."""
    header = (r"\begin{table*}[t]" "\n"
             r"\centering" "\n"
             r"\begin{tabular}{lrrrrrrrrr}" "\n"
             r"\toprule" "\n"
             r"Method & Acc.\ (\%) & Macro-F1 (\%) & Macro-F1 ex-FQ (\%) & Mean $D_{JS}$ & Adapt.\ (\%) "
             r"& HURR (\%) & BUAR (\%) & Resets & $p_{95}$ lat.\ (ms) \\" "\n"
             r"\midrule")
    rows = []
    for name, r in results.items():
        s = r.summary
        sm = safety[name]
        djs = "n/a" if s["mean_js_div"] is None else f"{s['mean_js_div']:.2e}"
        exfq = s.get("macro_f1_ex_fq")
        exfq_str = "n/a" if exfq is None else f"{100*exfq:.1f}"
        rows.append(
            f"{_latex_escape(name)} & {100*s['accuracy']:.1f} & {100*s['macro_f1']:.1f} & {exfq_str} & "
            f"{djs} & {100*s['adaptation_rate']:.1f} & {_fmt_pct(sm.hurr)} & {_fmt_pct(sm.buar)} & "
            f"{s['reset_count']} & {s['latency_ms_p95']:.2f} \\\\"
        )
    footer = (r"\bottomrule" "\n"
             r"\end{tabular}" "\n"
             f"\\caption{{{caption}}}" "\n"
             f"\\label{{{label}}}" "\n"
             r"\end{table*}")
    return "\n".join([header, *rows, footer])


def save_all_artifacts(results: Dict[str, StreamingResult], out_dir: str,
                       corrupt_thresh: Optional[float] = None, clean_thresh: Optional[float] = None,
                       rolling_window: Optional[int] = None) -> Dict[str, str]:
    """
    Computes safety metrics for every method, writes the LaTeX table, the drift-curve
    PNG, and a machine-readable JSON summary, all under `out_dir`. Returns their paths.
    `corrupt_thresh`/`clean_thresh` default to None, meaning: auto-calibrate from the
    pooled observed `corrupt_frac` distribution via `calibrate_thresholds` (recommended
    -- see its docstring for why a fixed absolute default is not portable).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if corrupt_thresh is None or clean_thresh is None:
        calib = calibrate_thresholds(results)
        corrupt_thresh = calib["corrupt_thresh"] if calib else 0.3
        clean_thresh = calib["clean_thresh"] if calib else 0.05

    safety = {name: compute_safety_metrics(r, corrupt_thresh, clean_thresh) for name, r in results.items()}
    curves = [drift_curve(r) for r in results.values()]

    tex = latex_results_table(results, safety)
    tex_path = out / "results_table.tex"
    tex_path.write_text(tex)

    label = f"(w={rolling_window})" if rolling_window else ""
    fig_path = out / "drift_curves.png"
    plot_drift_curves(curves, str(fig_path), rolling_window_label=label)

    summary_json = {
        name: {"summary": r.summary, "safety": asdict(safety[name])}
        for name, r in results.items()
    }
    json_path = out / "safety_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, default=float))

    return {"latex_table": str(tex_path), "drift_curves_png": str(fig_path), "summary_json": str(json_path)}


# --------------------------------------------------------------------------- #
# Self-test: synthetic 4-method comparison (reusing baselines_tta / safety_controller's
# own synthetic-stream self-test setup) exercising HURR/BUAR, drift curves and both
# publication artifacts end-to-end, without needing real MIT-BIH data.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import copy
    import torch
    import torch.nn.functional as F

    from base_cnn import CNNConfig, BeatCNN
    from baselines_tta import (SQIGatedTent, SourceOnlyTTA, TentConfig, TentTTA,
                               _SyntheticStreamDataset)
    from safety_controller import GuardedTTA, SafetyControllerConfig
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
    from streaming_eval import StreamingConfig, StreamingEvaluator, compare_methods, make_beat_stream

    print("[safety_metrics] self-test: HURR/BUAR + drift curves + LaTeX/PNG artifacts on a "
         "synthetic, space-shifted 4-method comparison stream.")
    torch.manual_seed(0)
    source_model = BeatCNN(CNNConfig(in_channels=1))
    ds = _SyntheticStreamDataset(n_records=3, beats_per_record=400, c=1, l=256, seed=0)

    # mini-batched warm-up before freeze_anchor(), same established rationale as every
    # other file in this project (batch-size-1 online SGD does not converge here).
    opt = torch.optim.Adam(list(source_model.backbone.parameters())
                           + list(source_model.adaptable_head.parameters()), lr=1e-3)
    source_model.train()
    for _ in range(15):
        perm = torch.randperm(len(ds.y))
        for i in range(0, len(ds.y), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(source_model(ds.x[idx]).logits, ds.y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    source_model.eval()
    source_model.freeze_anchor()

    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=0))
    severity_schedule = linear_ramp(start_idx=150, n_ramp=500)     # calm calibration window, then ramps
    cfg = StreamingConfig(device="cpu", rolling_window=150, log_every=0)

    methods = [SourceOnlyTTA(), TentTTA(TentConfig(scope="sparse")), SQIGatedTent(),
              GuardedTTA(SafetyControllerConfig())]
    results: Dict[str, StreamingResult] = {}
    for method in methods:
        m = copy.deepcopy(source_model)
        stream = make_beat_stream(ds, n_beats=None, sim=sim, severity=severity_schedule)
        ev = StreamingEvaluator(method, cfg)
        results[method.name] = ev.run(m, stream)

    print()
    print(compare_methods(results))

    calib = calibrate_thresholds(results)
    assert calib is not None, "SpaceShiftSimulator was used -- corrupt_frac must be present in every log"
    print(f"\nauto-calibrated thresholds (from the observed corrupt_frac distribution): "
         f"corrupt_thresh={calib['corrupt_thresh']:.4f}  clean_thresh={calib['clean_thresh']:.4f}")
    safety = {name: compute_safety_metrics(r, calib["corrupt_thresh"], calib["clean_thresh"])
             for name, r in results.items()}
    print("\nmethod               HURR(%)  BUAR(%)  n_corrupt  n_clean")
    print("-" * 60)
    for name, sm in safety.items():
        print(f"{name:<20} {_fmt_pct(sm.hurr):>7}  {_fmt_pct(sm.buar):>7}  "
             f"{sm.n_corrupt_beats:>9}  {sm.n_clean_beats:>7}")

    # ---- 1. every method logged an oracle corrupt_frac (SpaceShiftSimulator was used),
    # and the auto-calibrated thresholds actually populate BOTH buckets with a
    # meaningful number of beats (this is exactly the check that would have caught
    # the earlier hardcoded-0.3-threshold bug, which silently zeroed out n_corrupt).
    n_beats = len(next(iter(results.values())).log)
    for name, sm in safety.items():
        assert sm.hurr is not None and sm.buar is not None, f"{name}: expected simulator ground truth"
        assert 0.0 <= sm.hurr <= 1.0 and 0.0 <= sm.buar <= 1.0
        assert sm.n_corrupt_beats >= 0.02 * n_beats, \
            f"{name}: too few corrupt-bucket beats ({sm.n_corrupt_beats}) -- threshold miscalibrated"
        assert sm.n_clean_beats >= 0.02 * n_beats, \
            f"{name}: too few clean-bucket beats ({sm.n_clean_beats}) -- threshold miscalibrated"
    print("\nHURR/BUAR well-formed and both buckets adequately populated for every method -> OK")

    # ---- 2. TENT (no safety mechanism, unconditional adaptation) must show the WORST
    # (lowest) HURR of the three adapting methods -- it never withholds an update
    # regardless of corruption, which is exactly the naive failure mode this paper's
    # safety mechanisms exist to fix.
    tent_hurr = safety["tent_sparse"].hurr
    pg_hurr = safety["ttaguard"].hurr
    br_hurr = safety["sqi_gated_tent"].hurr
    assert tent_hurr < pg_hurr and tent_hurr < br_hurr, \
        "TENT's unconditional adaptation must show the lowest HURR of the adapting methods"
    print(f"TENT HURR={100*tent_hurr:.1f}% is the lowest among adapting methods -> OK "
         f"(this project={100*pg_hurr:.1f}%, SQIgate={100*br_hurr:.1f}%)")

    # ---- 3. this project's soft, sub-window masking should let it keep adapting
    # on clean beats (high BUAR) MORE than SQI-gate's blunt whole-beat gate does: a
    # clean beat SQI-gate's gate happens to reject loses 100% of its update, while
    # this project's mask on a genuinely clean beat stays near-full coverage (see
    # ts_sqi_masker.py's own self-test: ~0.7-0.9 in-band on a clean beat). This is the
    # real, asymmetric trade-off soft masking is FOR, and it shows up here as a lower
    # HURR than SQI-gate's (soft partial pullback, not a hard reject -- do not expect
    # HURR parity with a whole-beat gate) paired with a substantially higher BUAR.
    pg_buar, br_buar = safety["ttaguard"].buar, safety["sqi_gated_tent"].buar
    print(f"BUAR: this project={100*pg_buar:.1f}%  SQIgate={100*br_buar:.1f}%")
    assert pg_buar > br_buar, \
        "this project's soft masking must preserve more clean-beat adaptation than SQI-gate's blunt gate"

    # ---- 4. Source-Only is the trivial degenerate case: perfect HURR (never adapts,
    # so it can't commit a harmful update) but zero BUAR (never adapts on clean beats
    # either) -- confirms HURR alone is not a sufficient safety metric (module docstring).
    assert safety["source_only"].hurr == 1.0 and safety["source_only"].buar == 0.0, \
        "Source-Only must show the trivial HURR=1.0 / BUAR=0.0 degenerate case"
    print("Source-Only trivial case (HURR=100%, BUAR=0%) confirmed -> OK "
         "(demonstrates HURR alone is not sufficient; BUAR is the necessary complement)")

    # ---- 5. drift curves + publication artifacts, end to end ----
    out_dir = "./results/_selftest_safety_metrics"
    paths = save_all_artifacts(results, out_dir, rolling_window=cfg.rolling_window)
    for k, p in paths.items():
        assert Path(p).exists() and Path(p).stat().st_size > 0, f"{k} was not written: {p}"
    tex = Path(paths["latex_table"]).read_text()
    assert r"\begin{table*}" in tex and "ttaguard" in tex and "HURR" in tex
    print(f"\nwrote LaTeX table, drift-curve PNG, and JSON summary to {out_dir}/ -> OK")

    print("\nSelf-test passed.")
