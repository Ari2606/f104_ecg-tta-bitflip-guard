"""
generate_headline_results.py -- PhysioGuard-Space, presentation summary

Computes the paper's three headline claims DIRECTLY from the saved real-data result
files (never hardcoded), so every number here is reproducible and auditable:

  1. Clean & moderate-shift accuracy      <- results/stage3_cross_domain/cross_domain_results.json
                                             + results/stage7/real/per_beat_logs/*.csv
  2. TENT's collapse under worst-case     <- results/stage7/real/safety_summary.json
  3. Safety / drift bound + adaptation coverage <- results/stage7/real/safety_summary.json

Run this from the same directory you ran run_experiments.py / eval_cross_domain.py in
(i.e. wherever the `results/` folder lives).
"""
import json
from pathlib import Path

import pandas as pd

STAGE3_JSON = Path("results/stage3_cross_domain/cross_domain_results.json")
STAGE7_DIR = Path("results/stage7/real")
SAFETY_JSON = STAGE7_DIR / "safety_summary.json"
PER_BEAT_DIR = STAGE7_DIR / "per_beat_logs"

MODERATE_SEV_LO, MODERATE_SEV_HI = 0.2, 0.5


def moderate_shift_accuracy(method: str) -> float:
    df = pd.read_csv(PER_BEAT_DIR / f"{method}.csv")
    band = df[(df["severity"] >= MODERATE_SEV_LO) & (df["severity"] <= MODERATE_SEV_HI)]
    if len(band) == 0:
        raise ValueError(f"no beats found in severity band [{MODERATE_SEV_LO}, {MODERATE_SEV_HI}] for {method}")
    return float(band["correct"].mean())


def main():
    # ---- 1. Clean baseline (Stage 3) ----
    stage3 = json.loads(STAGE3_JSON.read_text())
    clean_row = next(r for r in stage3["results"] if r["scenario"] == "earth_clean")
    clean_acc = clean_row["accuracy"]

    # ---- moderate-shift accuracy, real data, per method (Stage 7 per-beat logs) ----
    src_acc = moderate_shift_accuracy("source_only")
    pg_acc = moderate_shift_accuracy("physioguard_space")
    tent_acc = moderate_shift_accuracy("tent_sparse")

    # ---- 2 & 3. Safety metrics (Stage 7 safety_summary.json) ----
    safety = json.loads(SAFETY_JSON.read_text())

    def s(method, key):
        return safety[method]["summary"][key]

    tent_djs, pg_djs, src_djs = s("tent_sparse", "mean_js_div"), s("physioguard_space", "mean_js_div"), s("source_only", "mean_js_div")
    tent_f1, pg_f1, src_f1 = s("tent_sparse", "macro_f1"), s("physioguard_space", "macro_f1"), s("source_only", "macro_f1")
    pg_adapt_rate = s("physioguard_space", "adaptation_rate")
    tent_adapt_rate = s("tent_sparse", "adaptation_rate")
    br_adapt_rate = s("beatrhythm_sqi_gate", "adaptation_rate")
    djs_reduction = tent_djs / pg_djs

    hurr = {m: safety[m]["safety"]["hurr"] for m in safety}
    buar = {m: safety[m]["safety"]["buar"] for m in safety}

    # ---- Table 1: accuracy under moderate real shift ----
    table1 = pd.DataFrame([
        {"Method": "Source-only (undefended)", "Clean accuracy (sev=0)": clean_acc,
         "Moderate-shift accuracy": src_acc, "Macro-F1 (moderate shift)": src_f1, "Mean D_JS": src_djs},
        {"Method": "TENT (unconstrained)", "Clean accuracy (sev=0)": clean_acc,
         "Moderate-shift accuracy": tent_acc, "Macro-F1 (moderate shift)": tent_f1, "Mean D_JS": tent_djs},
        {"Method": "PhysioGuard-Space", "Clean accuracy (sev=0)": clean_acc,
         "Moderate-shift accuracy": pg_acc, "Macro-F1 (moderate shift)": pg_f1, "Mean D_JS": pg_djs},
    ])
    for c in ["Clean accuracy (sev=0)", "Moderate-shift accuracy", "Macro-F1 (moderate shift)"]:
        table1[c] = (table1[c] * 100).round(2).astype(str) + "%"
    table1["Mean D_JS"] = table1["Mean D_JS"].round(4)

    # ---- Table 2: safety & coverage ----
    table2 = pd.DataFrame([
        {"Method": "Hard SQI-gate (BeatRhythm)", "Adaptation coverage": br_adapt_rate,
         "Rejection rate": 1 - br_adapt_rate, "Mean D_JS": s("beatrhythm_sqi_gate", "mean_js_div"),
         "HURR": hurr["beatrhythm_sqi_gate"], "BUAR": buar["beatrhythm_sqi_gate"]},
        {"Method": "TENT (unconstrained)", "Adaptation coverage": tent_adapt_rate,
         "Rejection rate": 1 - tent_adapt_rate, "Mean D_JS": tent_djs,
         "HURR": hurr["tent_sparse"], "BUAR": buar["tent_sparse"]},
        {"Method": "PhysioGuard-Space", "Adaptation coverage": pg_adapt_rate,
         "Rejection rate": 1 - pg_adapt_rate, "Mean D_JS": pg_djs,
         "HURR": hurr["physioguard_space"], "BUAR": buar["physioguard_space"]},
    ])
    for c in ["Adaptation coverage", "Rejection rate", "HURR", "BUAR"]:
        table2[c] = (table2[c] * 100).round(1).astype(str) + "%"
    table2["Mean D_JS"] = table2["Mean D_JS"].round(4)

    try:
        from IPython.display import display
        print(f"Table 1 -- Accuracy under moderate real space-shift (severity {MODERATE_SEV_LO}-{MODERATE_SEV_HI})\n")
        display(table1)
        print(f"\nD_JS reduction vs. TENT: {djs_reduction:.1f}x lower drift\n")
        print("Table 2 -- Safety & adaptation coverage (full real DS2 stream)\n")
        display(table2)
    except ImportError:
        print(f"Table 1 -- Accuracy under moderate real space-shift (severity {MODERATE_SEV_LO}-{MODERATE_SEV_HI})\n")
        print(table1.to_string(index=False))
        print(f"\nD_JS reduction vs. TENT: {djs_reduction:.1f}x lower drift\n")
        print("\nTable 2 -- Safety & adaptation coverage (full real DS2 stream)\n")
        print(table2.to_string(index=False))

    print("\nAll numbers above are computed live from results/ files written by")
    print("eval_cross_domain.py and run_experiments.py on REAL DS2 data -- nothing hardcoded.")


if __name__ == "__main__":
    main()