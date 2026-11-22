"""
method_zoo.py  --  one registry of every method / ablation / control the paper compares.

`build_method(key, hp)` returns a fresh `TTAMethod`. Keys are the stable names used in
result files and tables (the instances' own `.name` strings are internal). Defaults are
the settings we report: baselines use the best DS1-VAL setting found by tune_baselines.py
where one exists (see comments); this project uses its untuned defaults (a tuned
setting regressed on DS2, which the paper reports).

Groups (CLI `--methods main,ablation` or explicit keys):
  main      source_only tent sqi_gated_tent rotta beatrhythm eata sar cotta pg
  variants  pg_gate                        (operating point: skip adapting on abnormal beats)
  ablation  pg_nomask pg_notr pg_fixed_a0.02 pg_fixed_a0.1
            tent_l2_1 tent_l2_10 tent_l2_100 tent_reset500 tent_reset2000
  extra     eata_fisher                    (needs source beats -> run_benchmark supplies them)

Per-method hyper-parameter overrides: hp = {"rotta": {"lr": 3e-4}, "pg": {"tau": 0.1}}.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from baselines_tta import (BeatRhythmTTA, BeatRhythmTTAConfig, RoTTAConfig, RoTTATTA,
                           SourceOnlyTTA, SQIGatedTent, SQIGatedTentConfig, TentConfig, TentTTA)
from extra_baselines import (CoTTAConfig, CoTTATTA, EATAConfig, EATATTA, SARConfig, SARTTA,
                             TentRegConfig, TentRegularizedTTA)
from bn_stats_tta import BNStatsConfig, BNStatsTTA
from safety_controller import GuardedTTA, SafetyControllerConfig
from streaming_eval import TTAMethod

# key -> (factory(hp_dict) -> TTAMethod, needs_source_beats)
_REG: Dict[str, "tuple[Callable[[dict], TTAMethod], bool]"] = {}


def _reg(key: str, fn: Callable[[dict], TTAMethod], needs_source: bool = False) -> None:
    _REG[key] = (fn, needs_source)


_reg("source_only", lambda hp: SourceOnlyTTA())
_reg("tent", lambda hp: TentTTA(TentConfig(**{"scope": "sparse", "lr": 1e-3, **hp})))     # val plateau: lr>=1e-3
_reg("sqi_gated_tent", lambda hp: SQIGatedTent(SQIGatedTentConfig(**hp)))
_reg("rotta", lambda hp: RoTTATTA(RoTTAConfig(**{"lr": 1e-3, "ema_alpha": 0.999,
                                                  "update_every": 4, **hp})))             # DS1-VAL best
_reg("beatrhythm", lambda hp: BeatRhythmTTA(BeatRhythmTTAConfig(**{"lr": 3e-3, "steps": 5, **hp})))  # DS1-VAL best
_reg("eata", lambda hp: EATATTA(EATAConfig(**hp)))
_reg("eata_fisher", lambda hp: EATATTA(EATAConfig(**{"fisher_alpha": 100.0, **hp})), needs_source=True)
_reg("sar", lambda hp: SARTTA(SARConfig(**hp)))
_reg("cotta", lambda hp: CoTTATTA(CoTTAConfig(**hp)))
_reg("pg", lambda hp: GuardedTTA(SafetyControllerConfig(**hp)))
_reg("pg_gate", lambda hp: GuardedTTA(SafetyControllerConfig(**{"adapt_gate": "skip_abnormal", **hp})))
_reg("pg_cb", lambda hp: GuardedTTA(SafetyControllerConfig(**{"class_balance": True, **hp})))
_reg("pg_kl", lambda hp: GuardedTTA(SafetyControllerConfig(**{"anchor_kl": 1.0, **hp})))
_reg("pg_cbkl", lambda hp: GuardedTTA(SafetyControllerConfig(**{"class_balance": True, "anchor_kl": 1.0, **hp})))
_reg("bn_ema", lambda hp: BNStatsTTA(BNStatsConfig(**{"momentum": 0.005, **hp})))
_reg("bn_ema_fast", lambda hp: BNStatsTTA(BNStatsConfig(**{"momentum": 0.02, **hp})))
_reg("bn_ema_dw", lambda hp: BNStatsTTA(BNStatsConfig(**{"momentum": 0.005, "scope": "dw_only", **hp})))
_reg("pg_guard", lambda hp: GuardedTTA(SafetyControllerConfig(**{"fault_guard": True, **hp})))
_reg("pg_cbkl_guard", lambda hp: GuardedTTA(SafetyControllerConfig(**{"class_balance": True, "anchor_kl": 1.0, "fault_guard": True, **hp})))
for _gk in (1, 2, 3):                      # guard-box width sweep (std units); default pg_cbkl_guard is k=4
    _reg(f"pg_cbkl_guard_k{_gk}", (lambda k: lambda hp: GuardedTTA(SafetyControllerConfig(**{
        "class_balance": True, "anchor_kl": 1.0, "fault_guard": True, "guard_k": float(k), **hp})))(_gk))
_reg("guard_only", lambda hp: GuardedTTA(SafetyControllerConfig(**{"fault_guard": True, "adapt_enabled": False, "fixed_alpha": 0.0, **hp})))
_reg("pull_guard", lambda hp: GuardedTTA(SafetyControllerConfig(**{"fault_guard": True, "adapt_enabled": False, **hp})))
_reg("pg_nomask",lambda hp: GuardedTTA(SafetyControllerConfig(**{"use_mask": False, **hp})))
_reg("pg_notr", lambda hp: GuardedTTA(SafetyControllerConfig(**{"gamma": 0.0, **hp})))
_reg("pg_fixed_a0.02", lambda hp: GuardedTTA(SafetyControllerConfig(**{"fixed_alpha": 0.02, **hp})))
_reg("pg_fixed_a0.1", lambda hp: GuardedTTA(SafetyControllerConfig(**{"fixed_alpha": 0.1, **hp})))
for _lam in (1, 10, 100):
    _reg(f"tent_l2_{_lam}", (lambda lam: lambda hp: TentRegularizedTTA(
        TentRegConfig(**{"l2_lambda": float(lam), **hp})))(_lam))
for _n in (500, 2000):
    _reg(f"tent_reset{_n}", (lambda n: lambda hp: TentRegularizedTTA(
        TentRegConfig(**{"reset_every": n, **hp})))(_n))

GROUPS: Dict[str, List[str]] = {
    "main": ["source_only", "tent", "sqi_gated_tent", "rotta", "beatrhythm", "eata", "sar", "cotta", "pg"],
    "variants": ["pg_gate", "pg_cb", "pg_kl", "pg_cbkl"],
    "ablation": ["pg_nomask", "pg_notr", "pg_fixed_a0.02", "pg_fixed_a0.1",
                 "tent_l2_1", "tent_l2_10", "tent_l2_100", "tent_reset500", "tent_reset2000"],
    "extra": ["eata_fisher"],
    "bnstats": ["bn_ema", "bn_ema_fast", "bn_ema_dw"],
    "fault": ["pg_guard", "pg_cbkl_guard", "pg_cbkl_guard_k1", "pg_cbkl_guard_k2", "pg_cbkl_guard_k3",
              "pull_guard", "guard_only"],              # parameter-fault protection (run_fault_experiment.py)
}
ALL_KEYS: List[str] = list(_REG)


def expand(spec: str) -> List[str]:
    """'main,pg_gate,ablation' -> ordered, de-duplicated list of method keys."""
    out: List[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        for k in (GROUPS[tok] if tok in GROUPS else [tok]):
            if k not in _REG:
                raise KeyError(f"unknown method '{k}'. Known: {ALL_KEYS} and groups {list(GROUPS)}")
            if k not in out:
                out.append(k)
    return out


def needs_source_beats(key: str) -> bool:
    return _REG[key][1]


def build_method(key: str, hp: Optional[Dict[str, dict]] = None) -> TTAMethod:
    fn, _ = _REG[key]
    return fn(dict((hp or {}).get(key, {})))


if __name__ == "__main__":
    print(f"[method_zoo] {len(ALL_KEYS)} methods: {ALL_KEYS}")
    for k in ALL_KEYS:
        m = build_method(k)
        assert isinstance(m, TTAMethod) and m.name, k
    assert expand("main") == GROUPS["main"] and expand("pg_gate,main")[0] == "pg_gate"
    assert build_method("rotta", {"rotta": {"lr": 5e-4}}).cfg.lr == 5e-4
    assert build_method("pg_notr").cfg.gamma == 0.0 and build_method("pg_nomask").cfg.use_mask is False
    assert build_method("tent_l2_10").cfg.l2_lambda == 10.0 and build_method("tent_reset500").cfg.reset_every == 500
    try:
        expand("nope"); raise SystemExit("expected KeyError")
    except KeyError:
        pass
    print("Self-test passed.")
