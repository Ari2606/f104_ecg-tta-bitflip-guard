"""
eval_source_model.py  --  how good is the SOURCE model, on the whole test set, with the
numbers a cardiology-ML reviewer looks for first?

Every TTA claim is only as credible as the model it adapts. This script evaluates frozen
checkpoints on the CLEAN, FULL DS2 (all 22 test records -- not a truncated stream) and
reports, in the vocabulary of the inter-patient literature:
  * per-class Sensitivity (recall), Positive Predictive value (precision), F1;
  * SVEB (class S) and VEB (class V) in the AAMI-recommended form;
  * overall accuracy, macro-F1, macro-F1 over N/S/V;
  * 95% record-level bootstrap intervals (22 patients is a small sample: report it);
  * a per-record table (which patients carry the S and V beats, where recall is won or lost);
  * a literature comparison, with an explicit list of what makes it NOT like-for-like.

Literature numbers below were checked against the cited papers' own tables or a paper that
reproduces them (source noted per row) -- re-verify before submission.
  de Chazal et al. 2004, DS2: N Se 86.9 / PPV 99.2; SVEB Se 75.9 / PPV 38.5; VEB Se 77.7 / PPV 81.9
      (as reproduced in Huang et al. 2014, Table 7).
  Huang et al. 2014 (BioMed Eng OnLine 13:90), DS2: SVEB Se 91.1 / PPV 42.2; VEB Se 93.9 / PPV 90.9.

Usage:
  !python eval_source_model.py --checkpoint a.pt[,b.pt,...] --data_dir ... --cache_dir ... \
      --out_dir ./results/source_model
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from dataset import CLASS_NAMES, NUM_CLASSES, ECGDataConfig, build_datasets
from eval_cross_domain import classification_report, load_checkpoint
from analyze_results import metrics_from_cms

K = NUM_CLASSES

# Published inter-patient DS1/DS2 results (percent). See module docstring for sources.
LITERATURE = [
    {"name": "de Chazal et al. 2004", "N": (86.9, 99.2), "SVEB": (75.9, 38.5), "VEB": (77.7, 81.9),
     "note": "RR + morphology features, LDA; numbers as reproduced in Huang et al. 2014, Table 7"},
    {"name": "Huang et al. 2014", "N": None, "SVEB": (91.1, 42.2), "VEB": (93.9, 90.9),
     "note": "random projections + RR, hierarchical classifier; own table"},
]


@torch.no_grad()
def predict(model, ds, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    use_aux = model.cfg.aux_feat_dim > 0
    out = []
    for i in range(0, len(ds), batch_size):
        x = ds.x[i:i + batch_size].to(device)
        aux = ds.rr[i:i + batch_size].to(device) if use_aux else None
        out.append(model(x, aux=aux).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def per_record_cms(y: np.ndarray, p: np.ndarray, rid: np.ndarray):
    recs = sorted(set(int(r) for r in rid))
    idx = {r: i for i, r in enumerate(recs)}
    cms = np.zeros((len(recs), K, K), dtype=np.int64)
    np.add.at(cms, (np.array([idx[int(r)] for r in rid]), y, p), 1)
    return recs, cms


def se_ppv(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    rep = classification_report(cm)["per_class"]
    return {c: {"Se": 100 * rep[c]["recall"], "PPV": 100 * rep[c]["precision"],
                "F1": 100 * rep[c]["f1"], "n": rep[c]["support"]} for c in CLASS_NAMES}


def bootstrap(cms: np.ndarray, n_boot: int, seed: int) -> Dict[str, tuple]:
    rng = np.random.default_rng(seed)
    R = cms.shape[0]
    w = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(np.float64)
    boot = np.einsum("br,rij->bij", w, cms)
    m = metrics_from_cms(boot)
    # SVEB / VEB positive predictivity are not in metrics_from_cms: compute here
    tp = np.einsum("bkk->bk", boot.astype(np.float64))
    pred = boot.sum(1).astype(np.float64)
    ppv = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    m["ppv_S"], m["ppv_V"] = ppv[:, 1], ppv[:, 2]
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in m.items()}


def evaluate_checkpoint(ckpt: str, data_dir: str, cache_dir: str, device: str, n_boot: int) -> dict:
    model, _, data_cfg, trained = load_checkpoint(ckpt, "cpu", 1)
    if not trained:
        raise RuntimeError(f"{ckpt}: not a trained checkpoint")
    data_cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": data_dir, "cache_dir": cache_dir})
    ds = build_datasets(data_cfg)["test"]
    return summarize_predictions(Path(ckpt).stem, ds.y.numpy(), predict(model, ds, device=device),
                                 ds.record_ids.numpy(), n_boot)


def summarize_predictions(tag: str, y: np.ndarray, p: np.ndarray, rid: np.ndarray, n_boot: int) -> dict:
    recs, cms = per_record_cms(y, p, rid)
    cm = cms.sum(0)
    rep = classification_report(cm)
    table = se_ppv(cm)
    per_rec = []
    for r, c in zip(recs, cms):
        sup = c.sum(1)
        per_rec.append({"record": r, "n": int(c.sum()), "S": int(sup[1]), "V": int(sup[2]), "F": int(sup[3]),
                        "acc": float(np.trace(c) / max(c.sum(), 1)),
                        "recall_S": float(c[1, 1] / sup[1]) if sup[1] else None,
                        "recall_V": float(c[2, 2] / sup[2]) if sup[2] else None})
    return {"ckpt": tag, "n_beats": int(cm.sum()), "n_records": len(recs), "accuracy": rep["accuracy"],
            "macro_f1": rep["macro_f1"], "macro_f1_ex_fq": rep["macro_f1_ex_fq"], "per_class": table,
            "confusion": cm.tolist(), "per_record": per_rec, "ci": bootstrap(cms, n_boot, 0)}


def render(results: List[dict]) -> str:
    L = []
    for r in results:
        pc = r["per_class"]
        L += [f"### {r['ckpt']}: {r['n_beats']} beats, {r['n_records']} records",
              f"accuracy {100*r['accuracy']:.1f}% [{100*r['ci']['accuracy'][0]:.1f}, {100*r['ci']['accuracy'][1]:.1f}], "
              f"macro-F1 {r['macro_f1']:.3f}, macro-F1 (N/S/V) {r['macro_f1_ex_fq']:.3f} "
              f"[{r['ci']['macro_f1_ex_fq'][0]:.3f}, {r['ci']['macro_f1_ex_fq'][1]:.3f}]", "",
              "| class | n | Se % | PPV % | F1 % |", "|---|---|---|---|---|"]
        for c in CLASS_NAMES:
            L.append(f"| {c} | {pc[c]['n']} | {pc[c]['Se']:.1f} | {pc[c]['PPV']:.1f} | {pc[c]['F1']:.1f} |")
        L += ["", "Per-record (S/V-bearing records only):", "", "| record | beats | S | V | acc | recall S | recall V |",
              "|---|---|---|---|---|---|---|"]
        for q in r["per_record"]:
            if q["S"] or q["V"]:
                fmt = lambda v: "-" if v is None else f"{v:.2f}"
                L.append(f"| {q['record']} | {q['n']} | {q['S']} | {q['V']} | {q['acc']:.2f} | "
                         f"{fmt(q['recall_S'])} | {fmt(q['recall_V'])} |")
        L.append("")
    # literature
    mean = {c: {k: float(np.mean([r["per_class"][c][k] for r in results])) for k in ("Se", "PPV")} for c in CLASS_NAMES}
    L += ["### Literature comparison (inter-patient, DS1 -> DS2; percent)", "",
          "| method | N Se/PPV | SVEB Se/PPV | VEB Se/PPV |", "|---|---|---|---|"]
    for e in LITERATURE:
        f = lambda t: "-" if t is None else f"{t[0]:.1f} / {t[1]:.1f}"
        L.append(f"| {e['name']} | {f(e['N'])} | {f(e['SVEB'])} | {f(e['VEB'])} |")
    L.append(f"| **ours (source model, mean of {len(results)} ckpt)** | {mean['N']['Se']:.1f} / {mean['N']['PPV']:.1f} | "
             f"{mean['S']['Se']:.1f} / {mean['S']['PPV']:.1f} | {mean['V']['Se']:.1f} / {mean['V']['PPV']:.1f} |")
    L += ["", "NOT like-for-like: (i) published systems use hand-crafted RR+morphology features, ours is a "
          "<512 KB int8 CNN; (ii) papers differ in how F and Q are handled; (iii) published numbers are single runs "
          "without patient-level intervals; (iv) lead choice and preprocessing differ."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> List[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True, help="path or comma-separated list")
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--out_dir", default="./results/source_model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args(argv)
    res = [evaluate_checkpoint(c.strip(), a.data_dir, a.cache_dir, a.device, a.n_boot)
           for c in a.checkpoint.split(",")]
    md = render(res)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "source_model.md").write_text(md)
    (out / "source_model.json").write_text(json.dumps(res, indent=1))
    print(md)
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        sys.exit(0)
    rng = np.random.default_rng(0)
    R, per = 10, 300
    rid = np.repeat(np.arange(R) + 100, per)
    y = rng.choice(K, size=R * per, p=[0.7, 0.1, 0.15, 0.03, 0.02])
    p_ = np.where(rng.random(R * per) < 0.85, y, rng.integers(0, K, R * per))
    r1 = summarize_predictions("synthetic_a", y, p_, rid, 300)
    r2 = summarize_predictions("synthetic_b", y, np.where(rng.random(R * per) < 0.8, y, rng.integers(0, K, R * per)), rid, 300)
    # consistency: per-class tables must agree with the project's report
    rep = classification_report(np.array(r1["confusion"]))
    assert abs(r1["per_class"]["V"]["Se"] - 100 * rep["per_class"]["V"]["recall"]) < 1e-9
    assert abs(r1["accuracy"] - float((y == p_).mean())) < 1e-9
    lo, hi = r1["ci"]["accuracy"]
    assert lo < r1["accuracy"] < hi and sum(q["n"] for q in r1["per_record"]) == R * per
    md = render([r1, r2])
    assert "de Chazal" in md and "Huang" in md and "NOT like-for-like" in md and "| V |" in md
    print(md[:900])
    print("Self-test passed.")
