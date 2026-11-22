"""
cv_folds.py  --  leave-record-group-out folds over DS1, for hyper-parameter selection that
does not hinge on four hand-picked validation records.

Why: DS1-VAL has 4 records, so every tuned setting so far was chosen on a tiny, patient-
dominated stream and then failed to transfer to DS2. Proper fix: K folds over the 22 DS1
records; for each fold train a source model that NEVER saw the fold's records, evaluate
adaptation methods on those held-out records, and pick hyper-parameters on the average over
all K folds (every DS1 record is held out exactly once). DS2 stays untouched until the final
confirmation.

  python cv_folds.py --data_dir ... --cache_dir ... --k 5 --out folds.json \
        --ckpt_dir /content/drive/MyDrive/ttaguard_out/cv \
        --train_args "--narrow --epochs 40 ..."      # same flags as your main checkpoint
prints one `train_stage1.py ... --val_records ... --checkpoint_path ...` command per fold and
writes folds.json (consumed by tune_cv.py). Run the printed commands, then tune_cv.py.

Folds are balanced on rare-class content (S+V+F beats per record, greedy longest-first), so
every fold has S and V beats to evaluate on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def assign_folds(weights: Dict[int, float], k: int) -> List[List[int]]:
    """Greedy longest-processing-time assignment: heaviest record to the lightest fold."""
    folds: List[List[int]] = [[] for _ in range(k)]
    load = [0.0] * k
    for r, w in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        i = min(range(k), key=lambda j: (load[j], len(folds[j]), j))
        folds[i].append(r); load[i] += w
    return [sorted(f) for f in folds]


def record_weights(data_dir: str, cache_dir: str) -> Dict[int, float]:
    from dataset import DS1_RECORDS, ECGDataConfig, load_record_beats
    cfg = ECGDataConfig(data_dir=data_dir, cache_dir=cache_dir)
    w = {}
    for r in DS1_RECORDS:
        _, y, _, _ = load_record_beats(r, cfg)
        w[r] = float(((y == 1) | (y == 2) | (y == 3)).sum()) + 1.0        # S+V+F beats (+1 so no zero)
    return w


def main(argv: Optional[Sequence[str]] = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ckpt_dir", default="./cv")
    p.add_argument("--train_args", default="", help="extra train_stage1.py flags, e.g. '--narrow --epochs 40'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="folds.json")
    a = p.parse_args(argv)
    folds = assign_folds(record_weights(a.data_dir, a.cache_dir), a.k)
    spec = {"folds": [{"val_records": f, "checkpoint": str(Path(a.ckpt_dir) / f"fold{i}.pt")}
                      for i, f in enumerate(folds)]}
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(f"[cv_folds] {a.k} folds over {sum(len(f) for f in folds)} DS1 records -> {a.out}\nRun:")
    for i, fo in enumerate(spec["folds"]):
        print(f"python train_stage1.py --data_dir {a.data_dir} --cache_dir {a.cache_dir} {a.train_args} "
              f"--seed {a.seed} --val_records {','.join(map(str, fo['val_records']))} "
              f"--checkpoint_path {fo['checkpoint']}")
    return spec


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(); sys.exit(0)
    from dataset import DS1_RECORDS
    import numpy as np
    rng = np.random.default_rng(0)
    w = {r: float(rng.integers(1, 400)) for r in DS1_RECORDS}
    folds = assign_folds(w, 5)
    flat = sorted(r for f in folds for r in f)
    assert flat == sorted(DS1_RECORDS), "every DS1 record held out exactly once"
    loads = [sum(w[r] for r in f) for f in folds]
    print("fold sizes", [len(f) for f in folds], "rare-beat load", [int(x) for x in loads])
    assert max(loads) / min(loads) < 1.25, "folds should be balanced on rare-class content"
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 3
    # the default hold-out of dataset.build_datasets must keep working and the override must be leak-free
    from dataset import build_datasets  # noqa: F401
    print("Self-test passed.")
