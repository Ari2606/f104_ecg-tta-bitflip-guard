"""
migrate_rr_cache.py  --  this project, one-off cache migration (v2: late-fusion schema)

Rewrites an existing preprocessed-beat cache into the CURRENT schema (pure waveform `x`
[N, n_leads, L] + a separate `rr` [N, 3] array, fused late in the model, after pooling --
see base_cnn.CNNConfig.aux_feat_dim), without touching raw MIT-BIH .dat/.hea/.atr files.

Handles BOTH possible source formats it might find in your cache_dir, auto-detected by
filename:

  (B) channel-injected format (tag ends "..._rr1{n}.npz", e.g. "..._rr13.npz") -- the
      first RR-features attempt, now superseded: x had the 3 RR values baked in as extra
      constant-valued conv-input channels (x: [N, n_leads+3, L]). That design measurably
      REGRESSED DS1-val macro-F1/precision/recall, so it's being replaced -- but its cache
      already contains everything needed for the new schema: since each extra channel is
      literally the same scalar repeated across all L samples, `x[:, n_leads:, 0]` IS the
      rr vector. No recomputation needed, this is a pure, instant slice.

  (A) the ORIGINAL pre-RR format (tag has no "_rr" segment at all) -- RR is computed from
      `samples` (R-peak indices, already stored) via `_rr_interval_features`, same as the
      first migration script did.

Whichever source is found for a given record, the result is written under the new
"..._rrL{n}.npz" tag (deliberately distinct from both older tags, so this can never
silently collide with a stale cache written by either earlier schema).

Usage (Colab, pointing at the same Drive cache_dir used for training):
    !python migrate_rr_cache.py --cache_dir /content/drive/MyDrive/ttaguard_out/cache

Safe to re-run: records already migrated, or with no matching source file, are skipped
and reported, not errored on.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from dataset import FS_MITBIH, _rr_interval_features

# (A) original pre-RR tag, e.g. "101_c1_p96_q160_record_bp1_0.5-40.0.npz"
_TAG_A_RE = re.compile(
    r"^(?P<record>\d+)_c(?P<n_leads>\d+)_p(?P<pre>\d+)_q(?P<post>\d+)_(?P<norm>\w+)"
    r"_bp(?P<bp>\d)_(?P<lo>[\d.]+)-(?P<hi>[\d.]+)\.npz$"
)
# (B) channel-injected tag, e.g. "101_c1_p96_q160_record_bp1_0.5-40.0_rr13.npz"
_TAG_B_RE = re.compile(
    r"^(?P<record>\d+)_c(?P<n_leads>\d+)_p(?P<pre>\d+)_q(?P<post>\d+)_(?P<norm>\w+)"
    r"_bp(?P<bp>\d)_(?P<lo>[\d.]+)-(?P<hi>[\d.]+)_rr1(?P<n_rr>\d+)\.npz$"
)


def _new_name(m: re.Match, n_rr_features: int) -> str:
    return (f"{m['record']}_c{m['n_leads']}_p{m['pre']}_q{m['post']}_{m['norm']}"
            f"_bp{m['bp']}_{m['lo']}-{m['hi']}_rrL{n_rr_features}.npz")


def migrate(cache_dir: str, n_rr_features: int = 3, dry_run: bool = False) -> None:
    cache = Path(cache_dir)
    if not cache.exists():
        raise FileNotFoundError(f"cache_dir '{cache_dir}' does not exist")

    files_b = {m["record"]: (f, m) for f in cache.iterdir()
              if (m := _TAG_B_RE.match(f.name))}
    files_a = {m["record"]: (f, m) for f in cache.iterdir()
              if (m := _TAG_A_RE.match(f.name))}

    records = set(files_b) | set(files_a)
    if not records:
        print(f"[migrate_rr_cache] no source cache files found under {cache_dir} -- "
              f"nothing to migrate (either already migrated, or the cache is empty and "
              f"you need a real --download run).")
        return

    migrated, skipped = 0, 0
    for record in sorted(records, key=int):
        # Prefer (B): it's an instant slice, no RR recomputation needed.
        if record in files_b:
            f, m, source = *files_b[record], "channel-injected"
        else:
            f, m, source = *files_a[record], "pre-RR"

        new_file = cache / _new_name(m, n_rr_features)
        if new_file.exists():
            skipped += 1
            continue

        d = np.load(f)
        y, samples = d["y"], d["samples"]
        n_leads = int(m["n_leads"])
        if len(samples) == 0:
            print(f"[migrate_rr_cache] record {record}: 0 beats, skipping.")
            skipped += 1
            continue

        if source == "channel-injected":
            x_full = d["x"]                                  # [N, n_leads+3, L]
            x_new = x_full[:, :n_leads, :]                    # pure waveform
            rr = x_full[:, n_leads:, 0].astype(np.float32)    # constant-per-channel -> scalar
        else:
            x_new = d["x"]                                    # already pure waveform
            rr = _rr_interval_features(samples, fs=float(FS_MITBIH))   # [N, 3]

        print(f"[migrate_rr_cache] record {record} (source={source}): {f.name}  "
              f"x={tuple(x_new.shape)}  rr={tuple(rr.shape)}  ->  {new_file.name}")
        if not dry_run:
            np.savez_compressed(new_file, x=x_new, y=y, samples=samples, rr=rr)
        migrated += 1

    print(f"[migrate_rr_cache] done: {migrated} record(s) migrated, {skipped} skipped "
          f"(already migrated or empty). {'(dry run -- nothing written)' if dry_run else ''}")
    print("[migrate_rr_cache] source files were left in place untouched; delete them "
          "yourself once you've confirmed training picks up the new '_rrL' cache, if "
          "you want to free Drive space.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--cache_dir", required=True,
                   help="the SAME --cache_dir you pass to train_stage1.py / eval_cross_domain.py")
    p.add_argument("--n_rr_features", type=int, default=3)
    p.add_argument("--dry_run", action="store_true",
                   help="print what would be migrated without writing any files")
    a = p.parse_args()
    migrate(a.cache_dir, n_rr_features=a.n_rr_features, dry_run=a.dry_run)
