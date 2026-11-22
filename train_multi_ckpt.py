"""
train_multi_ckpt.py  --  independent source checkpoints (different training seeds) for the
checkpoint-level statistics in analyze_results.py.

A single checkpoint means every "significant" difference could be a property of one lucky or
unlucky model. Train N checkpoints with IDENTICAL flags except --seed, then pass them all to
run_benchmark.py (--checkpoint a.pt,b.pt,...). Use the same flags you used for your main
checkpoint (e.g. --narrow ...):

  !python train_multi_ckpt.py --seeds 0,1,2,3,4,5,6,7 --out_dir /content/drive/MyDrive/ttaguard_out/ckpts \
        --data_dir ... --cache_dir ... -- --narrow --epochs 40 --loss_type cb_focal

Everything after `--` goes to train_stage1.py verbatim. Existing checkpoints are skipped, so a
disconnected session resumes. With 8 checkpoints the smallest attainable checkpoint-level
two-sided p-value is 2^(1-8) = 0.0078; with 5 it is only 0.0625.
"""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--"); passthrough, argv = argv[i + 1:], argv[:i]
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = out / f"ckpt_seed{s}.pt"
        if ck.exists():
            print(f"[train_multi_ckpt] seed {s}: {ck} exists, skipping"); continue
        cmd = [sys.executable, "train_stage1.py", "--data_dir", a.data_dir, "--cache_dir", a.cache_dir,
               "--seed", str(s), "--checkpoint_path", str(ck), *passthrough]
        print("[train_multi_ckpt]", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)
