"""
download_data.py  --  fetch the public PhysioNet data the real-shift experiments need.

  !pip install wfdb
  !python download_data.py --root /content/drive/MyDrive/ttaguard_data/data --which nstdb,incartdb,svdb

Creates <root>/nstdb (bw, em, ma only), <root>/incartdb (75 records) and <root>/svdb (78 records).
MIT-BIH itself is fetched by `python dataset.py --download`. Safe to re-run (wfdb skips files
that already exist). Requires internet access (Colab has it; the offline sandbox does not).
"""
import argparse
from pathlib import Path

from real_shift import download_external, download_nstdb

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--which", default="nstdb,incartdb,svdb")
    a = p.parse_args()
    for w in [x.strip() for x in a.which.split(",") if x.strip()]:
        d = str(Path(a.root) / w)
        print(f"[download_data] {w} -> {d}", flush=True)
        if w == "nstdb":
            download_nstdb(d)
        elif w in ("incartdb", "svdb"):
            download_external(w, d)
        else:
            raise SystemExit(f"unknown dataset '{w}'")
    print("[download_data] done")
