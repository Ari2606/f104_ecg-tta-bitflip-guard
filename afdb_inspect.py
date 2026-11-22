"""
afdb_inspect.py -- what rhythm content does the MIT-BIH Atrial Fibrillation Database (AFDB) hold, per record?
Reads only headers + annotations (no signal files needed). Prints seconds in each rhythm (AFIB / AFL / N / J / other)
and which beat-annotation files (qrs, qrsc) exist, so the inter-patient split for the rhythm task can be fixed
BEFORE any model is trained (class counts only; nothing about model performance).

  python afdb_inspect.py --afdb_dir /content/local/afdb --out /content/afdb_summary.csv
"""
import argparse
import csv
from pathlib import Path

import wfdb


def segments(ann, n_samples):
    idx = [i for i, s in enumerate(ann.symbol) if s == "+"]
    segs = []
    for k, i in enumerate(idx):
        lab = (ann.aux_note[i] or "").strip("\x00").strip()
        end = ann.sample[idx[k + 1]] if k + 1 < len(idx) else n_samples
        segs.append((lab, int(ann.sample[i]), int(end)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afdb_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.afdb_dir)
    recs = sorted(p.stem for p in d.glob("*.atr"))
    rows = []
    for r in recs:
        hdr = wfdb.rdheader(str(d / r))
        fs, n = float(hdr.fs), int(hdr.sig_len)
        ann = wfdb.rdann(str(d / r), "atr")
        secs = {}
        for lab, s, e in segments(ann, n):
            secs[lab] = secs.get(lab, 0.0) + (e - s) / fs
        row = {"record": r, "fs": fs, "hours": round(n / fs / 3600, 2),
               "has_qrs": (d / f"{r}.qrs").exists(), "has_qrsc": (d / f"{r}.qrsc").exists(),
               "has_dat": (d / f"{r}.dat").exists()}
        for lab in ("(AFIB", "(AFL", "(N", "(J"):
            row[lab[1:] + "_s"] = round(secs.get(lab, 0.0))
        row["other_s"] = round(sum(v for k, v in secs.items() if k not in ("(AFIB", "(AFL", "(N", "(J")))
        rows.append(row)
    cols = list(rows[0].keys())
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    tot = {k: sum(r[k] for r in rows) for k in ("AFIB_s", "AFL_s", "N_s", "J_s", "other_s")}
    print("TOTAL seconds:", tot, " records:", len(rows))


if __name__ == "__main__":
    main()
