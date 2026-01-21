#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a diff CSV from analyze_c0_c3_distributions output.

Columns:
- question
- c3_mean_minus_c0_mean
- c3_std_minus_c0_std
- c3_range_minus_c0_range
- category_primary

Usage:
  python conflictQA/scripts/make_c0_c3_diff_csv.py \
    --csv conflictQA/results/llama3-8banalysis_c0_c3.csv \
    --out conflictQA/results/llama3-8banalysis_c0_c3_diff.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def to_float(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def load_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    ap = argparse.ArgumentParser(description="Make diff CSV between c3 and c0 metrics.")
    ap.add_argument("--csv", required=True, help="Input CSV from analyze_c0_c3_distributions.py")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out)

    rows = load_rows(in_path)

    out_rows = []
    for r in rows:
        c0_mean = to_float(r.get("c0_mean"))
        c3_mean = to_float(r.get("c3_mean"))
        c0_std = to_float(r.get("c0_std"))
        c3_std = to_float(r.get("c3_std"))
        c0_range = to_float(r.get("c0_range"))
        c3_range = to_float(r.get("c3_range"))

        out_rows.append(
            {
                "question": r.get("question", ""),
                "c3_mean_minus_c0_mean": None if c0_mean is None or c3_mean is None else c3_mean - c0_mean,
                "c3_std_minus_c0_std": None if c0_std is None or c3_std is None else c3_std - c0_std,
                "c3_range_minus_c0_range": None if c0_range is None or c3_range is None else c3_range - c0_range,
                "category_primary": r.get("category_primary", ""),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "c3_mean_minus_c0_mean",
        "c3_std_minus_c0_std",
        "c3_range_minus_c0_range",
        "category_primary",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
