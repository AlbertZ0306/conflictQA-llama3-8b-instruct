#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize mean/std/range distributions by category_primary from analyze_c0_c3_distributions.py CSV.

For each of the five categories (c0_correct, c0_wrong_to_correct, c0_wrong,
c0_wrong_to_wrong, c0_correct_to_wrong) this script creates one PNG that
contains the distributions (boxplots) of mean/std/range using the phase that
matches the category semantics:
  - c0_correct, c0_wrong           -> use c0_mean/std/range
  - c0_wrong_to_correct,
    c0_wrong_to_wrong,
    c0_correct_to_wrong            -> use c3_mean/std/range

Usage:
  python conflictQA/scripts/plot_category_metrics.py \
    --csv conflictQA/results/llama3-8banalysis_c0_c3.csv \
    --out-dir conflictQA/plots/category_metrics
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


CATS_PHASE = {
    "c0_correct": "c0",
    "c0_wrong": "c0",
    "c0_wrong_to_correct": "c3",
    "c0_wrong_to_wrong": "c3",
    "c0_correct_to_wrong": "c3",
}


def load_rows(csv_path: Path) -> List[Dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def collect_by_category(rows: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        cat = row.get("category_primary")
        if cat not in CATS_PHASE:
            continue
        phase = CATS_PHASE[cat]
        metrics = {
            "mean": to_float(row.get(f"{phase}_mean")),
            "std": to_float(row.get(f"{phase}_std")),
            "range": to_float(row.get(f"{phase}_range")),
        }
        for k, v in metrics.items():
            if v is None:
                continue
            buckets.setdefault(cat, {}).setdefault(k, []).append(v)
    return buckets


def plot_category(cat: str, metrics: Dict[str, List[float]], out_dir: Path):
    labels = []
    data = []
    for metric in ("mean", "std", "range"):
        vals = metrics.get(metric, [])
        if vals:
            labels.append(metric)
            data.append(vals)
    if not data:
        print(f"[WARN] No data for category {cat}, skip plotting.")
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title(f"{cat} ({CATS_PHASE[cat]})")
    plt.ylabel("value")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    out_path = out_dir / f"{cat}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot mean/std/range per category_primary.")
    ap.add_argument("--csv", required=True, help="CSV from analyze_c0_c3_distributions.py")
    ap.add_argument("--out-dir", default="conflictQA/plots/category_metrics", help="Output directory for PNGs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    buckets = collect_by_category(rows)

    for cat in CATS_PHASE.keys():
        plot_category(cat, buckets.get(cat, {}), out_dir)


if __name__ == "__main__":
    main()
