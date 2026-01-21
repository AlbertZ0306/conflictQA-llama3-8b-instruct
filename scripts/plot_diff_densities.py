#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot density (KDE) of metric deltas per category_primary from diff CSV, all in one figure.

Input CSV columns (from make_c0_c3_diff_csv.py):
  question, c3_mean_minus_c0_mean, c3_std_minus_c0_std, c3_range_minus_c0_range, category_primary

Layout:
  rows   = categories that have any data
  cols   = 3 metrics (mean/std/range deltas)
  Each cell: KDE curve with shaded area and μ/σ annotation.

Categories without any data are omitted entirely.

Usage:
  python conflictQA/scripts/plot_diff_densities.py \
    --csv conflictQA/results/llama3-8banalysis_c0_c3_diff.csv \
    --out conflictQA/plots/diff_densities_all.png
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

CATEGORIES_ORDER = [
    "c0_correct",
    "c0_wrong_to_correct",
    "c0_wrong",
    "c0_wrong_to_wrong",
    "c0_correct_to_wrong",
]
METRICS = ["mean", "std", "range"]


def to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def load_by_category(csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category_primary")
            if not cat:
                continue
            for metric in METRICS:
                key = f"c3_{metric}_minus_c0_{metric}"
                val = to_float(row.get(key))
                if val is None:
                    continue
                buckets.setdefault(cat, {}).setdefault(metric, []).append(val)
    return buckets


def kde_1d(values: List[float], xs: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE with Silverman's rule of thumb for bandwidth."""
    if not values:
        return np.zeros_like(xs)
    data = np.array(values, dtype=float)
    n = data.size
    if n == 1:
        return np.zeros_like(xs)
    std = np.std(data, ddof=1)
    if std == 0:
        std = max(1e-6, np.mean(np.abs(data)) * 1e-3)
    bw = 1.06 * std * (n ** (-1 / 5))
    if bw <= 0:
        bw = 1e-6
    diff = (xs[:, None] - data[None, :]) / bw
    kernel = np.exp(-0.5 * diff ** 2)
    density = kernel.mean(axis=1) / (bw * math.sqrt(2 * math.pi))
    return density


def plot_all(buckets: Dict[str, Dict[str, List[float]]], out_path: Path):
    cats_with_data = [
        cat for cat in CATEGORIES_ORDER
        if any(buckets.get(cat, {}).get(metric) for metric in METRICS)
    ]
    if not cats_with_data:
        print("No data found in CSV. Check file and columns.")
        return

    rows = len(cats_with_data)
    cols = len(METRICS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), squeeze=False)

    for r, cat in enumerate(cats_with_data):
        metrics = buckets.get(cat, {})
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            vals = metrics.get(metric, [])
            if not vals:
                ax.set_visible(False)
                continue

            xmin, xmax = min(vals), max(vals)
            span = xmax - xmin
            padding = 0.1 * span if span > 0 else 0.1
            xs = np.linspace(xmin - padding, xmax + padding, 300)
            ys = kde_1d(vals, xs)
            ax.plot(xs, ys, color="#4C72B0")
            ax.fill_between(xs, ys, alpha=0.2, color="#4C72B0")
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.text(
                0.97,
                0.95,
                f"μ={mu:.3f}\nσ={sigma:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
            )
            ax.grid(True, linestyle="--", alpha=0.3)
            if c == 0:
                ax.set_ylabel(cat)
            if r == 0:
                ax.set_title(f"{metric} Δ (c3 - c0)")
            else:
                ax.set_title(f"{metric} Δ")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot density of c3-c0 metric deltas per category in one figure.")
    ap.add_argument("--csv", required=True, help="Input diff CSV (from make_c0_c3_diff_csv.py)")
    ap.add_argument("--out", default="conflictQA/plots/diff_densities_all.png", help="Output PNG path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    buckets = load_by_category(csv_path)
    plot_all(buckets, out_path)


if __name__ == "__main__":
    main()
