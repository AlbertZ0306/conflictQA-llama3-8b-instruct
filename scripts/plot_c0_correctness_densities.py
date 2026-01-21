#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot KDE density of mean/std/range for C0 results, split by correctness.

Rows: correctness buckets (correct / incorrect, omitted if empty)
Cols: mean, std, range

Input JSON: C0 output array with fields: correct, mean, std, range

Usage:
  python conflictQA/scripts/plot_c0_correctness_densities.py \
    --c0 conflictQA/results/llama3-8b-instruct/c0.json \
    --out conflictQA/plots/c0_correctness_densities.png
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

METRICS = ["mean", "std", "range"]


def kde_1d(values: List[float], xs: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE with Silverman's rule of thumb."""
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


def load_buckets(c0_path: Path) -> Dict[str, Dict[str, List[float]]]:
    buckets: Dict[str, Dict[str, List[float]]] = {"correct": {}, "incorrect": {}}
    with c0_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        ok = bool(row.get("correct"))
        bucket = "correct" if ok else "incorrect"
        for m in METRICS:
            val = row.get(m)
            if val is None:
                continue
            buckets[bucket].setdefault(m, []).append(float(val))
    return buckets


def plot_all(buckets: Dict[str, Dict[str, List[float]]], out_path: Path):
    cats_with_data = [cat for cat in ["correct", "incorrect"] if any(buckets.get(cat, {}).get(m) for m in METRICS)]
    if not cats_with_data:
        print("No data to plot (no metrics found).")
        return

    rows = len(cats_with_data)
    cols = len(METRICS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), squeeze=False)

    for r, cat in enumerate(cats_with_data):
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            vals = buckets.get(cat, {}).get(metric, [])
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
                ax.set_title(metric)
            else:
                ax.set_title(metric)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot density of mean/std/range for C0 by correctness.")
    ap.add_argument("--c0", required=True, help="Path to c0.json (array of results)")
    ap.add_argument("--out", default="conflictQA/plots/c0_correctness_densities.png", help="Output PNG path")
    args = ap.parse_args()

    c0_path = Path(args.c0)
    out_path = Path(args.out)

    buckets = load_buckets(c0_path)
    plot_all(buckets, out_path)


if __name__ == "__main__":
    main()
