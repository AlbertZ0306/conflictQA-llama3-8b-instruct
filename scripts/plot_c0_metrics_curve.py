#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot metric value densities for mean/std/range.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_bins(text: str) -> List[float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError("At least two bin edges are required.")
    if sorted(parts) != parts:
        raise ValueError("Bin edges must be sorted in ascending order.")
    return parts


def assign_bin(value: Optional[float], edges: Sequence[float]) -> Optional[int]:
    if value is None:
        return None
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if start <= value <= end:
                return i
        else:
            if start <= value < end:
                return i
    return None


def collect_metric_values(
    metric_name: str,
    results: List[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    values_all: List[float] = []
    values_correct: List[float] = []

    for row in results:
        metric_value = row.get(metric_name)
        if metric_value is None:
            continue
        try:
            value = float(metric_value)
        except (TypeError, ValueError):
            continue
        values_all.append(value)
        if row.get("correct"):
            values_correct.append(value)

    return values_all, values_correct


def kde_1d(values: List[float], xs: np.ndarray) -> np.ndarray:
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
        bw = 1e-3
    diff = (xs[:, None] - data[None, :]) / bw
    kernel = np.exp(-0.5 * diff ** 2)
    density = kernel.mean(axis=1) / (bw * math.sqrt(2 * math.pi))
    return density


def plot_metric_density(
    metric_name: str,
    edges: Sequence[float],
    values_all: List[float],
    values_correct: List[float],
    output_dir: Path,
):
    min_edge = min(edges)
    max_edge = max(edges)
    xs = np.linspace(min_edge, max_edge, 300)

    density_all = kde_1d(values_all, xs)
    density_correct = kde_1d(values_correct, xs)

    plt.figure(figsize=(10, 5))
    plt.plot(xs, density_all, color="#4C72B0", label="all")
    plt.plot(xs, density_correct, color="#55A868", label="correct")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.title(f"{metric_name.capitalize()} density")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    output_path = output_dir / f"{metric_name}_density.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {metric_name} density to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot metric densities for c0 results.")
    ap.add_argument("--input", default="conflictQA/c0_results.json", help="Path to c0_results JSON file")
    ap.add_argument("--output-dir", default="conflictQA/plots", help="Directory for generated plots")
    ap.add_argument("--mean-bins", default="0.5,0.6,0.7,0.8,0.9,1.0", help="Comma-separated bin edges for mean")
    ap.add_argument("--std-bins", default="0,0.1,0.2,0.3,0.4,0.5,0.6", help="Comma-separated bin edges for std")
    ap.add_argument("--range-bins", default="0,0.2,0.4,0.6,0.8,1.0", help="Comma-separated bin edges for range")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    configs = [
        ("mean", parse_bins(args.mean_bins)),
        ("std", parse_bins(args.std_bins)),
        ("range", parse_bins(args.range_bins)),
    ]

    for metric_name, edges in configs:
        values_all, values_correct = collect_metric_values(metric_name, results)
        print(f"Metric: {metric_name} (n_all={len(values_all)}, n_correct={len(values_correct)})")
        plot_metric_density(metric_name, edges, values_all, values_correct, output_dir)


if __name__ == "__main__":
    main()
