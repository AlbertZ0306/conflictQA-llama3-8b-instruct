#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot accuracy for c0_results.json grouped by mean/std/range bins.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


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


def summarize_metric(
    results: List[Dict[str, Any]],
    metric_name: str,
    edges: Sequence[float],
) -> Tuple[List[int], List[int], List[float]]:
    totals = [0 for _ in range(len(edges) - 1)]
    corrects = [0 for _ in range(len(edges) - 1)]
    accuracies = [0.0 for _ in range(len(edges) - 1)]

    for row in results:
        metric_value = row.get(metric_name)
        bin_idx = assign_bin(metric_value, edges)
        if bin_idx is None:
            continue
        totals[bin_idx] += 1
        if row.get("correct"):
            corrects[bin_idx] += 1

    for i in range(len(accuracies)):
        if totals[i]:
            accuracies[i] = corrects[i] / totals[i]
    return totals, corrects, accuracies


def make_labels(edges: Sequence[float]) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            labels.append(f"[{start:.2f}, {end:.2f}]")
        else:
            labels.append(f"[{start:.2f}, {end:.2f})")
    return labels


def plot_metric(
    metric_name: str,
    edges: Sequence[float],
    totals: List[int],
    accuracies: List[float],
    output_dir: Path,
):
    labels = make_labels(edges)
    positions = range(len(labels))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(positions, accuracies, color="#4C72B0")
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{metric_name.capitalize()} bin accuracy")

    for bar, acc, total in zip(bars, accuracies, totals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}\n(n={total})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    output_path = output_dir / f"{metric_name}_accuracy.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {metric_name} plot to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot accuracy vs bins for c0 results.")
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
        totals, corrects, accuracies = summarize_metric(results, metric_name, edges)
        print(f"Metric: {metric_name}")
        for label, total, correct, acc in zip(make_labels(edges), totals, corrects, accuracies):
            print(f"  {label}: total={total}, correct={correct}, accuracy={acc:.2f}")
        plot_metric(metric_name, edges, totals, accuracies, output_dir)


if __name__ == "__main__":
    main()

