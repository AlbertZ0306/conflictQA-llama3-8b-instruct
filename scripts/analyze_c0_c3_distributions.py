#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare C0 (no context) vs C3 (with aligned context) token-prob metrics.

Per-question categories:
- c0_correct: C0 already correct (ignores what C3 did).
- c0_wrong: C0 wrong (ignores what C3 did).
- c0_wrong_to_correct: C0 wrong -> C3 correct.
- c0_wrong_to_wrong: C0 wrong -> C3 still wrong.
- c0_correct_to_wrong: C0 correct -> C3 regressed (not requested, but reported for completeness).

Outputs:
- A CSV with one row per question containing both C0/C3 metrics so you can
  slice distributions however you like.
- Console summary with basic stats (mean/median) of mean/std/range for each
  category (using the phase that best matches the category: C0 metrics for
  c0_correct / c0_wrong; C3 metrics for the transition buckets).

Usage:
  python conflictQA/scripts/analyze_c0_c3_distributions.py \
    --c0 conflictQA/results/llama3-8b-instruct/c0.json \
    --c3 conflictQA/results/llama3-8b-instruct/c3.json \
    --out conflictQA/results/analysis_c0_c3.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def basic_stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(statistics.mean(values)), float(statistics.median(values))


def summarize(category: str, phase: str, rows: List[Dict], metric: str):
    """
    Print basic stats for a category.
    phase: 'c0' or 'c3' (which metric fields to read).
    metric: 'mean' | 'std' | 'range'
    """
    key = f"{phase}_{metric}"
    vals = [r[key] for r in rows if r.get(key) is not None]
    avg, med = basic_stats(vals)
    print(f"[{category}][{phase}] {metric}: n={len(vals)}, mean={avg}, median={med}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", required=True, help="Path to c0.json (no context)")
    ap.add_argument("--c3", required=True, help="Path to c3.json (with aligned context)")
    ap.add_argument("--out", default="conflictQA/results/analysis_c0_c3.csv", help="Output CSV path")
    args = ap.parse_args()

    c0_path = Path(args.c0)
    c3_path = Path(args.c3)
    out_path = Path(args.out)

    c0_rows = load_results(c0_path)
    c3_rows = load_results(c3_path)

    c0_map = {r.get("question"): r for r in c0_rows if "question" in r}
    c3_map = {r.get("question"): r for r in c3_rows if "question" in r}

    questions = set(c0_map.keys()) & set(c3_map.keys())
    missing_c3 = set(c0_map.keys()) - questions
    missing_c0 = set(c3_map.keys()) - questions
    if missing_c3:
        print(f"Skip {len(missing_c3)} questions missing in C3.")
    if missing_c0:
        print(f"Skip {len(missing_c0)} questions missing in C0.")

    merged: List[Dict] = []
    for q in sorted(questions):
        r0 = c0_map[q]
        r3 = c3_map[q]
        merged.append(
            {
                "question": q,
                "c0_correct": bool(r0.get("correct")),
                "c3_correct": bool(r3.get("correct")),
                "c0_mean": r0.get("mean"),
                "c0_std": r0.get("std"),
                "c0_range": r0.get("range"),
                "c3_mean": r3.get("mean"),
                "c3_std": r3.get("std"),
                "c3_range": r3.get("range"),
            }
        )

    # Buckets
    buckets: Dict[str, List[Dict]] = {
        "c0_correct": [],
        "c0_wrong": [],
        "c0_wrong_to_correct": [],
        "c0_wrong_to_wrong": [],
        "c0_correct_to_wrong": [],
    }
    for row in merged:
        if row["c0_correct"]:
            buckets["c0_correct"].append(row)
            if not row["c3_correct"]:
                buckets["c0_correct_to_wrong"].append(row)
        else:
            buckets["c0_wrong"].append(row)
            if row["c3_correct"]:
                buckets["c0_wrong_to_correct"].append(row)
            else:
                buckets["c0_wrong_to_wrong"].append(row)

    # Write CSV so you can plot distributions as needed.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "c0_correct",
        "c3_correct",
        "c0_mean",
        "c0_std",
        "c0_range",
        "c3_mean",
        "c3_std",
        "c3_range",
        "category_primary",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            if row["c0_correct"]:
                cat = "c0_correct"
            else:
                cat = "c0_wrong_to_correct" if row["c3_correct"] else "c0_wrong_to_wrong"
            writer.writerow({**row, "category_primary": cat})

    print(f"Wrote merged CSV to {out_path} (rows={len(merged)})")

    # Console summaries (focus on distributions you described)
    print("\n== Summaries ==")
    # 1) before context correct -> use c0 metrics
    summarize("c0_correct", "c0", buckets["c0_correct"], "mean")
    summarize("c0_correct", "c0", buckets["c0_correct"], "std")
    summarize("c0_correct", "c0", buckets["c0_correct"], "range")

    # 2) before context wrong -> use c0 metrics
    summarize("c0_wrong", "c0", buckets["c0_wrong"], "mean")
    summarize("c0_wrong", "c0", buckets["c0_wrong"], "std")
    summarize("c0_wrong", "c0", buckets["c0_wrong"], "range")

    # 3) before wrong, after correct -> use c3 metrics
    summarize("c0_wrong_to_correct", "c3", buckets["c0_wrong_to_correct"], "mean")
    summarize("c0_wrong_to_correct", "c3", buckets["c0_wrong_to_correct"], "std")
    summarize("c0_wrong_to_correct", "c3", buckets["c0_wrong_to_correct"], "range")

    # 4) before wrong, after still wrong -> use c3 metrics
    summarize("c0_wrong_to_wrong", "c3", buckets["c0_wrong_to_wrong"], "mean")
    summarize("c0_wrong_to_wrong", "c3", buckets["c0_wrong_to_wrong"], "std")
    summarize("c0_wrong_to_wrong", "c3", buckets["c0_wrong_to_wrong"], "range")

    # Optional: show regressions
    if buckets["c0_correct_to_wrong"]:
        summarize("c0_correct_to_wrong", "c3", buckets["c0_correct_to_wrong"], "mean")
        summarize("c0_correct_to_wrong", "c3", buckets["c0_correct_to_wrong"], "std")
        summarize("c0_correct_to_wrong", "c3", buckets["c0_correct_to_wrong"], "range")


if __name__ == "__main__":
    main()
