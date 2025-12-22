#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from run_c0_vllm import (
    call_vllm_chat,
    extract_token_logprobs_from_chat,
    iter_jsonl,
    logprobs_to_probs,
    parse_pred_label,
    stats_from_probs,
)


def process_one(
    pos: int,
    idx: int,
    ex: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[int, int, Dict[str, Any], Optional[str]]:
    question = ex.get("question", "")
    gt_list = ex.get("ground_truth", [])
    gt = gt_list[0] if isinstance(gt_list, list) and len(gt_list) > 0 else None
    if isinstance(gt, str):
        gt = gt.capitalize()

    try:
        resp = call_vllm_chat(
            base_url=args.base_url,
            model=args.model,
            question=question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        text = resp["choices"][0]["message"]["content"]
    except Exception as e:
        result = {
            "question": question,
            "answer": "",
            "correct": False,
            "token_number_of_answer": 0,
            "mean": None,
            "std": None,
            "range": None,
        }
        return pos, idx, result, str(e)

    pred = parse_pred_label(text)
    correct = (pred is not None and gt is not None and pred == gt)

    logprobs = extract_token_logprobs_from_chat(resp)
    probs = logprobs_to_probs(logprobs)
    n_tokens, mean, std, rng = stats_from_probs(probs)

    result = {
        "question": question,
        "answer": text,
        "correct": bool(correct),
        "token_number_of_answer": n_tokens,
        "mean": mean,
        "std": std,
        "range": rng,
    }
    return pos, idx, result, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to conflictQA-strategyQA-chatgpt.json (JSONL)")
    ap.add_argument("--output", required=True, help="Output path (JSON array)")
    ap.add_argument(
        "--base-url",
        default="http://localhost:8002/v1",
        help="vLLM OpenAI base url, e.g. http://localhost:8002/v1",
    )
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct", help="served-model-name in vLLM")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all samples; otherwise process first N samples")
    ap.add_argument("--max-tokens", type=int, default=96, help="max new tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 for greedy-like behavior")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--workers", type=int, default=8, help="number of worker threads")
    args = ap.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer")

    examples: List[Tuple[int, Dict[str, Any]]] = []
    for idx, ex in enumerate(iter_jsonl(args.input), start=1):
        examples.append((idx, ex))
        if args.max_samples and len(examples) >= args.max_samples:
            break

    results: List[Optional[Dict[str, Any]]] = [None] * len(examples)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_one, pos, idx, ex, args)
            for pos, (idx, ex) in enumerate(examples)
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="sample"):
            pos, idx, result, error = fut.result()
            results[pos] = result
            if error:
                tqdm.write(f"[{idx}] ERROR: {error}")

    final_results = [r for r in results if r is not None]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(final_results)} records to {args.output}")


if __name__ == "__main__":
    main()
