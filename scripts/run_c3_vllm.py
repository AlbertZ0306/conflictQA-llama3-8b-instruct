#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

import requests

TRUE_FALSE_RE = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def parse_pred_label(text: str) -> Optional[str]:
    """Parse first True/False occurrence from the model output."""
    m = TRUE_FALSE_RE.search(text)
    if not m:
        return None
    return m.group(1).capitalize()


def extract_token_logprobs_from_chat(resp_json: Dict[str, Any]) -> List[float]:
    """
    Try to extract chosen-token logprobs from vLLM OpenAI-compatible response.
    Common schema:
      choices[0].logprobs.content = [{token, logprob, ...}, ...]
    """
    choice = resp_json["choices"][0]
    lp = choice.get("logprobs", None)
    if lp is None:
        return []

    if isinstance(lp, dict):
        if "content" in lp and isinstance(lp["content"], list):
            out = []
            for item in lp["content"]:
                if isinstance(item, dict) and "logprob" in item:
                    out.append(float(item["logprob"]))
            return out

        if "token_logprobs" in lp and isinstance(lp["token_logprobs"], list):
            return [float(x) for x in lp["token_logprobs"]]

    if isinstance(lp, list):
        out = []
        for item in lp:
            if isinstance(item, dict) and "logprob" in item:
                out.append(float(item["logprob"]))
        return out

    return []


def logprobs_to_probs(logprobs: List[float]) -> List[float]:
    probs = []
    for lp in logprobs:
        if lp is None:
            continue
        if lp == float("-inf"):
            probs.append(0.0)
        else:
            try:
                probs.append(math.exp(lp))
            except OverflowError:
                probs.append(0.0)
    return probs


def stats_from_probs(probs: List[float]) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    n = len(probs)
    if n == 0:
        return 0, None, None, None

    mean = sum(probs) / n
    std = statistics.stdev(probs) if n >= 2 else 0.0  # sample std
    rng = max(probs) - min(probs)
    return n, mean, std, rng


def call_vllm_chat_c3(
    base_url: str,
    model: str,
    question: str,
    evidence: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"

    system_prompt = (
        "You are a precise assistant. Output exactly ONE line.\n"
        "Format: `Answer: True/False. Because <one sentence, 25-35 words>.`\n"
        "Use the provided context if it is relevant.\n"
        "Do not add any extra text."
    )

    # C3: question + parametric_memory_aligned_evidence
    user_prompt = (
        f"Question: {question}\n"
        f"Context:\n<<<\n{evidence}\n>>>\n"
        "Respond in the required format."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 0,
    }

    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON at line {line_no}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to conflictQA-strategyQA-chatgpt.json (JSONL)")
    ap.add_argument("--output", required=True, help="Output path (JSON array)")
    ap.add_argument("--base-url", default="http://localhost:8002/v1", help="e.g. http://localhost:8002/v1")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct", help="served-model-name in vLLM")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all samples; otherwise first N samples")
    ap.add_argument("--max-tokens", type=int, default=96, help="max new tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic decoding")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []

    for i, ex in enumerate(iter_jsonl(args.input), start=1):
        if args.max_samples and i > args.max_samples:
            break

        question = ex.get("question", "")
        evidence = ex.get("parametric_memory_aligned_evidence", "")
        gt_list = ex.get("ground_truth", [])
        gt = gt_list[0] if isinstance(gt_list, list) and len(gt_list) > 0 else None
        if isinstance(gt, str):
            gt = gt.capitalize()

        try:
            resp = call_vllm_chat_c3(
                base_url=args.base_url,
                model=args.model,
                question=question,
                evidence=evidence,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            text = resp["choices"][0]["message"]["content"]
        except Exception as e:
            results.append({
                "question": question,
                "answer": "",
                "correct": False,
                "token_number_of_answer": 0,
                "mean": None,
                "std": None,
                "range": None,
            })
            print(f"[{i}] ERROR: {e}")
            continue

        pred = parse_pred_label(text)
        correct = (pred is not None and gt is not None and pred == gt)

        logprobs = extract_token_logprobs_from_chat(resp)
        probs = logprobs_to_probs(logprobs)
        n_tokens, mean, std, rng = stats_from_probs(probs)

        results.append({
            "question": question,
            "answer": text,
            "correct": bool(correct),
            "token_number_of_answer": n_tokens,
            "mean": mean,
            "std": std,
            "range": rng,
        })

        if i % 50 == 0:
            print(f"Processed {i} samples...")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
