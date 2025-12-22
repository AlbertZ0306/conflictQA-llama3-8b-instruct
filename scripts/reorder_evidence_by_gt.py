#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import requests


# ---------------------------
# helpers
# ---------------------------

def as_bool_gt(ground_truth) -> Optional[bool]:
    """
    StrategyQA ground_truth 通常是 ["True"] / ["False"]。
    返回 True/False；无法识别则 None。
    """
    if isinstance(ground_truth, list) and ground_truth:
        gt = str(ground_truth[0]).strip().lower()
    else:
        gt = str(ground_truth).strip().lower()

    if gt in ["true", "yes", "1"]:
        return True
    if gt in ["false", "no", "0"]:
        return False
    return None


def label_str(b: Optional[bool]) -> str:
    if b is None:
        return "Unknown"
    return "True" if b else "False"


def opposite_label(label: str) -> str:
    if label == "True":
        return "False"
    if label == "False":
        return "True"
    return "Uncertain"


def robust_json_loads(s: str) -> Optional[dict]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def call_vllm_chat(
    base_url: str,
    model: str,
    messages: list,
    timeout: int = 90,
    max_tokens: int = 256,
    temperature: float = 0.0,
    retries: int = 3,
    backoff: float = 1.6,
    response_format_json: bool = False,
) -> str:
    """
    调用 vLLM OpenAI-compatible /v1/chat/completions
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # vLLM 有的版本支持 response_format，若不支持会报错；默认关掉
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise RuntimeError(f"vLLM call failed after {retries} attempts: {last_err}") from last_err
    raise RuntimeError("unreachable")


# ---------------------------
# prompts
# ---------------------------

def build_judge_two_prompt(question: str, gt_label: str, ev_a: str, ev_b: str) -> list:
    """
    一次请求同时判别 Evidence A/B 的立场，并给出哪个更贴近 ground truth。
    """
    system = (
        "You are a precise NLI judge.\n"
        "Given a yes/no question and an evidence snippet, decide which answer the evidence supports.\n"
        "Output ONLY valid JSON."
    )

    user = f"""Question:
{question}

Ground truth label for alignment decision: {gt_label}
Rule:
- If evidence supports YES to the question -> stance is True.
- If evidence supports NO to the question -> stance is False.
- If evidence is irrelevant/ambiguous -> Uncertain.

Evidence A:
{ev_a}

Evidence B:
{ev_b}

Return STRICT JSON:
{{
  "stance_A": "True|False|Uncertain",
  "stance_B": "True|False|Uncertain",
  "aligned_to_ground_truth": "A|B|None",
  "confidence_A": 0.0,
  "confidence_B": 0.0,
  "notes": "short"
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_single_prompt(question: str, ev: str) -> list:
    """
    判别单段 evidence 立场（用于重写后验证）。
    """
    system = (
        "You are a precise NLI judge.\n"
        "Given a yes/no question and an evidence snippet, decide which answer the evidence supports.\n"
        "Output ONLY valid JSON."
    )

    user = f"""Question:
{question}

Evidence:
{ev}

Return STRICT JSON:
{{
  "stance": "True|False|Uncertain",
  "confidence": 0.0,
  "notes": "short"
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_generate_evidence_prompt(question: str, target_label: str) -> list:
    """
    生成与 target_label 一致的 evidence（2-5 句，尽量像“可检索到的陈述/背景信息”）。
    """
    system = (
        "You write concise, factual-sounding evidence snippets for a given yes/no question.\n"
        "Do not include disclaimers, uncertainty hedges, or meta commentary.\n"
        "Do not mention 'ground truth' or 'label'."
    )
    user = f"""Question:
{question}

Write an evidence snippet that strongly supports the answer being {target_label}.
Requirements:
- 2 to 5 sentences.
- Be directly relevant to the question.
- Do NOT include citations, URLs, or markdown.
- Do NOT mention being an AI or that you are generating text.
Return only the evidence text.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------
# core logic
# ---------------------------

def pick_parametric_index(
    gt_label: str,
    stance_a: str,
    stance_b: str,
    aligned: str,
    conf_a: float,
    conf_b: float
) -> Optional[int]:
    """
    选择哪个 evidence 放到 parametric_memory_aligned_evidence（目标：支持 ground_truth）。
    返回 0 表示选 A，1 表示选 B，None 表示无法判定。
    """
    # 1) 优先：哪一个直接等于 gt_label
    a_match = (stance_a == gt_label)
    b_match = (stance_b == gt_label)

    if a_match and not b_match:
        return 0
    if b_match and not a_match:
        return 1

    # 2) 使用模型给的 aligned_to_ground_truth
    if aligned == "A":
        return 0
    if aligned == "B":
        return 1

    # 3) 再用置信度 + 非 Uncertain 优先
    def score(stance: str, conf: float) -> float:
        base = 0.0
        if stance == gt_label:
            base += 2.0
        if stance == "Uncertain":
            base -= 0.5
        return base + float(conf)

    sa = score(stance_a, conf_a)
    sb = score(stance_b, conf_b)
    if sa > sb:
        return 0
    if sb > sa:
        return 1

    return None


def judge_two(
    base_url: str,
    model: str,
    question: str,
    gt_label: str,
    ev_a: str,
    ev_b: str,
    response_format_json: bool
) -> Tuple[Optional[dict], Optional[str]]:
    messages = build_judge_two_prompt(question, gt_label, ev_a, ev_b)
    out = call_vllm_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        timeout=90,
        retries=3,
        response_format_json=response_format_json,
    )
    j = robust_json_loads(out)
    if j is None:
        return None, out
    return j, None


def judge_single(
    base_url: str,
    model: str,
    question: str,
    ev: str,
    response_format_json: bool
) -> Tuple[Optional[dict], Optional[str]]:
    messages = build_judge_single_prompt(question, ev)
    out = call_vllm_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=128,
        timeout=90,
        retries=3,
        response_format_json=response_format_json,
    )
    j = robust_json_loads(out)
    if j is None:
        return None, out
    return j, None


def generate_evidence(
    base_url: str,
    model: str,
    question: str,
    target_label: str
) -> str:
    messages = build_generate_evidence_prompt(question, target_label)
    out = call_vllm_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.2,     # 轻微开放生成但仍可控
        max_tokens=220,
        timeout=90,
        retries=3,
        response_format_json=False,
    )
    return out.strip()


def ensure_alignment(
    base_url: str,
    model: str,
    question: str,
    gt_label: str,
    param_ev: str,
    counter_ev: str,
    response_format_json: bool,
    regen_param_if_needed: bool,
    regen_counter_if_needed: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    复核并必要时重写生成：
    - param_ev 需要支持 gt_label
    - counter_ev 需要支持 opposite(gt_label)
    """
    dbg: Dict[str, Any] = {"actions": []}
    opp = opposite_label(gt_label)

    # check param
    j1, raw1 = judge_single(base_url, model, question, param_ev, response_format_json)
    if j1 is None:
        dbg["actions"].append({"type": "judge_param_parse_fail", "raw": (raw1 or "")[:800]})
    else:
        dbg["param_stance"] = j1.get("stance")
        dbg["param_confidence"] = j1.get("confidence")
        if j1.get("stance") != gt_label and regen_param_if_needed:
            new_param = generate_evidence(base_url, model, question, gt_label)
            dbg["actions"].append({"type": "regen_param", "from": j1.get("stance"), "to": gt_label})
            param_ev = new_param
            # re-check once
            j1b, _ = judge_single(base_url, model, question, param_ev, response_format_json)
            dbg["param_stance_after"] = (j1b or {}).get("stance", "Unknown")

    # check counter
    j2, raw2 = judge_single(base_url, model, question, counter_ev, response_format_json)
    if j2 is None:
        dbg["actions"].append({"type": "judge_counter_parse_fail", "raw": (raw2 or "")[:800]})
    else:
        dbg["counter_stance"] = j2.get("stance")
        dbg["counter_confidence"] = j2.get("confidence")
        if j2.get("stance") != opp and regen_counter_if_needed:
            new_counter = generate_evidence(base_url, model, question, opp)
            dbg["actions"].append({"type": "regen_counter", "from": j2.get("stance"), "to": opp})
            counter_ev = new_counter
            # re-check once
            j2b, _ = judge_single(base_url, model, question, counter_ev, response_format_json)
            dbg["counter_stance_after"] = (j2b or {}).get("stance", "Unknown")

    return param_ev, counter_ev, dbg


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file")
    ap.add_argument("--output", required=True, help="Output JSONL file (aligned)")
    ap.add_argument("--review", required=True, help="Review JSONL for problematic items")
    ap.add_argument("--base-url", default="http://localhost:8002/v1", help="vLLM OpenAI base url, e.g. http://localhost:8002/v1")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct", help="served-model-name")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between requests")
    ap.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker threads (default: 1)")

    ap.add_argument("--max-items", type=int, default=0, help="process only first N items (0 means all)")
    ap.add_argument("--keep-orig", action="store_true", help="keep original evidences in *_orig keys")
    ap.add_argument("--response-format-json", action="store_true", help="try response_format={'type':'json_object'} for judge calls (may not be supported)")

    ap.add_argument("--regen-param-if-needed", action="store_true", help="if param evidence does not support ground_truth, regenerate it")
    ap.add_argument("--regen-counter-if-needed", action="store_true", help="if counter evidence is not opposite of ground_truth, regenerate it")
    ap.add_argument("--strict", action="store_true", help="if after regeneration still not aligned, always write to review")

    args = ap.parse_args()

    # 默认：只重写 counter（更贴合你描述的“语义相反”要求），param 只有在完全不支持 GT 时才需要动
    if not args.regen_param_if_needed and not args.regen_counter_if_needed:
        args.regen_param_if_needed = True
        args.regen_counter_if_needed = True

    n_total = 0
    n_swapped = 0
    n_regen_param = 0
    n_regen_counter = 0
    n_review = 0
    n_gt_unknown = 0
    n_parse_fail = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         open(args.review, "w", encoding="utf-8") as frev:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_total += 1
            if args.max_items and n_total > args.max_items:
                break

            obj = json.loads(line)

            question = obj.get("question", "")
            ground_truth = obj.get("ground_truth", ["Unknown"])
            gt_bool = as_bool_gt(ground_truth)
            gt_label = label_str(gt_bool)

            ev_a = obj.get("parametric_memory_aligned_evidence", "")
            ev_b = obj.get("counter_memory_aligned_evidence", "")

            if args.keep_orig:
                obj["parametric_memory_aligned_evidence_orig"] = ev_a
                obj["counter_memory_aligned_evidence_orig"] = ev_b

            if gt_bool is None:
                n_gt_unknown += 1
                # 无法定义“相反”，直接原样写出并进入 review
                frev.write(json.dumps({
                    "idx": n_total,
                    "reason": "ground_truth_not_boolean",
                    "question": question,
                    "ground_truth": ground_truth
                }, ensure_ascii=False) + "\n")
                n_review += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            # 1) 语义判别 A/B
            judge, raw = judge_two(
                base_url=args.base_url,
                model=args.model,
                question=question,
                gt_label=gt_label,
                ev_a=ev_a,
                ev_b=ev_b,
                response_format_json=args.response_format_json
            )

            if judge is None:
                n_parse_fail += 1
                frev.write(json.dumps({
                    "idx": n_total,
                    "reason": "judge_two_parse_fail",
                    "question": question,
                    "ground_truth": ground_truth,
                    "llm_output": (raw or "")[:2000]
                }, ensure_ascii=False) + "\n")
                n_review += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            stance_a = str(judge.get("stance_A", "Uncertain"))
            stance_b = str(judge.get("stance_B", "Uncertain"))
            aligned = str(judge.get("aligned_to_ground_truth", "None"))
            try:
                conf_a = float(judge.get("confidence_A", 0.0))
            except Exception:
                conf_a = 0.0
            try:
                conf_b = float(judge.get("confidence_B", 0.0))
            except Exception:
                conf_b = 0.0

            # 2) 选出 parametric evidence（支持 gt_label）
            pick = pick_parametric_index(gt_label, stance_a, stance_b, aligned, conf_a, conf_b)

            # 默认：保守策略，若无法判定就先不交换，但后续会通过 ensure_alignment 强制生成正确立场
            if pick is None:
                param_ev, counter_ev = ev_a, ev_b
                swapped = False
            elif pick == 0:
                param_ev, counter_ev = ev_a, ev_b
                swapped = False
            else:
                param_ev, counter_ev = ev_b, ev_a
                swapped = True

            if swapped:
                n_swapped += 1

            # 3) 必要时重写/生成，确保：
            #    param supports gt_label
            #    counter supports opposite(gt_label)
            param_ev2, counter_ev2, dbg = ensure_alignment(
                base_url=args.base_url,
                model=args.model,
                question=question,
                gt_label=gt_label,
                param_ev=param_ev,
                counter_ev=counter_ev,
                response_format_json=args.response_format_json,
                regen_param_if_needed=args.regen_param_if_needed,
                regen_counter_if_needed=args.regen_counter_if_needed,
            )

            for act in dbg.get("actions", []):
                if act.get("type") == "regen_param":
                    n_regen_param += 1
                if act.get("type") == "regen_counter":
                    n_regen_counter += 1

            obj["parametric_memory_aligned_evidence"] = param_ev2
            obj["counter_memory_aligned_evidence"] = counter_ev2

            # 4) 严格模式：不满足则入 review
            #   （这里再做一次最终判别，确保输出符合你定义）
            final_ok = True
            opp = opposite_label(gt_label)

            j_param, rawp = judge_single(args.base_url, args.model, question, obj["parametric_memory_aligned_evidence"], args.response_format_json)
            j_count, rawc = judge_single(args.base_url, args.model, question, obj["counter_memory_aligned_evidence"], args.response_format_json)

            if j_param is None or j_count is None:
                final_ok = False
            else:
                if str(j_param.get("stance", "Uncertain")) != gt_label:
                    final_ok = False
                if str(j_count.get("stance", "Uncertain")) != opp:
                    final_ok = False

            if args.strict and not final_ok:
                frev.write(json.dumps({
                    "idx": n_total,
                    "reason": "final_alignment_failed",
                    "question": question,
                    "ground_truth": ground_truth,
                    "expected_param": gt_label,
                    "expected_counter": opp,
                    "judge_param": j_param,
                    "judge_counter": j_count,
                    "judge_param_raw": (rawp or "")[:1200],
                    "judge_counter_raw": (rawc or "")[:1200],
                    "actions": dbg.get("actions", []),
                }, ensure_ascii=False) + "\n")
                n_review += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print("Done.")
    print(f"total={n_total}")
    print(f"swapped={n_swapped}")
    print(f"regen_param={n_regen_param}")
    print(f"regen_counter={n_regen_counter}")
    print(f"review={n_review}")
    print(f"gt_unknown={n_gt_unknown}")
    print(f"parse_fail={n_parse_fail}")
    print(f"Output: {args.output}")
    print(f"Review: {args.review}")


if __name__ == "__main__":
    main()
