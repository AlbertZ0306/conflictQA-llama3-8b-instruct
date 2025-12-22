ConflictQA 流程说明

本目录包含 ConflictQA 的脚本、数据、结果与可视化图，基于 vLLM 的
OpenAI 兼容接口运行。

目录结构
- dataset/
  - conflictQA-strategyQA-chatgpt.aligned.jsonl
- scripts/
  - run_c0_vllm.py, run_c0_vllm_parallel.py
  - run_c3_vllm.py, run_c3_vllm_parallel.py
  - run_c4_vllm.py, run_c4_vllm_parallel.py
  - run_c5_vllm.py, run_c5_vllm_parallel.py
  - plot_c0_metrics.py
  - plot_c0_metrics_curve.py
  - reorder_evidence_by_gt.py
- results/
  - c0.json, c3.json, c4.json, c5_cp.json, c5_pc.json
- plots/
  - 直方图与密度图（png）

数据格式（JSONL）
每行一个 JSON，至少包含以下字段：
- question
- ground_truth（例如 ["True"] 或 ["False"]）
- parametric_memory_aligned_evidence（C3/C5 使用）
- counter_memory_aligned_evidence（C4/C5 使用）

结果格式（JSON）
每个输出文件是一个 JSON 数组，包含：
- question
- answer
- correct
- token_number_of_answer
- mean, std, range（由生成 token 的概率计算）

前置条件
- 已启动 vLLM OpenAI 兼容服务，默认 base URL：
  http://localhost:8002/v1
- 服务端模型名与 --model 一致（默认 Llama-3.1-8B-Instruct）

运行推理（示例）
C0（仅问题）：
python conflictQA/scripts/run_c0_vllm_parallel.py \
  --input conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.jsonl \
  --output conflictQA/results/c0.json \
  --base-url http://localhost:8002/v1 \
  --model Llama-3.1-8B-Instruct \
  --max-samples 0 \
  --max-tokens 96 \
  --temperature 0.0 \
  --timeout 120 \
  --workers 8

C3（问题 + parametric evidence）：
python conflictQA/scripts/run_c3_vllm_parallel.py \
  --input conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.jsonl \
  --output conflictQA/results/c3.json \
  --base-url http://localhost:8002/v1 \
  --model Llama-3.1-8B-Instruct \
  --max-samples 0 \
  --max-tokens 96 \
  --temperature 0.0 \
  --timeout 120 \
  --workers 8

C4（问题 + counter evidence）：
python conflictQA/scripts/run_c4_vllm_parallel.py \
  --input conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.jsonl \
  --output conflictQA/results/c4.json \
  --base-url http://localhost:8002/v1 \
  --model Llama-3.1-8B-Instruct \
  --max-samples 0 \
  --max-tokens 96 \
  --temperature 0.0 \
  --timeout 120 \
  --workers 8

C5（问题 + 两段 evidence，顺序 pc/cp）：
python conflictQA/scripts/run_c5_vllm_parallel.py \
  --input conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.jsonl \
  --output conflictQA/results/c5_pc.json \
  --base-url http://localhost:8002/v1 \
  --model Llama-3.1-8B-Instruct \
  --order pc \
  --max-samples 0 \
  --max-tokens 96 \
  --temperature 0.0 \
  --timeout 120 \
  --workers 8

绘图
按 bins 画准确率直方图：
python conflictQA/scripts/plot_c0_metrics.py \
  --input conflictQA/results/c0.json \
  --output-dir conflictQA/plots/c0 \
  --mean-bins 0.5,0.6,0.7,0.8,0.9,1.0 \
  --std-bins 0,0.1,0.2,0.3,0.4,0.5,0.6 \
  --range-bins 0,0.2,0.4,0.6,0.8,1.0

密度图（KDE）：
python conflictQA/scripts/plot_c0_metrics_curve.py \
  --input conflictQA/results/c0.json \
  --output-dir conflictQA/plots/c0_density \
  --mean-bins 0.5,0.6,0.7,0.8,0.9,1.0 \
  --std-bins 0,0.1,0.2,0.3,0.4,0.5,0.6 \
  --range-bins 0,0.2,0.4,0.6,0.8,1.0

证据重排/重写（可选）
检查 evidence 与 ground_truth 的一致性，必要时交换 A/B，
并可重新生成 evidence 以强制对齐。

python conflictQA/scripts/reorder_evidence_by_gt.py \
  --input conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.jsonl \
  --output conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.fixed.jsonl \
  --review conflictQA/dataset/conflictQA-strategyQA-chatgpt.aligned.review.jsonl \
  --base-url http://localhost:8002/v1 \
  --model Llama-3.1-8B-Instruct \
  --num-workers 1
