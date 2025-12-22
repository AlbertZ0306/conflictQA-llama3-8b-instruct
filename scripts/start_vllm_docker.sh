#!/usr/bin/env bash
set -euo pipefail

docker run -it --rm \
  --gpus all \
  -p 8002:8002 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --name my_vllm \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.10.1.1
