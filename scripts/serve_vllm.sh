#!/usr/bin/env bash
# Step 1: serve a TRACE model with vLLM. Then, in another shell, step 2:
#   python -m trace_defense.server --backend http://localhost:30003/v1 --model "$MODEL"
set -euo pipefail
MODEL="${MODEL:-Dipto084/Llama3.1-8B-TRACE}"
PORT="${PORT:-30003}"
vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --gpu-memory-utilization "${GPU_MEM:-0.85}"
