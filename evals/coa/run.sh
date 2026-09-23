#!/usr/bin/env bash
# Chain-of-Attack vs. TRACE on the paper's 120 behaviors (80 HarmBench + 40 JailbreakBench).
# Attacker: Qwen3-32B (3 streams, up to 5 rounds x 6 iterations). Judge: GPT-4o, CoA 1-10 scale.
#
# Needs four services (start them first, in this order):
#   1. attacker vLLM     vllm serve Qwen/Qwen3-32B --port 30000 --max-model-len 16384
#   2. target vLLM       MODEL=Dipto084/Llama3.1-8B-TRACE PORT=30001 ../../scripts/serve_vllm.sh
#   3. SimCSE service    python local_simcse_api.py            (port 8001, CPU)
#   4. toxicity service  python local_toxigen_api.py           (port 8002; uses OPENAI_API_KEY)
# and OPENAI_API_KEY in the environment for the GPT-4o judge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export VLLM_ATTACKER_PORT="${VLLM_ATTACKER_PORT:-30000}"
export VLLM_TARGET_PORT="${VLLM_TARGET_PORT:-30001}"
export SIMCSE_PORT="${SIMCSE_PORT:-8001}"
export TOXIGEN_PORT="${TOXIGEN_PORT:-8002}"
export TRACE_MODEL="${TRACE_MODEL:-Dipto084/Llama3.1-8B-TRACE}"   # must match what the target vLLM serves
: "${OPENAI_API_KEY:?set OPENAI_API_KEY for the GPT-4o judge}"

python -u experiment.py --baseline coa-test120-trace
python build_master_log.py          # flattens logs/<run>/*/logs.json into one master_log.json
