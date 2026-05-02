#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server on localhost:8000 backed by 2× A100.
#
# This reproduces the runtime we used to detect inconsistencies in the paper:
# every script in this repo speaks the OpenAI chat-completions protocol, so
# pointing LLM_BASE_URL at http://localhost:8000/v1 is enough — no code
# changes required.
#
# Default model: gemma3:27b (the verifier we used for the production findings).
# Override by exporting MODEL=<hf-id> before invoking the script. For the
# verdict-agreement cross-model experiment, run this once per model under a
# different port.
#
# Usage
# -----
#   # 1. Start the server (foreground; Ctrl-C to stop)
#   bash llm_serving/serve_vllm.sh
#
#   # 2. In a second shell, point the pipeline at it and run anything
#   source llm_serving/env_local.sh
#   python scripts/run_extraction.py --policy data/sample_policy.txt
#
# Requirements
# ------------
#   pip install "vllm>=0.6"     # bundled in requirements-server.txt
#   2x NVIDIA A100 80GB         # tensor-parallel size 2 for 27B-class models
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
PORT="${PORT:-8000}"
TP="${TP:-2}"           # tensor-parallel size; one shard per A100
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Starting vLLM"
echo "  model        : $MODEL"
echo "  port         : $PORT"
echo "  TP size      : $TP    (= number of GPUs)"
echo "  max_len      : $MAX_LEN"
echo "  GPU mem util : $GPU_UTIL"
echo
echo "OpenAI-compatible endpoint will be available at:"
echo "  http://localhost:${PORT}/v1"
echo

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name "$(basename "$MODEL")"
