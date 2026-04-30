#!/usr/bin/env bash
# Alternative to serve_vllm.sh: use Ollama, which auto-shards across the two
# A100s and ships an OpenAI-compatible /v1 endpoint by default.
#
# Usage
# -----
#   bash llm_serving/serve_ollama.sh                      # starts the daemon
#   ollama pull gemma3:27b                                # in another shell
#   source llm_serving/env_local.sh
#   python scripts/run_extraction.py --policy <path>
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:8000}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

echo "Starting Ollama daemon"
echo "  host                  : $OLLAMA_HOST"
echo "  parallel requests     : $OLLAMA_NUM_PARALLEL"
echo "  max loaded models     : $OLLAMA_MAX_LOADED_MODELS"
echo "  keep alive            : $OLLAMA_KEEP_ALIVE"
echo
echo "OpenAI-compatible endpoint will be available at:"
echo "  http://localhost:${OLLAMA_HOST##*:}/v1"
echo

exec ollama serve
