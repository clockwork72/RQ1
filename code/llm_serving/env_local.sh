# Source this file (`source llm_serving/env_local.sh`) to point every script
# in this repo at the locally-served LLM started by serve_vllm.sh or
# serve_ollama.sh.
#
# All code paths that hit an LLM read these two environment variables from
# pipeline/config.py — no code changes are required to swap between a hosted
# API and a local server.

export OLLAMA_PRO_BASE_URL="${OLLAMA_PRO_BASE_URL:-http://localhost:8000/v1}"
export OLLAMA_PRO_API_KEY="${OLLAMA_PRO_API_KEY:-local}"
export OLLAMA_PRO_MODEL="${OLLAMA_PRO_MODEL:-gemma3:27b}"

# Optional: pin the model used by the perturbation / agreement evaluations to
# the same locally-served model.
export EVAL_VERIFIER_MODEL="${EVAL_VERIFIER_MODEL:-$OLLAMA_PRO_MODEL}"

echo "[env_local] LLM endpoint  : $OLLAMA_PRO_BASE_URL"
echo "[env_local] LLM model     : $OLLAMA_PRO_MODEL"
