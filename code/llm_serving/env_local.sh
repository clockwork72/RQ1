# Source this file (`source llm_serving/env_local.sh`) to point every script
# in this repo at the locally-served LLM started by serve_vllm.sh or
# serve_ollama.sh.
#
# All code paths that hit an LLM read these three environment variables from
# pipeline/config.py — no code changes are required to swap between a
# locally-served 2× A100 server and a rented Vast.ai 4× A100 endpoint.

export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-local}"
export LLM_MODEL="${LLM_MODEL:-gemma3:27b}"

# Optional: pin the model used by the perturbation / agreement evaluations to
# the same locally-served model.
export EVAL_VERIFIER_MODEL="${EVAL_VERIFIER_MODEL:-$LLM_MODEL}"

echo "[env_local] LLM endpoint  : $LLM_BASE_URL"
echo "[env_local] LLM model     : $LLM_MODEL"
