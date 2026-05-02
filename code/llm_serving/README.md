# LLM serving

Every LLM call in this repo speaks the OpenAI chat-completions protocol. You
have two ways to satisfy that contract; the rest of the pipeline does not
care which one you use.

| | What it is | When to use |
|---|---|---|
| `serve_vllm.sh`   | Launches `vllm.entrypoints.openai.api_server` with tensor-parallel size 2. | Local 2× A100 box. Lowest latency for one model at a time. |
| `serve_ollama.sh` | Starts the Ollama daemon (which has a built-in OpenAI-compatible `/v1`). | Local 2× A100 box. Easier model swapping; multiple models loaded at once. |

Both expose the same endpoint at `http://localhost:8000/v1`. The default
served model is **`gemma3:27b`**, the verifier we used for the production
findings reported in §3 of the paper.

## Hardware we used

Inconsistencies and evaluations in the paper were produced on a node with:

* 2× NVIDIA A100 80 GB (NVLink)
* 256 GB host RAM
* CUDA 12.4, vLLM 0.6.x

A 27B-class model (gemma3:27b) fits in tensor-parallel mode across the two
A100s with `--max-model-len 8192` and `--gpu-memory-utilization 0.90`. Larger
models in our model registry (`gemma4:31b`, `qwen3-vl:235b`, `gpt-oss:120b`,
`deepseek-v3.1:671b`) do not fit in 2× A100; we ran those on rented Vast.ai
instances with **4× A100 80 GB**, behind the same OpenAI-compatible /v1
endpoint, and pointed the pipeline at them by changing only `LLM_BASE_URL`.

## End-to-end recipe (local 2× A100)

```bash
# Terminal 1 — start the server (blocks)
bash llm_serving/serve_vllm.sh

# Terminal 2 — point the rest of the repo at the local server
source llm_serving/env_local.sh

# Run anything that needs an LLM
python scripts/run_extraction.py --policy data/sample_policy.txt --out out/
python scripts/run_verification.py --extractions out/ --pairs data/sample_pairs.csv --out findings.csv
```

## Rented-GPU alternative (Vast.ai 4× A100)

If you do not have a 2× A100 box, rent a 4× A100 instance on Vast.ai (or
any other provider), launch the same vLLM / Ollama server on it, and point
the repo at it by exporting:

```bash
export LLM_BASE_URL="http://<your-vast-host>:<port>/v1"
export LLM_API_KEY="<your key>"
export LLM_MODEL="gemma3:27b"      # or any other model your instance serves
```

The pipeline does not care whether the endpoint is local or rented — only
that it accepts OpenAI-style chat-completions.
