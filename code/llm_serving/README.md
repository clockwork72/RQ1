# Local LLM serving (2× NVIDIA A100)

Every LLM call in this repo speaks the OpenAI chat-completions protocol. You
have two ways to satisfy that contract on a 2× A100 machine, and both expose
the same endpoint at `http://localhost:8000/v1`:

| | What it is | When to use |
|---|---|---|
| `serve_vllm.sh`   | Launches `vllm.entrypoints.openai.api_server` with tensor-parallel size 2. | Lowest latency for one model at a time. |
| `serve_ollama.sh` | Starts the Ollama daemon (which has a built-in OpenAI-compatible `/v1`). | Easier model swapping; multiple models loaded at once. |

The default served model is **`gemma3:27b`**, which is the verifier we used
for the production findings reported in §3 of the paper.

## Hardware we used

Inconsistencies and evaluations in the paper were produced on a node with:

* 2× NVIDIA A100 80 GB (NVLink)
* 256 GB host RAM
* CUDA 12.4, vLLM 0.6.x

A 27B-class model (gemma3:27b) fits in tensor-parallel mode across the two
A100s with `--max-model-len 8192` and `--gpu-memory-utilization 0.90`. Larger
models in our model registry (e.g. `gemma4:31b`, `qwen3-vl:235b`,
`gpt-oss:120b`) require more GPUs or were run via the hosted Ollama Pro API
during evaluation; see the notes inside `notebooks/reproduce_evaluation.ipynb`
for which model needs which configuration.

## End-to-end recipe

```bash
# Terminal 1 — start the server (blocks)
bash llm_serving/serve_vllm.sh

# Terminal 2 — point the rest of the repo at the local server
source llm_serving/env_local.sh

# Run anything that needs an LLM
python scripts/run_extraction.py --policy data/sample_policy.txt --out out/
python scripts/run_verification.py --extractions out/ --pairs data/sample_pairs.csv --out findings.csv
```

## Hosted-API alternative

If you do not have GPUs, you can use the same Ollama Pro endpoint we used:

```bash
export OLLAMA_PRO_BASE_URL="https://ollama.com/v1"
export OLLAMA_PRO_API_KEY="<your key>"
export OLLAMA_PRO_MODEL="gemma3:27b"
```

The pipeline does not care whether the endpoint is local or hosted — only that
it accepts OpenAI-style chat-completions.
