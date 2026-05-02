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

The paper's inconsistencies and evaluations were produced on two tiers
of hardware:

**Production verifier (gemma3:27b) — local 2× NVIDIA A100 80 GB.**
A 27B-class model fits in tensor-parallel mode across the two A100s with
`--max-model-len 8192` and `--gpu-memory-utilization 0.90`. Node spec:
2× A100 80 GB (NVLink), 256 GB host RAM, CUDA 12.4, vLLM 0.6.x.

**Heavier models — rented 4× NVIDIA A100 80 GB on Vast.ai.**
Larger models in our registry (`qwen3-vl:235b`, `gpt-oss:120b`,
`deepseek-v3.1:671b`) do not fit in 2× A100, so we
**rented a 4× A100 80 GB instance on Vast.ai** for those runs. The
rented box served the same vLLM / Ollama OpenAI-compatible `/v1`
endpoint; the pipeline was pointed at it by changing only `LLM_BASE_URL`.

## End-to-end recipe (local 2× A100)

```bash
# Terminal 1 — start the server (blocks)
bash llm_serving/serve_vllm.sh

# Terminal 2 — point the rest of the repo at the local server
source llm_serving/env_local.sh

# Run anything that needs an LLM
python code/scripts/run_extraction.py \
    --policy data/sample_policies/example_site.txt --out out/
python code/scripts/run_verification.py \
    --pairs data/sample_pairs.csv --out out/
```

## Models referenced in the paper

The paper benchmarks several LLMs as extractors and verifiers. Each name
below is the tag used in the bundled result JSONs and in the leaderboard /
perturbation tables. To reproduce a row, serve the corresponding model
through your `/v1` endpoint and re-run `code/scripts/run_evaluation.py` with
that model.

| Tag in the paper | Where to obtain weights | How we served it |
|---|---|---|
| `gemma3:27b`         | `ollama pull gemma3:27b` (Google, public)               | Local 2× A100 |
| `gemma4:31b`         | `ollama pull gemma4:31b` (Google, public)               | Rented Vast.ai 4× A100 |
| `qwen3-next:80b`     | `ollama pull qwen3-next:80b` (Alibaba, public)          | Rented Vast.ai 4× A100 |
| `qwen3-vl:235b`      | `ollama pull qwen3-vl:235b` (Alibaba, public)           | Rented Vast.ai 4× A100 |
| `gpt-oss:120b`       | `ollama pull gpt-oss:120b` (OpenAI weights release)     | Rented Vast.ai 4× A100 |
| `deepseek-v3.1:671b` | `ollama pull deepseek-v3.1:671b` (DeepSeek, public)     | Rented Vast.ai 4× A100 |
| `deepseek-v4-flash`  | DeepSeek hosted API, model name `deepseek-v4-flash`     | Hosted endpoint |

For exact byte-level reproducibility, pin the Ollama Modelfile digest
shown by `ollama show <tag> --modelfile` after pulling. LLM outputs are
not byte-deterministic across vLLM / Ollama versions even at
`temperature=0`, so reviewers should expect ≤1% drift on the binary
metrics in `eval_perturbation_v3.json` even when serving the same tag.

