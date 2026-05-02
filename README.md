# When Policies Disagree

Reproducibility package for *When Policies Disagree: A Cross-Policy Audit of
GDPR Transparency on the Web* (CCS '26).

```
.
├── data/
│   └── dataset.tar.gz          # bundled raw data — extracted on first notebook run
├── notebooks/
│   ├── RQ1.ipynb               # availability, length, readability  (Findings.tex §RQ1)
│   ├── RQ2.ipynb               # GDPR completeness                  (Findings.tex §RQ2)
│   ├── RQ3.ipynb               # cross-policy inconsistency         (Findings.tex §RQ3)
│   └── Evaluation.ipynb        # extractor / verifier evaluation    (Evaluation.tex + Appendix.tex)
├── code/
│   ├── prompts/                # unified prompt registry (single source of truth)
│   ├── pipeline/               # vendored extractor + verifier
│   ├── llm_serving/            # 2× A100 vLLM / Ollama setup
│   ├── scripts/                # CLIs: run_extraction, run_verification, train_roberta
│   ├── requirements.txt
│   ├── requirements-server.txt
│   └── .env.example
└── scraper/                    # methodology reference for the FP/TP crawler (see scraper/README.md)
```

## Quickstart — verify the paper numbers without any GPU

```bash
git clone <repo>
cd When-Policies-Disagree

python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

jupyter nbconvert --to notebook --execute --inplace notebooks/RQ1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/RQ2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/RQ3.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Evaluation.ipynb
```

Each notebook starts by extracting `data/dataset.tar.gz` (a 103 MB tarball
containing the crawl results, per-policy GDPR coverage, and the cross-policy
findings) into `data/raw/` and then computes everything from those raw files.
Each notebook ends with a sanity-check cell that prints every reproduced
number side-by-side with the value reported in the paper and reports
`PASSED` if every check matches.

## Reproducing the inconsistency-detection pipeline end-to-end

You will need a GPU. We ran on **2× NVIDIA A100 80 GB**.

```bash
# 1) Start a local LLM
bash code/llm_serving/serve_vllm.sh        # default: gemma3:27b across both A100s

# 2) Point the pipeline at it
cp code/.env.example code/.env             # the defaults already target localhost
source code/llm_serving/env_local.sh

# 3) Run the verifier on a list of (FP, TP) pairs
python code/scripts/run_verification.py \
    --pairs       data/sample_pairs.csv \
    --extractions data/extractions/ \
    --out         findings_local.csv
```

`code/llm_serving/README.md` documents the hardware and the alternative
hosted-API path.

## Hardware

The full run reported in the paper used **2× NVIDIA A100 80 GB** (NVLink),
256 GB host RAM, CUDA 12.4, vLLM 0.6.x. Models that exceed that envelope
(`qwen3-vl:235b`, `gpt-oss:120b`, `deepseek-v3.1:671b`) were served on
rented GPU instances from **Vast.ai** behind an OpenAI-compatible endpoint —
the same env vars switch between local and rented serving with no code
changes.

## Dataset

See [`data/README.md`](data/README.md) for the contents of `dataset.tar.gz`.

## Issues / questions

If a sanity-check cell prints something that does not match what the paper
says, that is the most useful kind of bug report — please attach the cell
output when filing an issue.
