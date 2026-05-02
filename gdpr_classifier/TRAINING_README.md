# GDPR classifier

Trainer + dataset + per-model evaluation outputs for the 18-category GDPR
transparency classifier described in `Appendix.tex`. The pipeline picks
RoBERTa-base as the production classifier (best on every overall metric);
BERT-base-uncased and Legal-BERT are kept for the comparison.

## Layout

```
gdpr_classifier/
├── train.py                                # single-script trainer (HF Trainer + focal BCE + per-class threshold tuning)
├── data/
│   └── gdpr_dataset_1.2.csv                # Rahat et al. (WPES'22) multi-label segments, 10,510 rows × 18 categories
└── results/
    ├── gdpr_roberta_results.json           # per-model overall + per-label P/R/F1
    ├── gdpr_bert_base_results.json
    └── gdpr_legalbert_results.json
```

Each `results/gdpr_*.json` was emitted by a full `train.py` run on the same
80/10/10 stratified split.

## Reproducing the numbers in the paper

Without re-training, just run the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Classifier.ipynb
```

It loads the three result JSONs, prints the overall-metrics table
(EM, μP, μR, μF1, MF1), regenerates `notebooks/figures/fig1_overall_comparison.pdf`,
and ends with a sanity-check cell that compares every reproduced number to
the value reported in `Appendix.tex`.

## Re-training from scratch

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets scikit-learn pandas

python gdpr_classifier/train.py \
    --csv         gdpr_classifier/data/gdpr_dataset_1.2.csv \
    --model       roberta-base \
    --epochs      6 \
    --lr          2e-5 \
    --out-dir     out/roberta
```

Switch `--model` to `bert-base-uncased` or `nlpaueb/legal-bert-base-uncased`
to reproduce the comparison rows. Each run writes a fresh
`gdpr_<model>_results.json` to `--out-dir`.

A GPU is recommended; the run takes about an hour on one A100. The trainer
does early stopping on macro F1 over the validation split and tunes one
threshold per class on validation before reporting test metrics.

## Trained weights

The 476 MB fine-tuned RoBERTa weights (`model.safetensors` + tokenizer
config) are not bundled with this repo. They are published on the
GitHub release page as `gdpr_roberta.tar.gz` so reviewers can download
once and skip training. Extract that tarball into `gdpr_classifier/model/`
to use the model directly via
`AutoModelForSequenceClassification.from_pretrained`.
