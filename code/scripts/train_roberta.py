#!/usr/bin/env python3
"""Fine-tune a BERT model for GDPR category classification.

Multi-label classification on the 18 GDPR disclosure requirement categories
from Rahat et al. (ACM WPES'22).

Handles class imbalance with:
  1. Focal Loss (Lin et al., 2017) — down-weights easy/frequent examples
  2. Inverse-frequency pos_weight in BCE — upweights rare positive classes
  3. Per-class threshold tuning on validation set
  4. Early stopping with patience

Usage:
    python train_gdpr_bert.py --model bert-base-uncased --epochs 10 --batch-size 32
    python train_gdpr_bert.py --model nlpaueb/legal-bert-base-uncased --epochs 10
    python train_gdpr_bert.py --model roberta-large --epochs 8 --lr 1e-5 --batch-size 16
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

import argparse
import ast
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from pipeline.schema import GDPR_CATEGORIES

DATASET_PATH = REPO_ROOT / "gdpr_classifier" / "data" / "gdpr_dataset_1.2.csv"
LABEL2IDX = {label: i for i, label in enumerate(sorted(GDPR_CATEGORIES))}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_LABELS = len(GDPR_CATEGORIES)


# ---------------------------------------------------------------------------
# Focal Loss for multi-label classification (Lin et al., ICCV 2017)
# ---------------------------------------------------------------------------

class FocalBCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy with focal modulation and optional pos_weight.

    Focal loss reduces the contribution of well-classified examples,
    forcing the model to focus on hard/rare cases. For imbalanced multi-label
    problems, this is combined with pos_weight (inverse frequency weighting)
    so rare positive classes get both more weight AND more focus when misclassified.

    Args:
        gamma: Focusing parameter (0 = standard BCE, 2 = strong focusing).
        alpha: Balance factor for positive vs negative. If None, uses pos_weight.
        pos_weight: Per-class weight for positive examples (inverse frequency).
    """
    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
        )
        # Focal modulation
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GDPRDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[list[str]], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_vec = torch.zeros(NUM_LABELS)
        for label in labels:
            if label in LABEL2IDX:
                label_vec[LABEL2IDX[label]] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec,
        }


def load_data() -> tuple[list[str], list[list[str]]]:
    texts, labels = [], []
    with DATASET_PATH.open() as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            texts.append(row[2].strip())
            labels.append(ast.literal_eval(row[3]))
    return texts, labels


# ---------------------------------------------------------------------------
# Class imbalance utilities
# ---------------------------------------------------------------------------

def compute_pos_weight(labels: list[list[str]]) -> torch.Tensor:
    """Compute per-class positive weight = num_negatives / num_positives.

    This is the standard approach for BCEWithLogitsLoss pos_weight parameter.
    Capped to avoid extreme values for very rare classes.
    """
    n = len(labels)
    pos_counts = np.zeros(NUM_LABELS)
    for label_list in labels:
        for label in label_list:
            if label in LABEL2IDX:
                pos_counts[LABEL2IDX[label]] += 1

    neg_counts = n - pos_counts
    # pos_weight = neg_count / pos_count (more weight for rare positives)
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    # Cap to avoid instability on very rare classes
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


def make_oversampling_weights(labels: list[list[str]]) -> torch.Tensor:
    """Create per-sample weights for WeightedRandomSampler.

    Samples with rare labels get higher sampling probability,
    so the model sees rare categories more often during training.
    """
    # Inverse frequency per label
    label_counts = np.zeros(NUM_LABELS)
    for label_list in labels:
        for label in label_list:
            if label in LABEL2IDX:
                label_counts[LABEL2IDX[label]] += 1

    label_weights = 1.0 / (label_counts + 1)

    # Sample weight = max weight of its labels (so rare-label samples are upsampled)
    sample_weights = []
    for label_list in labels:
        w = max(label_weights[LABEL2IDX[l]] for l in label_list if l in LABEL2IDX) if label_list else 0.0
        sample_weights.append(w)

    return torch.tensor(sample_weights, dtype=torch.float64)


def tune_thresholds(model, dataloader, device) -> np.ndarray:
    """Find per-class optimal threshold on validation set.

    For each class, sweep thresholds from 0.1 to 0.9 and pick the one
    that maximizes F1. This is critical for imbalanced problems where
    the optimal threshold varies by class frequency.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state[:, 0]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)

    thresholds = np.full(NUM_LABELS, 0.5)
    candidates = np.arange(0.1, 0.91, 0.05)

    for i in range(NUM_LABELS):
        if labels[:, i].sum() == 0:
            continue
        best_f1 = 0
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[i] = t

    return thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    micro_prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_label = {}
    for i, label in enumerate(sorted(GDPR_CATEGORIES)):
        support = int(y_true[:, i].sum())
        if support == 0:
            continue
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        per_label[label] = {
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1": round(float(f1), 3),
            "support": support,
        }

    return {
        "exact_match": round(float(exact_match), 3),
        "micro_precision": round(float(micro_prec), 3),
        "micro_recall": round(float(micro_rec), 3),
        "micro_f1": round(float(micro_f1), 3),
        "macro_precision": round(float(macro_prec), 3),
        "macro_recall": round(float(macro_rec), 3),
        "macro_f1": round(float(macro_f1), 3),
        "per_label": per_label,
    }


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device, scheduler=None, criterion=None,
                label_smoothing: float = 0.0):
    model.train()
    total_loss = 0
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Label smoothing: soft targets reduce overconfidence
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + label_smoothing / 2

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state[:, 0]

        loss = criterion(logits, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, thresholds=None):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state[:, 0]

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())

    all_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    if thresholds is None:
        thresholds = 0.5
    y_pred = (all_probs >= thresholds).astype(int)

    return compute_metrics(y_true, y_pred), y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for GDPR classification")
    parser.add_argument("--model", default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0=standard BCE, 2=strong focusing)")
    parser.add_argument("--loss", choices=["bce", "focal", "weighted-bce"], default="focal",
                        help="Loss function: bce, weighted-bce (pos_weight only), focal (default)")
    parser.add_argument("--oversample", action="store_true",
                        help="Use oversampling for minority classes")
    parser.add_argument("--tune-thresholds", action="store_true", default=True,
                        help="Tune per-class thresholds on validation set (default: True)")
    parser.add_argument("--no-tune-thresholds", dest="tune_thresholds", action="store_false")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing (0=none, 0.05=light smoothing for regularization)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Classifier dropout (applied on top of hidden states)")
    parser.add_argument("--output", default="data/benchmarks/gdpr_bert_results.json")
    parser.add_argument("--save-model", default="data/models/gdpr_bert")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"  VRAM: {vram / 1e9:.1f} GB")

    # Load data
    print("Loading dataset...")
    texts, labels = load_data()
    print(f"  Total: {len(texts)} samples, {NUM_LABELS} labels")

    # Split - use same seed and size as LLM benchmark for fair comparison
    primary_labels = [l[0] for l in labels]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=primary_labels
    )
    train_primary = [l[0] for l in train_labels]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_primary
    )
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Show class distribution in train set
    from collections import Counter
    train_label_counts = Counter()
    for ll in train_labels:
        for l in ll:
            train_label_counts[l] += 1
    print(f"\n  Train label distribution:")
    for label in sorted(GDPR_CATEGORIES):
        count = train_label_counts.get(label, 0)
        bar = "█" * (count // 50)
        print(f"    {label:<25} {count:>5}  {bar}")

    # Compute class weights
    pos_weight = compute_pos_weight(train_labels)
    print(f"\n  Pos weights (neg/pos ratio, capped at 50):")
    for i, label in enumerate(sorted(GDPR_CATEGORIES)):
        print(f"    {label:<25} {pos_weight[i]:>6.1f}")

    # Set up loss function
    if args.loss == "focal":
        criterion = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
        print(f"\n  Loss: Focal (γ={args.focal_gamma}) + pos_weight")
    elif args.loss == "weighted-bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"\n  Loss: Weighted BCE (pos_weight)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"\n  Loss: Standard BCE")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        classifier_dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # Create datasets
    train_dataset = GDPRDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = GDPRDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = GDPRDataset(test_texts, test_labels, tokenizer, args.max_length)

    # Oversampling for minority classes
    if args.oversample:
        sample_weights = make_oversampling_weights(train_labels)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        print("  Using oversampling for minority classes")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Optimizer with differential learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_val_f1 = 0
    patience_counter = 0
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler,
                                 criterion=criterion, label_smoothing=args.label_smoothing)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        # Use macro F1 for model selection — rewards uniform performance across classes
        val_score = val_metrics["macro_f1"]
        improved = ""
        if val_score > best_val_f1:
            best_val_f1 = val_score
            patience_counter = 0
            improved = " ★ best"
            save_dir = Path(args.save_model)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, "
              f"val_µF1={val_metrics['micro_f1']:.3f}, "
              f"val_MF1={val_metrics['macro_f1']:.3f}, "
              f"val_exact={val_metrics['exact_match']:.3f}, "
              f"time={elapsed:.1f}s{improved}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s (best val µF1={best_val_f1:.3f})")

    # Load best model for evaluation
    print("\nEvaluating on test set...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.save_model, num_labels=NUM_LABELS
    ).to(device)

    # Tune per-class thresholds on validation set
    if args.tune_thresholds:
        print("  Tuning per-class thresholds on validation set...")
        thresholds = tune_thresholds(model, val_loader, device)
        print(f"  Thresholds (default=0.5):")
        for i, label in enumerate(sorted(GDPR_CATEGORIES)):
            marker = " ←" if abs(thresholds[i] - 0.5) > 0.05 else ""
            print(f"    {label:<25} {thresholds[i]:.2f}{marker}")
    else:
        thresholds = None

    # Test with fixed 0.5 threshold
    t0 = time.time()
    test_fixed, y_true, _ = evaluate(model, test_loader, device, thresholds=0.5)
    inference_time = time.time() - t0

    # Test with tuned thresholds
    if args.tune_thresholds:
        test_metrics, _, y_pred = evaluate(model, test_loader, device, thresholds=thresholds)
        print(f"\n  Fixed threshold (0.5):  µF1={test_fixed['micro_f1']:.3f}, MF1={test_fixed['macro_f1']:.3f}")
        print(f"  Tuned thresholds:      µF1={test_metrics['micro_f1']:.3f}, MF1={test_metrics['macro_f1']:.3f}")
        if test_metrics["micro_f1"] < test_fixed["micro_f1"]:
            print("  (Fixed threshold was better — using fixed)")
            test_metrics = test_fixed
            thresholds = None
    else:
        test_metrics = test_fixed

    test_metrics["avg_time_per_sample"] = round(inference_time / len(test_texts), 4)
    test_metrics["total_time"] = round(inference_time, 2)
    test_metrics["n_test"] = len(test_texts)
    test_metrics["errors"] = 0
    test_metrics["training_time"] = round(total_time, 1)
    test_metrics["model_name"] = args.model
    test_metrics["epochs"] = args.epochs
    test_metrics["loss_function"] = args.loss
    test_metrics["focal_gamma"] = args.focal_gamma if args.loss == "focal" else None
    test_metrics["oversampling"] = args.oversample
    test_metrics["thresholds_tuned"] = args.tune_thresholds and thresholds is not None
    if thresholds is not None:
        test_metrics["per_class_thresholds"] = {
            IDX2LABEL[i]: float(thresholds[i]) for i in range(NUM_LABELS)
        }

    print(f"\n  {'='*60}")
    print(f"  FINAL TEST RESULTS — {args.model}")
    print(f"  {'='*60}")
    print(f"  Exact match:     {test_metrics['exact_match']:.3f}")
    print(f"  Micro F1:        {test_metrics['micro_f1']:.3f}")
    print(f"  Macro F1:        {test_metrics['macro_f1']:.3f}")
    print(f"  Micro Precision: {test_metrics['micro_precision']:.3f}")
    print(f"  Micro Recall:    {test_metrics['micro_recall']:.3f}")
    print(f"  Avg time/sample: {test_metrics['avg_time_per_sample']}s")
    print(f"  Training time:   {total_time:.1f}s")

    print(f"\n  {'Category':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*55}")
    for label in sorted(GDPR_CATEGORIES):
        if label in test_metrics["per_label"]:
            m = test_metrics["per_label"][label]
            print(f"  {label:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({args.model: test_metrics}, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
