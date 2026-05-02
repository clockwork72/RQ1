#!/usr/bin/env python3
"""
Train and evaluate a BERT baseline on the GDPR annotation dataset.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


GDPR_CLASSES = [
    "Processing Purpose",
    "Data Categories",
    "Data Recipients",
    "Right to Object",
    "Source of Data",
    "Right to Portability",
    "Provision Requirement",
    "DPO Contact",
    "Withdraw Consent",
    "Right to Restrict",
    "Storage Period",
    "Right to Access",
    "Profiling",
    "Lodge Complaint",
    "Safeguards Copy",
    "Controller Contact",
    "Adequacy Decision",
    "Right to Erase",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(GDPR_CLASSES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class Example:
    row_id: str
    domain: str
    text: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT baseline for GDPR classification")
    parser.add_argument("--dataset", required=True, help="Path to gdpr_dataset_1.2.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and reports")
    parser.add_argument("--model-name", default="bert-base-uncased", help="HF model identifier")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--train-subset-per-class", type=int, default=0, help="Cap train rows per class; 0 uses all")
    parser.add_argument("--val-subset-per-class", type=int, default=0, help="Cap validation rows per class; 0 uses all")
    parser.add_argument("--test-subset-per-class", type=int, default=0, help="Cap test rows per class; 0 uses all")
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Early stopping patience in eval rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Checkpoint limit")
    return parser.parse_args()


def load_examples(dataset_path: str) -> list[Example]:
    examples: list[Example] = []
    with open(dataset_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            row_id, domain, text, raw_label = row[0], row[1], row[2], row[3]
            if not text.strip():
                continue
            try:
                labels = ast.literal_eval(raw_label)
            except (ValueError, SyntaxError):
                labels = [raw_label]
            if not labels:
                continue
            label = str(labels[0]).strip()
            if label not in LABEL_TO_ID:
                continue
            examples.append(Example(row_id=row_id, domain=domain, text=text, label=label))
    return examples


def subset_per_class(examples: list[Example], limit: int, seed: int) -> list[Example]:
    if limit <= 0:
        return examples
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)
    rng = random.Random(seed)
    reduced: list[Example] = []
    for label in GDPR_CLASSES:
        bucket = grouped.get(label, [])
        if len(bucket) <= limit:
            reduced.extend(bucket)
        else:
            reduced.extend(rng.sample(bucket, limit))
    return reduced


def split_examples(examples: list[Example], args: argparse.Namespace) -> tuple[list[Example], list[Example], list[Example]]:
    if not np.isclose(args.train_fraction + args.val_fraction + args.test_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    labels = [example.label for example in examples]
    train_examples, test_examples = train_test_split(
        examples,
        test_size=args.test_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    remaining_val_fraction = args.val_fraction / (args.train_fraction + args.val_fraction)
    remaining_labels = [example.label for example in train_examples]
    train_examples, val_examples = train_test_split(
        train_examples,
        test_size=remaining_val_fraction,
        random_state=args.seed,
        stratify=remaining_labels,
    )

    train_examples = subset_per_class(train_examples, args.train_subset_per_class, args.seed)
    val_examples = subset_per_class(val_examples, args.val_subset_per_class, args.seed + 1)
    test_examples = subset_per_class(test_examples, args.test_subset_per_class, args.seed + 2)
    return train_examples, val_examples, test_examples


def examples_to_dataset(examples: list[Example]) -> Dataset:
    payload = {
        "row_id": [item.row_id for item in examples],
        "domain": [item.domain for item in examples],
        "text": [item.text for item in examples],
        "label_text": [item.label for item in examples],
        "labels": [LABEL_TO_ID[item.label] for item in examples],
    }
    return Dataset.from_dict(payload)


def build_dataset_dict(
    train_examples: list[Example],
    val_examples: list[Example],
    test_examples: list[Example],
) -> DatasetDict:
    return DatasetDict(
        train=examples_to_dataset(train_examples),
        validation=examples_to_dataset(val_examples),
        test=examples_to_dataset(test_examples),
    )


def summarize_split(examples: list[Example]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts.get(label, 0) for label in GDPR_CLASSES}


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def tokenizer_fn(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset_dict.map(tokenizer_fn, batched=True)
    return tokenized.remove_columns(["text", "label_text", "domain", "row_id"])


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_predictions(path: Path, dataset: Dataset, predictions: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "domain", "gold_label", "pred_label", "correct", "text"])
        for row_id, domain, gold_id, pred_id, text in zip(
            dataset["row_id"],
            dataset["domain"],
            dataset["labels"],
            predictions.tolist(),
            dataset["text"],
        ):
            gold_label = ID_TO_LABEL[int(gold_id)]
            pred_label = ID_TO_LABEL[int(pred_id)]
            writer.writerow([row_id, domain, gold_label, pred_label, pred_label == gold_label, text])


def save_loss_curves(trainer: Trainer, output_dir: Path) -> None:
    history = trainer.state.log_history
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps) + 1))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps) + 1))
            eval_loss.append(entry["eval_loss"])

    if train_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(train_steps, train_loss, label="train_loss")
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label="eval_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png")
        plt.close()


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    examples = load_examples(args.dataset)
    train_examples, val_examples, test_examples = split_examples(examples, args)
    split_summary = {
        "train": summarize_split(train_examples),
        "validation": summarize_split(val_examples),
        "test": summarize_split(test_examples),
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "test_examples": len(test_examples),
    }
    save_json(output_dir / "split_summary.json", split_summary)

    raw_dataset = build_dataset_dict(train_examples, val_examples, test_examples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(GDPR_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    precision_bf16 = bf16_supported()
    precision_fp16 = torch.cuda.is_available() and not precision_bf16
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        fp16=precision_fp16,
        bf16=precision_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"], metric_key_prefix="validation")
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_predictions = np.argmax(test_output.predictions, axis=-1)
    test_labels = np.array(raw_dataset["test"]["labels"])

    report_text = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(GDPR_CLASSES))),
        target_names=GDPR_CLASSES,
        zero_division=0,
    )
    report_json = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(GDPR_CLASSES))),
        target_names=GDPR_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "model_name": args.model_name,
        "seed": args.seed,
        "dataset": os.path.abspath(args.dataset),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "training_args": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "train_subset_per_class": args.train_subset_per_class,
            "val_subset_per_class": args.val_subset_per_class,
            "test_subset_per_class": args.test_subset_per_class,
        },
        "split_summary": split_summary,
        "validation_metrics": eval_metrics,
        "test_metrics": test_output.metrics,
        "test_classification_report": report_json,
    }

    save_json(output_dir / "metrics_summary.json", summary)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)
    with open(output_dir / "trainer_state.json", "w", encoding="utf-8") as handle:
        handle.write(trainer.state.to_json_string())
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    save_predictions(output_dir / "test_predictions.csv", raw_dataset["test"], test_predictions)
    save_loss_curves(trainer, output_dir)

    print(json.dumps(summary["validation_metrics"], indent=2))
    print(json.dumps(summary["test_metrics"], indent=2))
    print(report_text)


if __name__ == "__main__":
    main()
