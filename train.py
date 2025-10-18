import argparse
import json
import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text classifier with HF Trainer")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def build_label_maps(labels: List[int]):
    unique = sorted(list({int(x) for x in labels}))
    label2id = {str(v): i for i, v in enumerate(unique)}
    id2label = {i: str(v) for i, v in enumerate(unique)}
    return label2id, id2label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary" if len(set(labels))==2 else "macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary" if len(set(labels))==2 else "macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="binary" if len(set(labels))==2 else "macro", zero_division=0)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    data_files = {"train": args.train_file, "validation": args.validation_file}
    raw = load_dataset("json", data_files=data_files)

    # Ensure fields exist
    for split in raw:
        assert "text" in raw[split].column_names and "label" in raw[split].column_names, "JSONL must have 'text' and 'label'"

    # Label maps (assume numeric labels but normalize to consecutive ids)
    label2id, id2label = build_label_maps(raw["train"]["label"])

    def normalize_labels(example: Dict):
        # Map original label to 0..N-1 according to label2id
        example["label"] = label2id[str(int(example["label"]))]
        return example

    raw = raw.map(normalize_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def tokenize_fn(batch: Dict[str, List[str]]):
        return tokenizer(batch["text"], padding=False, truncation=True, max_length=args.max_length)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=[c for c in raw["train"].column_names if c not in ["label"]])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label2id),
        id2label={i: l for i, l in id2label.items()},
        label2id={l: i for l, i in label2id.items()},
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        seed=args.seed,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save label maps
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    # Evaluate best model on validation and save
    val_metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "eval_validation.json"), "w") as f:
        json.dump({k: float(v) for k, v in val_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
