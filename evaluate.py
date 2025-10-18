import argparse
import json
import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned classifier on a test JSONL")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--metrics", type=str, default="f1,precision,recall,accuracy")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    ds = load_dataset("json", data_files={"test": args.test_file})

    def tokenize_fn(batch: Dict[str, List[str]]):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(tokenize_fn, batched=True)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    bs = 64
    input_cols = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in ds["test"].column_names else [])

    for idxs in chunks(list(range(len(ds["test"]))), bs):
        batch = {k: [ds["test"][k][i] for i in idxs] for k in input_cols}
        batch = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        labels = [int(ds["test"]["label"][i]) for i in idxs]
        all_preds.extend(preds)
        all_labels.extend(labels)

    results = {}
    if "accuracy" in metric_list:
        results["accuracy"] = float(accuracy_score(all_labels, all_preds))
    if "precision" in metric_list:
        avg = "binary" if len(set(all_labels)) == 2 else "macro"
        results["precision"] = float(precision_score(all_labels, all_preds, average=avg, zero_division=0))
    if "recall" in metric_list:
        avg = "binary" if len(set(all_labels)) == 2 else "macro"
        results["recall"] = float(recall_score(all_labels, all_preds, average=avg, zero_division=0))
    if "f1" in metric_list:
        avg = "binary" if len(set(all_labels)) == 2 else "macro"
        results["f1"] = float(f1_score(all_labels, all_preds, average=avg, zero_division=0))

    print(json.dumps(results, indent=2))

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "eval_test.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
