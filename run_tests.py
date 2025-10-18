import argparse
import json
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser(description="Run behavioral minimal-pair tests on a classification model")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--tests_file", type=str, required=True, help="JSONL with fields: text_a, text_b, expected_relation (e.g., 'flip') and optional label_a, label_b")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--report_path", type=str, required=True)
    return p.parse_args()


def softmax(x):
    e_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return e_x / e_x.sum(dim=-1, keepdim=True)


def predict(model, tokenizer, texts: List[str], max_length: int, device: torch.device):
    enc = tokenizer(texts, truncation=True, max_length=max_length, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = softmax(logits).cpu().tolist()
        preds = logits.argmax(dim=-1).cpu().tolist()
    return preds, probs


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    tests = load_jsonl(args.tests_file)

    results = []
    correct_relation = 0
    for t in tests:
        a = t["text_a"].strip()
        b = t["text_b"].strip()
        expected = t.get("expected_relation", "flip")  # 'flip' or 'same'

        preds, probs = predict(model, tokenizer, [a, b], args.max_length, device)
        pred_a, pred_b = preds[0], preds[1]

        relation_ok = (pred_a != pred_b) if expected == "flip" else (pred_a == pred_b)
        correct_relation += int(relation_ok)

        results.append({
            "text_a": a,
            "text_b": b,
            "pred_a": int(pred_a),
            "pred_b": int(pred_b),
            "expected_relation": expected,
            "relation_ok": bool(relation_ok),
            "probs_a": probs[0],
            "probs_b": probs[1],
        })

    summary = {
        "total_pairs": len(tests),
        "relation_accuracy": (correct_relation / max(1, len(tests))),
    }

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
