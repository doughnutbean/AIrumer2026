"""Export Transformer voter probabilities and labels for a CSV file.

This script is a thin wrapper around the existing Transformer voter prediction
pipeline. It writes the probability outputs to a JSONL file that can be used for
analysis, fusion, or downstream selective correction experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependencies. Install: pip install torch transformers datasets accelerate") from exc

try:
    from .text_utils import preprocess_text
except ImportError:
    from text_utils import preprocess_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Transformer voter probabilities for a CSV file.")
    parser.add_argument("--input", default="dataset/val.csv", help="Input CSV file.")
    parser.add_argument("--manifest", default="models/transformer_voter/manifest.json", help="Transformer voter manifest.")
    parser.add_argument("--out", default="outputs/transformer_voter_val.jsonl", help="Output JSONL path. Avoid overwriting retained final-fusion predictions by default.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument(
        "--add-base-alias",
        action="store_true",
        help="Also write base_label/base_prob_1 aliases so the output can be consumed directly by fn_aware_selective_correction.py.",
    )
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_one_model(model_path: str, texts: List[str], max_length: int, batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            probs.append(softmax(logits))
    return np.vstack(probs)


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    df = pd.read_csv(args.input)
    texts = [preprocess_text(t) for t in df["text"].fillna("").astype(str).tolist()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_probs = []
    for seed_info in manifest["seeds"]:
        prob = predict_one_model(seed_info["path"], texts, int(manifest.get("max_length", 128)), args.batch_size, device)
        seed_probs.append(prob)
        print(f"Predicted with {seed_info['path']}")
    avg_prob = np.mean(seed_probs, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in df.iterrows():
        record = {
            "index": int(i),
            "id": int(row.get("id", i)),
            "text": str(row["text"]),
            "true_label": int(row["label"]) if "label" in row else None,
            "transformer_prob_0": float(avg_prob[i, 0]),
            "transformer_prob_1": float(avg_prob[i, 1]),
            "transformer_label": int(avg_prob[i, 1] >= 0.5),
        }
        if args.add_base_alias:
            record["base_prob_1"] = record["transformer_prob_1"]
            record["base_label"] = record["transformer_label"]
        records.append(record)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    print(f"Saved Transformer voter export to {out_path}")


if __name__ == "__main__":
    main()
