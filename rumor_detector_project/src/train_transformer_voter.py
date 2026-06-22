"""Fine-tune a tweet-oriented Transformer voter for rumor detection.

Recommended models:
- vinai/bertweet-base
- cardiffnlp/twitter-roberta-base-sentiment
- roberta-base / distilroberta-base as fallbacks

The saved models are not intended to replace the conservative local backbone;
they provide a probability voter for suspected 0->1 false negatives.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing transformer dependencies. Install with: pip install torch transformers datasets accelerate"
    ) from exc

try:
    from .text_utils import preprocess_text
except ImportError:
    from text_utils import preprocess_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_dict(
        {
            "text": [preprocess_text(t) for t in df["text"].fillna("").astype(str).tolist()],
            "labels": df["label"].astype(int).tolist(),
        }
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Transformer rumor voter.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--val", default="dataset/val.csv")
    parser.add_argument("--model-name", default="vinai/bertweet-base")
    parser.add_argument("--out-dir", default="models/transformer_voter")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--metric-for-best-model", default="f1", choices=["f1", "accuracy"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(args.train)
    val_ds = build_dataset(args.val)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, normalization=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_val = val_ds.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    seed_results: List[Dict[str, object]] = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        set_seed(seed)
        seed_dir = out_root / f"seed_{seed}"
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        training_kwargs = {
            "output_dir": str(seed_dir),
            "save_strategy": "epoch",
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size * 2,
            "num_train_epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "load_best_model_at_end": True,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": True,
            "logging_steps": 20,
            "save_total_limit": 1,
            "seed": seed,
            "report_to": [],
        }
        # transformers changed the argument name from evaluation_strategy to eval_strategy in some releases.
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            training_kwargs["eval_strategy"] = "epoch"
        else:
            training_kwargs["evaluation_strategy"] = "epoch"
        training_args = TrainingArguments(**training_kwargs)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(str(seed_dir / "best"))
        tokenizer.save_pretrained(str(seed_dir / "best"))
        seed_results.append({"seed": seed, "metrics": metrics, "path": str(seed_dir / "best")})
        print(json.dumps(seed_results[-1], ensure_ascii=False, indent=2))

    manifest = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "seeds": seed_results,
        "usage": "Use predict_transformer_voter.py to produce transformer_prob_1 for final fusion.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest to {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
