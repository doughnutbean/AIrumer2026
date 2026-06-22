"""Build final-fusion prediction JSONL for AIrumer.

This script fills the missing bridge between model training/export and
``fn_aware_selective_correction.py``.  It combines:

1. the local AIrumer ensemble probability,
2. an optional separately trained dedicated local model probability,
3. optional Transformer-voter probabilities exported as JSONL, and
4. TF-IDF nearest-neighbour retrieval features.

The generated JSONL is intentionally compatible with
``src/fn_aware_selective_correction.py``.  By default the conservative
``final_label`` equals the local ``base_label``; the auxiliary features are used
by the downstream FN-aware review step to decide which base non-rumor samples
should be reviewed by an LLM or oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from .hybrid_classifier import EnsembleRumourDetectClass
    from .retriever import SimilarExampleRetriever
except ImportError:  # pragma: no cover - supports ``python src/build_final_fusion_predictions.py``
    from hybrid_classifier import EnsembleRumourDetectClass
    from retriever import SimilarExampleRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AIrumer final-fusion predictions from local model, Transformer output and retrieval features."
    )
    parser.add_argument("--input", default="dataset/val.csv", help="CSV to predict. Must contain text and optionally label/id/event.")
    parser.add_argument("--model", default="models/ensemble.joblib", help="Base local AIrumer ensemble joblib.")
    parser.add_argument(
        "--dedicated-model",
        default="",
        help=(
            "Optional dedicated_local.joblib trained by train_dedicated_local.py. "
            "If omitted or missing, dedicated_prob_1 falls back to the base model probability."
        ),
    )
    parser.add_argument("--retriever", default="models/retriever.joblib", help="Retriever joblib built from train.csv.")
    parser.add_argument(
        "--transformer-predictions",
        default="outputs/transformer_voter_val.jsonl",
        help="Optional JSONL from export_transformer_voter_predictions.py / predict_transformer_voter.py.",
    )
    parser.add_argument("--out", default="outputs/final_fusion_with_transformer/predictions.jsonl")
    parser.add_argument("--metrics", default="outputs/final_fusion_with_transformer/metrics.json")
    parser.add_argument("--top-k", type=int, default=7, help="Number of retrieved neighbours to save.")
    parser.add_argument(
        "--fusion-mode",
        choices=["conservative", "rule"],
        default="conservative",
        help=(
            "conservative: final_label equals base_label; rule: apply a simple deterministic auxiliary-feature rule. "
            "For the reported AIrumer score, use conservative and run FN-aware LLM correction downstream."
        ),
    )
    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=1.55,
        help="Risk threshold used only when --fusion-mode rule is selected.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_transformer_lookup(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if "index" in row and row["index"] is not None:
            by_index[int(row["index"])] = row
        if "id" in row and row["id"] is not None:
            try:
                by_id[int(row["id"])] = row
            except (TypeError, ValueError):
                pass
    return by_index, by_id


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_row_id(row: pd.Series, index: int) -> int:
    if "id" in row:
        return safe_int(row.get("id"), index)
    return index


def compute_retrieval_features(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [safe_int(ex.get("label", 0)) for ex in examples]
    top1 = examples[0] if examples else {}

    def ratio(k: int) -> float:
        subset = labels[:k]
        return float(sum(subset) / len(subset)) if subset else 0.0

    return {
        "top1_label": safe_int(top1.get("label", 0), 0),
        "top1_similarity": safe_float(top1.get("similarity", 0.0), 0.0),
        "knn_rumor_ratio_top3": ratio(3),
        "knn_rumor_ratio_top5": ratio(5),
        "knn_rumor_ratio_top7": ratio(7),
        "retrieved_examples": examples,
    }


def compute_suspicious_score(row: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Compute a review-priority score for base non-rumor predictions.

    The score is deliberately conservative.  It is metadata for downstream
    selective correction, not a claim that the sample should be flipped.
    """
    if int(row.get("base_label", 0)) != 0:
        return 0.0, []

    score = 0.0
    evidence: List[str] = []
    base_prob = safe_float(row.get("base_prob_1"), 0.0)
    dedicated_prob = safe_float(row.get("dedicated_prob_1"), base_prob)
    transformer_prob = safe_float(row.get("transformer_prob_1"), 0.0)
    knn3 = safe_float(row.get("knn_rumor_ratio_top3"), 0.0)
    knn5 = safe_float(row.get("knn_rumor_ratio_top5"), 0.0)
    knn7 = safe_float(row.get("knn_rumor_ratio_top7"), 0.0)
    top1_label = safe_int(row.get("top1_label"), 0)
    top1_similarity = safe_float(row.get("top1_similarity"), 0.0)

    if dedicated_prob >= 0.55:
        score += 0.9
        evidence.append(f"dedicated_prob_1={dedicated_prob:.2f}")
    elif dedicated_prob >= 0.42:
        score += 0.65
        evidence.append(f"dedicated_prob_1={dedicated_prob:.2f}")
    elif dedicated_prob >= 0.25:
        score += 0.25
        evidence.append(f"dedicated_prob_1={dedicated_prob:.2f}")

    if transformer_prob >= 0.45:
        score += 0.8
        evidence.append(f"transformer_prob_1={transformer_prob:.2f}")
    elif transformer_prob >= 0.20:
        score += 0.3
        evidence.append(f"transformer_prob_1={transformer_prob:.2f}")

    if knn3 >= 1.0:
        score += 1.0
        evidence.append(f"knn_rumor_ratio_top3={knn3:.2f}")
    elif knn3 >= 0.67:
        score += 0.65
        evidence.append(f"knn_rumor_ratio_top3={knn3:.2f}")

    if knn5 >= 0.80:
        score += 0.55
        evidence.append(f"knn_rumor_ratio_top5={knn5:.2f}")
    elif knn5 >= 0.60:
        score += 0.35
        evidence.append(f"knn_rumor_ratio_top5={knn5:.2f}")

    if knn7 >= 0.57:
        score += 0.35
        evidence.append(f"knn_rumor_ratio_top7={knn7:.2f}")

    if top1_label == 1 and top1_similarity >= 0.55:
        score += 1.0
        evidence.append(f"top1_label=1,similarity={top1_similarity:.2f}")
    elif top1_label == 1 and top1_similarity >= 0.35:
        score += 0.75
        evidence.append(f"top1_label=1,similarity={top1_similarity:.2f}")
    elif top1_label == 1 and top1_similarity >= 0.20:
        score += 0.35
        evidence.append(f"top1_label=1,similarity={top1_similarity:.2f}")

    if base_prob >= 0.50:
        score += 0.35
        evidence.append(f"base_prob_1={base_prob:.2f}")
    elif base_prob >= 0.40:
        score += 0.2
        evidence.append(f"base_prob_1={base_prob:.2f}")

    if top1_label == 0 and top1_similarity >= 0.35:
        score -= 0.35
        evidence.append(f"non_rumor_top1_penalty={top1_similarity:.2f}")

    return float(score), evidence


def decide_final_label(row: Dict[str, Any], mode: str, rule_threshold: float) -> int:
    base_label = int(row["base_label"])
    if mode == "conservative":
        return base_label

    # Optional deterministic rule.  It is intentionally only allowed to recover
    # likely false negatives and never flips 1 -> 0.
    if base_label == 0 and safe_float(row.get("suspicious_score"), 0.0) >= rule_threshold:
        return 1
    return base_label


def predict_probability(model: EnsembleRumourDetectClass, text: str) -> Tuple[int, float, float]:
    probs = np.asarray(model.predict_proba(text), dtype=float)
    prob_1 = float(probs[1])
    label = int(prob_1 >= float(model.threshold))
    confidence = prob_1 if label == 1 else 1.0 - prob_1
    return label, prob_1, float(confidence)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError(f"{input_path} must contain a 'text' column")

    base_model = EnsembleRumourDetectClass(args.model)
    dedicated_model: Optional[EnsembleRumourDetectClass] = None
    if args.dedicated_model and Path(args.dedicated_model).exists():
        dedicated_model = EnsembleRumourDetectClass(args.dedicated_model)

    retriever = SimilarExampleRetriever(args.retriever)
    transformer_rows = load_jsonl(args.transformer_predictions)
    transformer_by_index, transformer_by_id = build_transformer_lookup(transformer_rows)
    transformer_available = bool(transformer_rows)

    records: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        text = str(row["text"] if not pd.isna(row["text"]) else "")
        row_id = get_row_id(row, int(index))
        base_label, base_prob_1, base_confidence = predict_probability(base_model, text)

        if dedicated_model is not None:
            dedicated_label, dedicated_prob_1, _ = predict_probability(dedicated_model, text)
        else:
            dedicated_label, dedicated_prob_1 = base_label, base_prob_1

        transformer_row = transformer_by_index.get(int(index), transformer_by_id.get(row_id, {}))
        transformer_prob_1 = safe_float(transformer_row.get("transformer_prob_1"), 0.0)
        transformer_label = safe_int(transformer_row.get("transformer_label"), int(transformer_prob_1 >= 0.5))

        retrieved_examples = retriever.retrieve(text, k=args.top_k)
        retrieval_features = compute_retrieval_features(retrieved_examples)

        record: Dict[str, Any] = {
            "index": int(index),
            "id": row_id,
            "text": text,
            "true_label": safe_int(row.get("label"), None) if "label" in row else None,
            "base_label": int(base_label),
            "base_prob_1": float(base_prob_1),
            "base_confidence": float(base_confidence),
            "dedicated_label": int(dedicated_label),
            "dedicated_prob_1": float(dedicated_prob_1),
            "dedicated_prob_0": float(1.0 - dedicated_prob_1),
            "transformer_prob_1": float(transformer_prob_1),
            "transformer_label": int(transformer_label),
            "transformer_available": bool(transformer_available),
        }
        if "event" in row:
            record["event"] = safe_int(row.get("event"), 0)
        record.update(retrieval_features)
        suspicious_score, evidence = compute_suspicious_score(record)
        record["suspicious_score"] = suspicious_score
        record["evidence"] = evidence
        record["final_label"] = decide_final_label(record, args.fusion_mode, args.rule_threshold)
        records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    metrics: Dict[str, Any] = {
        "input": str(input_path),
        "model": args.model,
        "dedicated_model": args.dedicated_model if dedicated_model is not None else None,
        "retriever": args.retriever,
        "transformer_predictions": args.transformer_predictions if transformer_available else None,
        "fusion_mode": args.fusion_mode,
        "rows": len(records),
        "output": str(out_path),
    }
    y_true = [r["true_label"] for r in records if r.get("true_label") is not None]
    if len(y_true) == len(records) and records:
        base_pred = [int(r["base_label"]) for r in records]
        final_pred = [int(r["final_label"]) for r in records]
        transformer_pred = [int(r["transformer_label"]) for r in records]
        dedicated_pred = [int(r["dedicated_label"]) for r in records]
        metrics.update(
            {
                "base_accuracy": float(accuracy_score(y_true, base_pred)),
                "base_confusion_matrix": confusion_matrix(y_true, base_pred).tolist(),
                "dedicated_accuracy": float(accuracy_score(y_true, dedicated_pred)),
                "dedicated_confusion_matrix": confusion_matrix(y_true, dedicated_pred).tolist(),
                "final_accuracy": float(accuracy_score(y_true, final_pred)),
                "final_confusion_matrix": confusion_matrix(y_true, final_pred).tolist(),
                "classification_report": classification_report(y_true, final_pred, digits=4, output_dict=True),
            }
        )
        if transformer_available:
            metrics["transformer_accuracy"] = float(accuracy_score(y_true, transformer_pred))
            metrics["transformer_confusion_matrix"] = confusion_matrix(y_true, transformer_pred).tolist()

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved final-fusion predictions to {out_path}")
    print(f"Saved metrics to {metrics_path}")
    if "final_accuracy" in metrics:
        print(f"base_accuracy = {metrics['base_accuracy']:.6f}")
        print(f"final_accuracy = {metrics['final_accuracy']:.6f}")
        print(f"final_confusion_matrix = {metrics['final_confusion_matrix']}")


if __name__ == "__main__":
    main()
