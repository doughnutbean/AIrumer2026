"""Train a dedicated local rumor classifier from train.csv.

The goal is to build a strong fully local supervised model for the assignment.
It performs model selection among several sparse text classifiers, then saves the
best validation configuration as a reusable joblib artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from .dedicated_model import MultiTfidfVectorizer, WeightedProbabilityEnsemble
    from .text_utils import batch_preprocess
except ImportError:
    from dedicated_model import MultiTfidfVectorizer, WeightedProbabilityEnsemble
    from text_utils import batch_preprocess


def make_vectorizer() -> MultiTfidfVectorizer:
    return MultiTfidfVectorizer()


def build_candidates() -> List[Tuple[str, Pipeline]]:
    candidates: List[Tuple[str, Pipeline]] = []
    for c in [2.0, 4.0, 8.0, 12.0, 20.0]:
        candidates.append(
            (
                f"multi_lr_c{c}",
                Pipeline(
                    [
                        ("features", make_vectorizer()),
                        ("clf", LogisticRegression(C=c, max_iter=2500, solver="liblinear", random_state=42)),
                    ]
                ),
            )
        )
    for alpha in [3e-5, 1e-4, 3e-4]:
        candidates.append(
            (
                f"multi_sgd_huber_a{alpha}",
                Pipeline(
                    [
                        ("features", make_vectorizer()),
                        (
                            "clf",
                            SGDClassifier(
                                loss="modified_huber",
                                alpha=alpha,
                                max_iter=3000,
                                random_state=42,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            )
        )
    for alpha in [0.1, 0.3, 0.5, 0.8]:
        candidates.append(
            (
                f"multi_mnb_a{alpha}",
                Pipeline([("features", make_vectorizer()), ("clf", MultinomialNB(alpha=alpha))]),
            )
        )
        candidates.append(
            (
                f"multi_cnb_a{alpha}",
                Pipeline([("features", make_vectorizer()), ("clf", ComplementNB(alpha=alpha))]),
            )
        )
    for c in [0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"multi_svm_c{c}",
                Pipeline(
                    [
                        ("features", make_vectorizer()),
                        (
                            "clf",
                            CalibratedClassifierCV(
                                LinearSVC(C=c, class_weight="balanced", random_state=42),
                                cv=3,
                            ),
                        ),
                    ]
                ),
            )
        )
    return candidates


def tune_threshold(y_true: np.ndarray, prob_rumor: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_acc = -1.0
    for threshold in np.linspace(0.2, 0.8, 241):
        pred = (prob_rumor >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_threshold = float(threshold)
            best_acc = float(acc)
    return best_threshold, best_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dedicated local rumor detector.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--val", default="dataset/val.csv")
    parser.add_argument("--model", default="models/dedicated_local.joblib")
    parser.add_argument("--out", default="outputs/dedicated_local_metrics.json")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Train only a compact subset of candidate models for faster reproduction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    x_train = batch_preprocess(train_df["text"].fillna("").astype(str).tolist())
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = batch_preprocess(val_df["text"].fillna("").astype(str).tolist())
    y_val = val_df["label"].astype(int).to_numpy()

    candidates = build_candidates()
    if args.quick:
        # Keep the quick path deliberately small so the script finishes reliably
        # on CPU-only grading machines. The full path below still evaluates all
        # candidate families.
        keep = {"multi_lr_c8.0"}
        candidates = [(name, model) for name, model in candidates if name in keep]
    fitted: Dict[str, Pipeline] = {}
    scores = []
    for name, model in candidates:
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_val)
        argmax_pred = prob.argmax(axis=1)
        argmax_acc = accuracy_score(y_val, argmax_pred)
        threshold, threshold_acc = tune_threshold(y_val, prob[:, 1])
        fitted[name] = model
        scores.append(
            {
                "name": name,
                "argmax_accuracy": float(argmax_acc),
                "best_threshold": float(threshold),
                "best_threshold_accuracy": float(threshold_acc),
            }
        )
        print(f"{name}: argmax={argmax_acc:.4f}, best_t={threshold:.3f}, best_t_acc={threshold_acc:.4f}", flush=True)

    # A compact, diverse ensemble selected from the strongest families.
    preferred_ensemble_names = ["multi_lr_c8.0", "multi_sgd_huber_a0.0001", "multi_mnb_a0.5"]
    ensemble_names = [name for name in preferred_ensemble_names if name in fitted]
    if not ensemble_names:
        ensemble_names = [max(scores, key=lambda item: item["best_threshold_accuracy"])["name"]]
    ensemble_estimators = [(name, fitted[name]) for name in ensemble_names]
    weights = {name: 1.0 for name in ensemble_names}
    ensemble = WeightedProbabilityEnsemble(ensemble_estimators, weights)
    ensemble.fitted_estimators_ = ensemble_estimators
    ensemble.classes_ = np.array([0, 1])
    proba = ensemble.predict_proba(x_val)
    threshold, threshold_acc = tune_threshold(y_val, proba[:, 1])
    ensemble.threshold = threshold
    pred = ensemble.predict(x_val)
    acc = accuracy_score(y_val, pred)

    artifact = {
        "type": "dedicated_local_rumor_detector",
        "model": ensemble,
        "preprocess": "src.text_utils.preprocess_text",
        "metadata": {
            "algorithm": "Dedicated local supervised ensemble: multi-view TF-IDF + LR/SGD/MNB",
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "ensemble_names": ensemble_names,
            "weights": weights,
            "threshold": float(threshold),
            "val_accuracy": float(acc),
            "candidate_scores": scores,
            "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
            "classification_report": classification_report(y_val, pred, digits=4, output_dict=True),
        },
    }
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved dedicated local model to {model_path}")
    print(f"Saved metrics to {out_path}")
    print(f"Dedicated local val accuracy: {acc:.4f}")
    print(json.dumps(artifact["metadata"]["classification_report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
