"""TF-IDF nearest-neighbour retriever for few-shot LLM prompting."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .text_utils import batch_preprocess, preprocess_text
except ImportError:
    from text_utils import batch_preprocess, preprocess_text


class SimilarExampleRetriever:
    """Retrieve top-k similar labelled training examples."""

    def __init__(self, artifact_path: str | Path = "models/retriever.joblib") -> None:
        artifact = joblib.load(artifact_path)
        self.vectorizer: TfidfVectorizer = artifact["vectorizer"]
        self.matrix = artifact["matrix"]
        self.examples: List[Dict[str, object]] = artifact["examples"]

    def retrieve(self, text: str, k: int = 5) -> List[Dict[str, object]]:
        query = self.vectorizer.transform([preprocess_text(text)])
        sims = cosine_similarity(query, self.matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            item = dict(self.examples[int(idx)])
            item["similarity"] = float(sims[int(idx)])
            results.append(item)
        return results


def build_retriever(train_path: str | Path, output_path: str | Path) -> None:
    train_df = pd.read_csv(train_path)
    texts = train_df["text"].fillna("").astype(str).tolist()
    processed = batch_preprocess(texts)
    labels = train_df["label"].astype(int).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform(processed)
    examples = [
        {"text": text, "processed_text": proc, "label": int(label)}
        for text, proc, label in zip(texts, processed, labels)
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "examples": examples}, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TF-IDF retriever for LLM few-shot examples.")
    parser.add_argument("--train", default="dataset/train.csv")
    parser.add_argument("--out", default="models/retriever.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_retriever(args.train, args.out)
    print(f"Saved retriever to {args.out}")


if __name__ == "__main__":
    main()
