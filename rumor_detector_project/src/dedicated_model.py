"""Reusable classes for the dedicated local rumor detector."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class MultiTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Combine word, char_wb and hashtag-oriented TF-IDF features."""

    def __init__(
        self,
        word_max_features: int = 50000,
        char_max_features: int = 80000,
        hashtag_max_features: int = 20000,
        word_ngram_range: Tuple[int, int] = (1, 3),
        char_ngram_range: Tuple[int, int] = (3, 6),
    ) -> None:
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.hashtag_max_features = hashtag_max_features
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range

    def fit(self, x: List[str], y=None):
        self.word_vectorizer_ = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=self.word_ngram_range,
            min_df=1,
            max_features=self.word_max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.char_vectorizer_ = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            min_df=1,
            max_features=self.char_max_features,
            sublinear_tf=True,
        )
        self.hashtag_vectorizer_ = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.hashtag_max_features,
            sublinear_tf=True,
            token_pattern=r"#?\b\w\w+\b",
        )
        self.word_vectorizer_.fit(x)
        self.char_vectorizer_.fit(x)
        self.hashtag_vectorizer_.fit(x)
        return self

    def transform(self, x: List[str]):
        return hstack(
            [
                self.word_vectorizer_.transform(x),
                self.char_vectorizer_.transform(x),
                self.hashtag_vectorizer_.transform(x),
            ],
            format="csr",
        )

    def get_feature_names_out(self):
        word = np.array([f"word:{v}" for v in self.word_vectorizer_.get_feature_names_out()])
        char = np.array([f"char:{v}" for v in self.char_vectorizer_.get_feature_names_out()])
        hashtag = np.array([f"tag:{v}" for v in self.hashtag_vectorizer_.get_feature_names_out()])
        return np.concatenate([word, char, hashtag])


class WeightedProbabilityEnsemble(BaseEstimator, ClassifierMixin):
    """Fit several probabilistic pipelines and average their probabilities."""

    def __init__(self, estimators, weights, threshold: float = 0.5):
        self.estimators = estimators
        self.weights = weights
        self.threshold = threshold

    def fit(self, x, y):
        self.fitted_estimators_ = []
        self.classes_ = np.array([0, 1])
        for name, estimator in self.estimators:
            estimator.fit(x, y)
            self.fitted_estimators_.append((name, estimator))
        return self

    def predict_proba(self, x):
        probs = []
        weights = []
        for name, estimator in self.fitted_estimators_:
            probs.append(estimator.predict_proba(x))
            weights.append(float(self.weights.get(name, 1.0)))
        weight_arr = np.asarray(weights, dtype=float)
        weight_arr = weight_arr / weight_arr.sum()
        return np.tensordot(weight_arr, np.stack(probs, axis=0), axes=(0, 0))

    def predict(self, x):
        probs = self.predict_proba(x)
        return (probs[:, 1] >= self.threshold).astype(int)
