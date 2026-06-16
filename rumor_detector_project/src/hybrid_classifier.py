"""Hybrid rumor detector: local ensemble + retrieval + optional DeepSeek review."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

try:
    from .config_utils import load_config
    from .llm_reranker import LLMReranker
    from .retriever import SimilarExampleRetriever
    from .text_utils import preprocess_text
except ImportError:
    from config_utils import load_config
    from llm_reranker import LLMReranker
    from retriever import SimilarExampleRetriever
    from text_utils import preprocess_text


LABEL_NAMES = {0: "非谣言", 1: "谣言"}


class EnsembleRumourDetectClass:
    """Local weighted ensemble with model-internal explanations."""

    def __init__(self, model_path: str | Path = "models/ensemble.joblib") -> None:
        artifact = joblib.load(model_path)
        self.artifact_type = artifact.get("type", "tfidf_ensemble")
        self.metadata = artifact.get("metadata", {})
        self.models = artifact["models"]
        self.weights = artifact["weights"]
        self.threshold = float(artifact.get("threshold", 0.5))

    def predict_proba(self, text: str) -> np.ndarray:
        processed = preprocess_text(text)
        probs = []
        weights = []
        for name, model in self.models.items():
            probs.append(model.predict_proba([processed])[0])
            weights.append(float(self.weights.get(name, 1.0)))
        weight_arr = np.asarray(weights, dtype=float)
        weight_arr = weight_arr / weight_arr.sum()
        return np.tensordot(weight_arr, np.stack(probs, axis=0), axes=(0, 0))

    def classify(self, text: str) -> int:
        probs = self.predict_proba(text)
        return int(probs[1] >= self.threshold)

    def _extract_lr_feature_cues(self, text: str, label: int, k: int = 8) -> List[str]:
        """Extract readable cues from the word logistic regression component."""
        model = self.models.get("word_lr")
        if model is None:
            model = self.models.get("lr_union_c8")
        if model is None:
            return []
        clf = model.named_steps.get("clf")
        features = model.named_steps.get("features")
        if features is None or clf is None or not hasattr(clf, "coef_"):
            return []

        x = features.transform([preprocess_text(text)])
        if x.nnz == 0:
            return []
        feature_names = np.asarray(features.get_feature_names_out())
        coef = clf.coef_[0]
        signed = x.data * coef[x.indices]
        target_scores = signed if label == 1 else -signed
        order = np.argsort(target_scores)[::-1]

        cues: List[str] = []
        seen = set()
        for pos in order:
            if target_scores[pos] <= 0:
                continue
            feat = str(feature_names[x.indices[pos]])
            if feat in seen:
                continue
            seen.add(feat)
            cues.append(feat)
            if len(cues) >= k:
                break
        return cues

    def explain(self, text: str) -> str:
        label = self.classify(text)
        probs = self.predict_proba(text)
        confidence = float(probs[label])
        cues = self._extract_lr_feature_cues(text, label)
        if cues:
            cue_text = "、".join([f"“{cue}”" for cue in cues])
            basis = f"主要证据包括 {cue_text} 等词或短语，这些特征在本地集成模型中更支持“{LABEL_NAMES[label]}”。"
        else:
            basis = "文本中没有明显的单个高权重词，模型主要依据整体 word/char TF-IDF 表示进行判断。"
        return (
            f"本地集成模型判断为{LABEL_NAMES[label]}（类别 {label}），置信度约为 {confidence:.3f}。"
            f"{basis} 本地模型由词级/字符级 TF-IDF 与多个线性分类器加权融合得到。"
        )

    def feature_cues(self, text: str, label: int, k: int = 8) -> List[str]:
        return self._extract_lr_feature_cues(text, label, k=k)

    def predict_with_reason(self, text: str) -> Dict[str, object]:
        label = self.classify(text)
        probs = self.predict_proba(text)
        return {
            "label": label,
            "label_name": LABEL_NAMES[label],
            "prob_non_rumor": float(probs[0]),
            "prob_rumor": float(probs[1]),
            "confidence": float(probs[label]),
            "reason": self.explain(text),
            "source": "local_ensemble",
        }


class HybridRumourDetectClass:
    """High-accuracy hybrid detector with optional LLM review for hard samples."""

    def __init__(
        self,
        model_path: str | Path = "models/ensemble.joblib",
        retriever_path: str | Path = "models/retriever.joblib",
        config_path: str | Path = "config.json",
        use_llm: bool = False,
    ) -> None:
        self.local = EnsembleRumourDetectClass(model_path)
        self.config = load_config(config_path)
        hybrid_cfg = self.config.get("hybrid", {})
        self.high_threshold = float(hybrid_cfg.get("high_confidence_threshold", 0.85))
        self.low_threshold = float(hybrid_cfg.get("low_confidence_threshold", 0.65))
        self.top_k_examples = int(hybrid_cfg.get("top_k_examples", 5))
        self.use_llm = use_llm
        self.config_path = config_path
        self.retriever = None
        self.reranker = None
        if use_llm:
            self.retriever = SimilarExampleRetriever(retriever_path)
            self.reranker = LLMReranker(config_path=config_path)

    def _needs_llm(self, confidence: float) -> bool:
        return self.use_llm and confidence < self.high_threshold

    def classify(self, text: str) -> int:
        return int(self.predict_with_reason(text)["label"])

    def explain(self, text: str) -> str:
        return str(self.predict_with_reason(text)["reason"])

    def predict_proba(self, text: str) -> np.ndarray:
        return self.local.predict_proba(text)

    def predict_with_reason(self, text: str) -> Dict[str, object]:
        local_result = self.local.predict_with_reason(text)
        local_label = int(local_result["label"])
        confidence = float(local_result["confidence"])
        cues = self.local.feature_cues(text, local_label)

        if not self._needs_llm(confidence):
            local_result["llm_used"] = False
            return local_result

        if self.retriever is None or self.reranker is None:
            local_result["llm_used"] = False
            local_result["reason"] += " 由于检索器或大模型复核器未初始化，未调用大模型。"
            return local_result

        similar_examples = self.retriever.retrieve(text, k=self.top_k_examples)
        llm_review = self.reranker.review(text, local_label, confidence, cues, similar_examples)
        llm_label = int(llm_review["label"])

        probs = self.local.predict_proba(text)
        final_label = local_label
        source = "local_ensemble"
        if confidence < self.low_threshold:
            final_label = llm_label
            source = "deepseek_reasoner_review"
        elif llm_label == local_label:
            final_label = local_label
            source = "local_ensemble_confirmed_by_deepseek"
        else:
            final_label = local_label
            source = "local_ensemble_llm_disagreed"

        final_reason = (
            f"本地集成模型预测为{LABEL_NAMES[local_label]}（类别 {local_label}），置信度 {confidence:.3f}。"
            f"大模型 DeepSeek V3.2 思考模式复核结论为{LABEL_NAMES[llm_label]}（类别 {llm_label}）。"
            f"最终采用{LABEL_NAMES[final_label]}（类别 {final_label}）。"
            f"本地重要特征：{('、'.join(cues) if cues else '无明显高权重特征')}。"
            f"大模型解释：{llm_review['reason']}"
        )
        return {
            "label": final_label,
            "label_name": LABEL_NAMES[final_label],
            "prob_non_rumor": float(probs[0]),
            "prob_rumor": float(probs[1]),
            "confidence": float(probs[final_label]) if final_label == local_label else float(probs[llm_label]),
            "local_label": local_label,
            "llm_label": llm_label,
            "llm_used": True,
            "source": source,
            "similar_examples": similar_examples,
            "reason": final_reason,
        }


RumourDetectClass = HybridRumourDetectClass
