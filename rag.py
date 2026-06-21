"""
RAG 检索模块
- 从训练集中检索与输入文本最相似的样本
- 用 TF-IDF + 余弦相似度进行检索
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


class RAGRetriever:
    """基于 TF-IDF 的 RAG 检索器"""

    def __init__(self, train_path: str = config.TRAIN_PATH):
        self.df = pd.read_csv(train_path)
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.labels_dict = {0: "非谣言", 1: "谣言"}

        # 预处理文本
        clean_texts = [
            self._clean(t) for t in self.texts
        ]

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5000, ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(clean_texts)
        print(f"[RAG] TF-IDF 向量化完成: {self.tfidf_matrix.shape}")

    def _clean(self, text: str) -> str:
        import re
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
        """检索与查询最相似的训练样本"""
        query_clean = self._clean(query)
        query_vec = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "label": self.labels_dict[self.labels[idx]],
                "similarity": float(similarities[idx]),
            })
        return results

    def format_retrieved(self, query: str, top_k: int = None) -> str:
        """格式化成可嵌入 prompt 的文本"""
        if top_k is None:
            top_k = config.RAG_TOP_K
        results = self.retrieve(query, top_k)
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r['label']}] {r['text'][:120]}... (相似度: {r['similarity']:.2f})")
        return "\n".join(lines)


# 全局单例
_retriever = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
