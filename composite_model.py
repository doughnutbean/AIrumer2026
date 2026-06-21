"""
复合谣言检测模型
- BiGRU给出检测结果
- LLM给出判断依据
- RAG检索相似样本增强LLM判断
"""

import sys
import torch
import config
from data import Vocab, clean_tweet
from model import BiGRUModel
from llm_client import SJTU_LLM_Client
from rag import get_retriever, RAGRetriever


class CompositeRumorDetector:
    """复合谣言检测器：深度模型 + 大语言模型"""

    def __init__(
        self,
        model_path: str = config.MODEL_SAVE_PATH,
        vocab_path: str = config.TOKENIZER_SAVE_PATH,
        llm_api_key: str = None,
        device: torch.device = config.DEVICE,
    ):
        self.device = device

        # 1. 加载深度学习检测器
        print("=" * 50)
        print("[加载] 复合谣言检测器")
        print("=" * 50)

        self.vocab = Vocab.load(vocab_path)
        print(f"  [1/3] 词表: {len(self.vocab)} 词")

        self.model = BiGRUModel(
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.pad_idx,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"  [2/3] BiGRU检测器已加载")

        # 2. 初始化 LLM 客户端
        api_key = llm_api_key or config.LLM_API_KEY
        self.use_llm = bool(api_key)
        if self.use_llm:
            self.llm = SJTU_LLM_Client(api_key=api_key)
            print(f"  [3/3] LLM客户端已初始化 (模型: {self.llm.model})")
        else:
            self.llm = None
            print(f"  [3/3] LLM未配置 (缺少API_KEY)")

        # 3. RAG 检索器
        self.use_rag = config.USE_RAG
        if self.use_rag:
            try:
                self.retriever = get_retriever()
            except Exception as e:
                print(f"  [警告] RAG加载失败: {e}，已禁用")
                self.use_rag = False

        print()

    def detect(self, text: str, event: int = None) -> dict:
        """BiGRU 谣言检测"""
        text_clean = clean_tweet(text)
        input_ids = torch.tensor(
            [self.vocab.encode(text_clean)], dtype=torch.long, device=self.device
        )
        event_tensor = None
        if event is not None:
            event_tensor = torch.tensor([event], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logit = self.model(input_ids, event=event_tensor)
            prob = torch.sigmoid(logit).item()

        label = 1 if prob > 0.5 else 0
        return {
            "label": label,
            "probability": round(prob, 4),
            "confidence": round(max(prob, 1 - prob), 4),  # 与0.5的距离即置信度
        }

    def judge(self, text: str, detection: dict) -> str:
        """调用 LLM 生成判断依据"""
        if not self.use_llm:
            return (
                f"[模型判断] 该推文被检测为{'谣言' if detection['label'] == 1 else '非谣言'}，"
                f"置信度 {detection['confidence']:.2%}。"
                f"(LLM未配置，无法生成详细依据)"
            )

        # RAG 检索相似样本
        similar = ""
        if self.use_rag:
            try:
                similar = self.retriever.format_retrieved(text)
            except Exception:
                pass

        return self.llm.generate_judgment(
            text=text,
            label=detection["label"],
            confidence=detection["confidence"],
            similar_cases=similar,
        )

    def analyze(self, text: str, event: int = None) -> dict:
        """完整分析：检测 + 判断依据"""
        # 第一步：深度学习检测
        detection = self.detect(text, event)

        # 第二步：LLM生成判断依据
        judgment = self.judge(text, detection)

        return {
            "text": text,
            "label": detection["label"],
            "prediction": "谣言" if detection["label"] == 1 else "非谣言",
            "probability": detection["probability"],
            "confidence": detection["confidence"],
            "judgment_basis": judgment,
        }

    def analyze_batch(self, texts: list[str], events: list[int] = None) -> list[dict]:
        """批量分析"""
        results = []
        for i, text in enumerate(texts):
            event = events[i] if events else None
            results.append(self.analyze(text, event))
        return results
