"""
推理类 - 加载训练好的模型进行谣言检测
"""

import re
import torch
import config
from data import Vocab
from model import BiGRUModel


class RumorDetector:
    """谣言检测器"""

    def __init__(
        self,
        model_path: str = config.MODEL_SAVE_PATH,
        vocab_path: str = config.TOKENIZER_SAVE_PATH,
        device: torch.device = config.DEVICE,
    ):
        self.device = device

        # 加载词表
        self.vocab = Vocab.load(vocab_path)
        print(f"[加载] 词表大小: {len(self.vocab)}")

        # 加载模型
        self.model = BiGRUModel(
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.pad_idx,
        ).to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[加载] 模型已加载: {model_path}")

    def preprocess(self, text: str) -> str:
        """文本预处理"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def classify(self, text: str, event: int = None) -> dict:
        """
        对单条文本进行谣言检测

        Args:
            text: 输入文本
            event: 事件类别（可选，0-6）

        Returns:
            dict: {"label": 0或1, "probability": float, "prediction": str}
        """
        # 预处理
        text_clean = self.preprocess(text)

        # 编码
        input_ids = torch.tensor(
            [self.vocab.encode(text_clean)], dtype=torch.long, device=self.device
        )

        # 事件编码
        event_tensor = None
        if event is not None:
            event_tensor = torch.tensor([event], dtype=torch.long, device=self.device)

        # 推理
        with torch.no_grad():
            logit = self.model(input_ids, event=event_tensor)
            prob = torch.sigmoid(logit).item()

        label = 1 if prob > 0.5 else 0
        prediction = "谣言" if label == 1 else "非谣言"

        return {
            "label": label,
            "probability": round(prob, 4),
            "prediction": prediction,
        }

    def classify_batch(self, texts: list[str], events: list[int] = None) -> list[dict]:
        """批量预测"""
        results = []
        for i, text in enumerate(texts):
            event = events[i] if events else None
            results.append(self.classify(text, event))
        return results


# 使用示例
if __name__ == "__main__":
    detector = RumorDetector()

    # 测试样本
    test_texts = [
        "Swiss museum confirms it will take on Gurlitt collection",
        "BREAKING: Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen",
    ]

    for text in test_texts:
        result = detector.classify(text)
        print(f"\n文本: {text[:60]}...")
        print(f"预测: {result['prediction']} (概率: {result['probability']:.4f})")
