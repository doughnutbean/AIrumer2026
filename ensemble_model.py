"""
集成模型：BiGRU + DistilBERT 联合检测
- 概率平均融合
- 支持加载两个模型各自的最优权重
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from data import Vocab, clean_tweet
from model import BiGRUModel
from bert_model import DistilBERTRumorDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import config


class EnsembleRumorDetector:
    """集成谣言检测器：BiGRU + DistilBERT"""

    def __init__(
        self,
        bigru_model_path: str = config.MODEL_SAVE_PATH,
        bert_model_path: str = "distilbert_rumor.pt",
        vocab_path: str = config.TOKENIZER_SAVE_PATH,
        device: torch.device = config.DEVICE,
    ):
        self.device = device
        self.bigru_weight = 0.4   # BiGRU 投票权重
        self.bert_weight = 0.6     # DistilBERT 投票权重（一般表现更好，权重大一些）

        print("=" * 60)
        print("[加载] 集成检测器: BiGRU + DistilBERT")
        print("=" * 60)

        # 1. 加载 BiGRU
        print("\n[1/2] 加载 BiGRU...")
        self.vocab = Vocab.load(vocab_path)
        self.bigru_model = BiGRUModel(
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.pad_idx,
        ).to(device)
        self.bigru_model.load_state_dict(
            torch.load(bigru_model_path, map_location=device, weights_only=True)
        )
        self.bigru_model.eval()
        print(f"  BiGRU 加载完成 | 词表: {len(self.vocab)}")

        # 2. 加载 DistilBERT
        print("\n[2/2] 加载 DistilBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBERTRumorDetector().to(device)
        self.bert_model.load_state_dict(
            torch.load(bert_model_path, map_location=device, weights_only=True)
        )
        self.bert_model.eval()
        print(f"  DistilBERT 加载完成 | 参数量: {sum(p.numel() for p in self.bert_model.parameters()):,}")

        print("\n" + "=" * 60)
        print(f"集成权重: BiGRU={self.bigru_weight}, BERT={self.bert_weight}")
        print("=" * 60)

    @torch.no_grad()
    def predict_bigru(self, text: str) -> float:
        """BiGRU 预测，返回谣言概率"""
        text_clean = clean_tweet(text)
        input_ids = torch.tensor(
            [self.vocab.encode(text_clean)], dtype=torch.long, device=self.device
        )
        logit = self.bigru_model(input_ids)
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def predict_bert(self, text: str) -> float:
        """DistilBERT 预测，返回谣言概率 (label=1)"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        logits = self.bert_model(encoding["input_ids"], encoding["attention_mask"])
        probs = F.softmax(logits, dim=1)
        return probs[0, 1].item()  # 类别1=谣言

    def analyze(self, text: str) -> dict:
        """融合预测"""
        prob_bigru = self.predict_bigru(text)
        prob_bert = self.predict_bert(text)

        # 加权平均
        prob_ensemble = self.bigru_weight * prob_bigru + self.bert_weight * prob_bert

        label = 1 if prob_ensemble > 0.5 else 0

        return {
            "text": text,
            "label": label,
            "prediction": "谣言" if label == 1 else "非谣言",
            "probability": round(prob_ensemble, 4),
            "prob_bigru": round(prob_bigru, 4),
            "prob_bert": round(prob_bert, 4),
            "judgment_basis": (
                f"[集成检测] BiGRU概率={prob_bigru:.2%}, BERT概率={prob_bert:.2%}, "
                f"融合概率={prob_ensemble:.2%} → 判定为{'谣言' if label==1 else '非谣言'}"
            ),
        }

    def evaluate_on_val(self) -> dict:
        """在验证集上评估集成模型"""
        import pandas as pd
        df = pd.read_csv(config.VAL_PATH)
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        print(f"\n在验证集上评估集成模型 ({len(texts)} 条)...")
        preds = []
        bigru_probs = []
        bert_probs = []

        for i, text in enumerate(texts):
            pb = self.predict_bigru(text)
            pt = self.predict_bert(text)
            bigru_probs.append(pb)
            bert_probs.append(pt)
            prob = self.bigru_weight * pb + self.bert_weight * pt
            preds.append(1 if prob > 0.5 else 0)

            if (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{len(texts)}")

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        print(f"\n{'='*60}")
        print("集成模型 - 验证集结果")
        print(f"{'='*60}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\n{classification_report(labels, preds, target_names=['非谣言(0)', '谣言(1)'])}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


if __name__ == "__main__":
    detector = EnsembleRumorDetector()

    test_texts = [
        "Swiss museum confirms it will take on Gurlitt collection",
        "BREAKING: Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen",
        "Just had a great lunch today with my friends",
    ]

    for text in test_texts:
        result = detector.analyze(text)
        print(f"\n文本: {text[:55]}...")
        print(f"  BiGRU: {result['prob_bigru']:.4f} | BERT: {result['prob_bert']:.4f}")
        print(f"  融合: {result['prediction']} (概率={result['probability']:.4f})")
