"""
DistilBERT 谣言检测模型
- 使用 transformers 库加载预训练 DistilBERT
- 微调二分类
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import config


class DistilBERTRumorDetector(nn.Module):
    """基于 DistilBERT 的谣言检测模型"""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits  # (batch, num_labels)

    def predict_proba(self, input_ids, attention_mask=None):
        """返回概率 [batch, 2]"""
        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        return probs


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """获取 DistilBERT 分词器"""
    return AutoTokenizer.from_pretrained(model_name)
