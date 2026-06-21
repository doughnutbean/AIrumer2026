"""
DistilBERT 训练与评估
"""

import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import config
from bert_model import DistilBERTRumorDetector


class BERTRumorDataset(Dataset):
    """BERT 格式的数据集"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_bert_data(tokenizer, batch_size=16):
    """加载 BERT 格式的数据"""
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)

    train_set = BERTRumorDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer
    )
    val_set = BERTRumorDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print(f"[BERT] 训练集: {len(train_set)} 条, 验证集: {len(val_set)} 条")
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集评估"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(loader), acc, precision, recall, f1, all_preds, all_labels


def run_bert_training(seed=42, save_path="distilbert_rumor.pt"):
    """完整的 DistilBERT 训练流程"""
    print("=" * 60)
    print(f"DistilBERT 谣言检测训练 (seed={seed})")
    print("=" * 60)

    device = config.DEVICE
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 设置随机种子
    torch.manual_seed(seed)

    # 加载数据
    print("\n[1/3] 加载数据...")
    train_loader, val_loader = load_bert_data(tokenizer)

    # 创建模型
    print("\n[2/3] 创建模型...")
    model = DistilBERTRumorDetector().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total_params:,} | 可训练: {trainable:,}")

    # 优化器（BERT 适合用稍小的学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 训练
    print("\n[3/3] 开始训练...")
    print(f"{'='*60}")
    print(f"设备: {device} | 种子: {seed} | Epochs: 5")
    print(f"{'='*60}")

    best_f1 = 0
    total_start = time.time()

    for epoch in range(1, 6):
        start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, preds, labels = evaluate(
            model, val_loader, device
        )
        elapsed = time.time() - start

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            marker = " ★"

        print(
            f"Epoch {epoch}/5 | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} "
            f"| {elapsed:.0f}s{marker}"
        )

    total_time = time.time() - total_start

    # 最终评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, val_acc, val_prec, val_rec, val_f1, preds, labels = evaluate(
        model, val_loader, device
    )

    print(f"\n{'='*60}")
    print(f"训练完成! 总耗时: {total_time:.0f}s")
    print(f"最佳验证 F1: {best_f1:.4f}")
    print(f"\n最终验证集结果:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['非谣言(0)', '谣言(1)'])}")

    return model, save_path
