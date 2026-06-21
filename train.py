"""
训练与评估函数
"""

import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import config


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        events = batch.get("event", None)

        if events is not None:
            events = events.to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, event=events)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录预测结果
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        events = batch.get("event", None)

        if events is not None:
            events = events.to(device)

        logits = model(input_ids, event=events)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, acc, precision, recall, f1, all_preds, all_labels


def run_training(model, train_loader, val_loader, device):
    """完整的训练流程"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_f1 = 0.0
    best_epoch = 0
    train_history = []

    print(f"\n{'='*50}")
    print(f"开始训练 | 设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"训练集: {len(train_loader.dataset)} 条 | 验证集: {len(val_loader.dataset)} 条")
    print(f"批次大小: {config.BATCH_SIZE} | Epochs: {config.EPOCHS}")
    print(f"{'='*50}\n")

    total_start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, preds, labels = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        })

        print(
            f"Epoch {epoch:2d}/{config.EPOCHS} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} "
            f"| {epoch_time:.1f}s"
        )

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  → 新最佳模型 (F1={best_f1:.4f}) 已保存")

    total_time = time.time() - total_start

    # 最终评估
    print(f"\n{'='*50}")
    print(f"训练完成! 总耗时: {total_time:.1f}s")
    print(f"最佳Epoch: {best_epoch} | 最佳验证F1: {best_f1:.4f}")

    # 加载最佳模型进行详细评估
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    _, val_acc, val_prec, val_rec, val_f1, preds, labels = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\n最终验证结果:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")

    print(f"\n详细分类报告:")
    print(classification_report(labels, preds, target_names=["非谣言(0)", "谣言(1)"]))

    return model, train_history
