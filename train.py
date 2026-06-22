import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataset import load_data
from model import RumorDetector

def __train__(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()     # 反向传播
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def main():
    MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 2e-5    # 学习率

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    # 加载数据
    train_loader, val_loader, tokenizer = load_data(
        'data/train.csv', 'data/val.csv',
        model_name=MODEL_NAME, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )

    model = RumorDetector(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        train_loss, train_acc = __train__(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            tokenizer.save_pretrained('models/tokenizer')
            print(f"最佳模型 (Val Acc: {val_acc:.4f})")

if __name__ == '__main__':
    main()
