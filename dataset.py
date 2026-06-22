import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class RumorDataset(Dataset):
    """谣言检测数据集"""
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path)
        self.texts = self.data['text'].astype(str).tolist()
        if 'label' in self.data.columns:
            self.labels = self.data['label'].tolist()
        else:
            self.labels = None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_data(train_path, val_path, model_name='bert-base-uncased', max_len=128, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if train_path:
        train_dataset = RumorDataset(train_path, tokenizer, max_len)
    else:
        train_dataset = None
    val_dataset = RumorDataset(val_path, tokenizer, max_len)
    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
