"""
数据预处理与数据集类
- 分词器与词表构建
- PyTorch Dataset
"""

import re
import pickle
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


def clean_tweet(text: str) -> str:
    """Twitter推文文本清洗"""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """简单分词：小写 + 提取字母数字词"""
    return re.findall(r"\w+", text.lower())


class Vocab:
    """词表类"""

    def __init__(self, texts: list[str], min_freq: int = config.MIN_FREQ):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.pad_idx = self.word2idx["<PAD>"]
        self.unk_idx = self.word2idx["<UNK>"]

    def encode(self, text: str) -> list[int]:
        tokens = tokenize(text)
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        if len(ids) < config.MAX_LEN:
            ids += [self.pad_idx] * (config.MAX_LEN - len(ids))
        else:
            ids = ids[:config.MAX_LEN]
        return ids

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"Vocab(size={self.vocab_size})"


class RumorDataset(Dataset):
    """谣言检测数据集"""

    def __init__(self, df: pd.DataFrame, vocab: Vocab):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.events = df["event"].tolist() if "event" in df.columns else None
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_ids = self.vocab.encode(self.texts[idx])
        label = self.labels[idx]
        item = {"input_ids": torch.tensor(text_ids, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.float)}
        if self.events is not None:
            item["event"] = torch.tensor(self.events[idx], dtype=torch.long)
        return item


def load_data() -> tuple[DataLoader, DataLoader, Vocab]:
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    vocab = Vocab(train_df["text"].tolist(), min_freq=config.MIN_FREQ)
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, vocab
