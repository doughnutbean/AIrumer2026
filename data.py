"""
数据预处理与数据集类
- 分词器与词表构建
- PyTorch Dataset
- DataLoader工厂函数
"""

import re
import pickle
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


def tokenize(text: str) -> list[str]:
    """简单分词：小写 + 提取字母数字词"""
    return re.findall(r"\w+", text.lower())


class Vocab:
    """词表类：管理词到索引的映射"""

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
        """将文本编码为索引序列，长度固定为MAX_LEN"""
        tokens = tokenize(text)
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]

        # 截断或填充
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
        return f"Vocab(size={self.vocab_size}, pad={self.pad_idx}, unk={self.unk_idx})"


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

        item = {
            "input_ids": torch.tensor(text_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }

        if self.events is not None:
            item["event"] = torch.tensor(self.events[idx], dtype=torch.long)

        return item


def load_data() -> tuple[DataLoader, DataLoader, Vocab]:
    """
    加载训练集和验证集，返回 DataLoader 和词表
    """
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)

    # 构建词表
    vocab = Vocab(train_df["text"].tolist(), min_freq=config.MIN_FREQ)
    print(f"[数据] 词表大小: {vocab.vocab_size}")
    print(f"[数据] 训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")

    # 创建 Dataset
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, vocab
