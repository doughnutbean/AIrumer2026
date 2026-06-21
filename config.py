"""
配置文件 - 所有超参数集中管理
"""

import torch

# ===== 数据路径 =====
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"

# ===== 文本处理 =====
MAX_LEN = 64            # 最大序列长度（参考文档推荐值）
MIN_FREQ = 2            # 最小词频（低于此频率的词视为<UNK>）

# ===== 模型参数 =====
EMBEDDING_DIM = 100     # 词嵌入维度
HIDDEN_DIM = 128        # GRU隐藏层维度
NUM_LAYERS = 2          # GRU层数
DROPOUT = 0.3           # Dropout率
USE_EVENT_EMB = True    # 是否使用事件类别嵌入
EVENT_EMB_DIM = 16      # 事件嵌入维度
NUM_EVENTS = 7          # 事件类别数（0-6）
USE_ATTENTION = True    # 是否使用注意力机制

# ===== 训练参数 =====
BATCH_SIZE = 32         # 批大小
EPOCHS = 15             # 训练轮数
LEARNING_RATE = 1e-3    # 学习率
WEIGHT_DECAY = 1e-5     # 权重衰减（L2正则化）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== 保存路径 =====
MODEL_SAVE_PATH = "bigru_rumor_model.pt"
TOKENIZER_SAVE_PATH = "vocab.pkl"
