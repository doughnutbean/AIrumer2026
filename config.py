"""
配置文件 - 所有超参数集中管理
"""

import torch

# ===== 数据路径 =====
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"

# ===== 文本处理 =====
MAX_LEN = 64
MIN_FREQ = 2

# ===== 模型参数 =====
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
USE_EVENT_EMB = True
EVENT_EMB_DIM = 16
NUM_EVENTS = 7
USE_ATTENTION = True

# ===== 训练参数 =====
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== SJTU LLM API =====
LLM_BASE_URL = "https://models.sjtu.edu.cn/api/v1"
LLM_API_KEY = "sk-xnTgrXSSCjzejfGOURYa3g"
LLM_MODEL = "deepseek-chat"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3

# ===== RAG检索 =====
USE_RAG = True
RAG_TOP_K = 3

# ===== 保存路径 =====
MODEL_SAVE_PATH = "bigru_rumor_model.pt"
TOKENIZER_SAVE_PATH = "vocab.pkl"
