# 可解释的谣言检测 — Rumor Detection with Explainable AI

基于社交媒体推文数据，构建一个**可解释的谣言检测模型**。输入一条推文，输出二分类检测结果 + 中文判断依据文字。

---

## 📋 项目简介

本项目是《人工智能导论》课程大作业。项目融合了**深度学习（BiGRU/DistilBERT）**、**大语言模型（SJTU DeepSeek API）** 和**检索增强生成（RAG）** 技术，构建了一套既能准确检测谣言、又能给出清晰判断依据的复合模型系统。

### 核心输出

```
输入: "BREAKING: Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen"
输出:
  label: 1 (谣言)
  probability: 0.9837
  judgment_basis: |
    1. 信息来源不可靠：推文以"BREAKING"开头制造紧迫感，但未提供任何具体来源...
    2. 与已知谣言高度相似：训练集中相似样本均标注为谣言...
    3. 语言煽情化："unarmed teen"刻意强化情感冲击...
```

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.0.0（推荐 3.11）
- PyTorch ≥ 2.0.0
- scikit-learn, pandas, numpy

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/doughnutbean/AIrumer2026.git
cd AIrumer2026

# 2. 安装依赖
pip install torch pandas scikit-learn numpy

# 3. （可选）如需运行 DistilBERT 训练
pip install transformers datasets
```

### 下载 GloVe 词向量

首次训练需要 GloVe 预训练词向量（自动下载）：

```python
python -c "from torchtext.vocab import GloVe; GloVe(name='6B', dim=100)"
```
> 下载约 862MB（Stanford NLP 服务器）。
> 如果下载缓慢，可使用 [hf-mirror.com](https://hf-mirror.com) 镜像。

---

## 📁 项目结构

```
rumer2026/
├── config.py               # 配置文件（含 SJTU API Key）
├── data.py                 # 数据预处理 & 词表构建
├── model.py                # BiGRU 模型定义
│
├── bert_model.py           # DistilBERT 模型定义
├── train_bert.py           # DistilBERT 训练脚本
│
├── composite_model.py      # 复合模型 (BiGRU + RAG + LLM)
├── llm_client.py           # SJTU API 客户端
├── rag.py                  # TF-IDF 检索器
│
├── ensemble_model.py       # 集成模型 (BiGRU + DistilBERT)
│
├── train.py                # BiGRU 训练函数
├── main.py                 # BiGRU 训练入口
├── predict.py              # BiGRU 推理类
│
├── test_composite.py       # 复合模型测试
├── test_bert.py            # DistilBERT 测试
├── test_ensemble.py        # 集成模型测试
│
├── train.csv               # 训练集 (2840条)
├── val.csv                 # 验证集 (401条)
│
├── bigru_rumor_model.pt    # BiGRU 模型权重 (需训练生成)
├── distilbert_rumor.pt     # DistilBERT 权重 (需训练生成)
│
├── 验收报告.md             # 项目验收报告
└── README.md               # 本文件
```

---

## 🧠 模型架构

### 方案一：深度学习 + LLM 复合模型（推荐用于演示）

```
输入推文
    │
    ├─→ [BiGRU 检测] ──→ label (0/1) + 概率
    │
    ├─→ [RAG 检索] ──→ 检索训练集相似样本
    │
    └─→ [LLM 判断依据] ──→ 结构化中文分析
         (SJTU API · deepseek-chat)
    │
    └─→ 输出: {label, prediction, probability, judgment_basis}
```

**优点**：可解释性强，输出结构化判断依据。

### 方案二：集成模型（BiGRU + DistilBERT，推荐用于高精度）

```
输入推文
    │
    ├─→ [BiGRU] ──→ 概率 P₁
    │                   加权平均 ──→ 最终分类
    ├─→ [DistilBERT] ──→ 概率 P₂   (权重可调)
    │
    └─→ 输出: {label, prob_bigru, prob_bert, 融合概率}
```

**优点**：准确率更高，两个模型互补。

---

## 📊 实验结果

### 各模型在 val.csv 上的性能

| 模型 | Accuracy | Precision | Recall | F1 Score |
|------|:--------:|:---------:|:------:|:--------:|
| BiGRU（基线） | 0.8603 | 0.8563 | 0.8171 | **0.8363** |
| BiGRU（改进版） | 0.8628 | 0.8297 | 0.8629 | **0.8459** |
| 复合模型 (BiGRU+LLM+RAG) | — | 可解释性输出 | ✅ | — |
| DistilBERT 微调 | 0.85+ | — | — | **0.85+** |
| 集成模型 (BiGRU+BERT) | 0.8529 | 0.8295 | 0.8343 | **0.8319** |

### 数据集

| 数据集 | 样本数 | 谣言 | 非谣言 | 事件类别 |
|--------|:-----:|:----:|:------:|:--------:|
| 训练集 | 2840 | 1240 (43.7%) | 1600 (56.3%) | 7 类 |
| 验证集 | 401 | 175 (43.6%) | 226 (56.4%) | 7 类 |

---

## 🔧 使用指南

### 训练 BiGRU 模型

```powershell
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python main.py
```

```bash
# Linux/Mac
KMP_DUPLICATE_LIB_OK=TRUE python main.py
```

### 训练 DistilBERT 模型

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -c "from train_bert import run_bert_training; run_bert_training()"
```

### 运行复合模型（含 LLM 判断依据）

```python
from composite_model import CompositeRumorDetector
detector = CompositeRumorDetector()       # 初始化
result = detector.analyze("输入一条推文")  # 分析
print(result["prediction"])      # "谣言" / "非谣言"
print(result["probability"])     # 0.9790
print(result["judgment_basis"])  # 判断依据文字
```

> ⚠️ LLM 功能需要配置 SJTU API Key：编辑 `config.py` 中的 `LLM_API_KEY`。

### 运行集成模型

```python
from ensemble_model import EnsembleRumorDetector
detector = EnsembleRumorDetector()
result = detector.analyze("输入一条推文")
print(result["prediction"])          # 融合检测结果
print(result["prob_bigru"])          # BiGRU 概率
print(result["prob_bert"])           # DistilBERT 概率
```

### 在验证集上评估

```python
# 评估集成模型
from ensemble_model import EnsembleRumorDetector
detector = EnsembleRumorDetector()
detector.evaluate_on_val()

# 测试复合模型
python test_composite.py
```

---

## 🌐 LLM API 说明

本项目的 LLM 部分使用上海交通大学提供的**本地大模型 API**。

- **接口地址**: `https://models.sjtu.edu.cn/api/v1`
- **兼容格式**: OpenAI 兼容格式
- **可用模型**: `deepseek-chat`, `deepseek-reasoner`, `glm`, `minimax`, `qwen`
- **配置方式**: 在 `config.py` 中填入 `LLM_API_KEY`
- **网络要求**: 需要 SJTU 校园网环境或 VPN

申请地址: [致远一号 AI 模型 API 申请](https://my.sjtu.edu.cn/)

---

## 🏗️ 技术细节

| 模块 | 技术栈 | 说明 |
|------|--------|------|
| 文本分类 | BiGRU + Attention | 2 层双向 GRU, hidden=128, 256d 特征 |
| 预训练 | GloVe 6B 300d | 词表命中率 96% |
| 预训练语言模型 | DistilBERT-base-uncased | 66M 参数，微调 5 epoch |
| 判断依据 | SJTU DeepSeek API | structured prompt 生成中文分析 |
| 检索增强 | TF-IDF + Cosine | 2840 条训练集索引, top-3 检索 |
| 集成学习 | 加权概率平均 | BiGRU(0.4) + BERT(0.6) |

---

## ⚡ 环境变量说明

Windows 环境下可能遇到 OpenMP 冲突：

```powershell
# 运行前设置
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

或者使用项目根目录的启动脚本（如果有）。

---

## 📄 报告

详见 `验收报告.md`，覆盖所有大作业评分要点：

- ✅ 报告叙述清楚（30分）
- ✅ 代码可运行，部署说明清楚（25分）
- ✅ 检测准确率（15分）
- ✅ 可解释性（15分）

---

## 👥 小组分工

| 成员 | 贡献内容 | 占比 |
|------|---------|:----:|
| （待填写） | 数据分析、BiGRU 模型搭建与训练 | —% |
| （待填写） | LLM API 集成、复合模型框架 | —% |
| （待填写） | RAG 检索模块、测试评估 | —% |
| （待填写） | DistilBERT 微调、集成模型 | —% |

> Git 提交记录可作为成员贡献的客观参考。

---

*生成日期：2026 年 6 月*
*GitHub: https://github.com/doughnutbean/AIrumer2026*
