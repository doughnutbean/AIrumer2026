# AIRumer2026

> **人工智能导论** · 课程大作业  
> 基于 Transformer 主干与大模型选择性复核的可解释谣言检测系统

---

## 📖 项目简介

**AIrumer2026** 是一个面向社交媒体短文本的 **可解释谣言检测系统**，针对二分类任务（0 = 非谣言，1 = 谣言）。项目采用"强主干 + 选择性复核"的两阶段架构：

- **Transformer 主干模型** 提供稳定的基础分类，准确率约 **88.78%**
- **FN-aware Selective Correction** 针对主干容易漏检的样本进行风险排序
- **DeepSeek Reasoner** 作为保守型复核器，只对高风险候选样本进行 DeepSeek 复核
- **规则门控**（高相似度谣言例外 + 客观新闻保护）防止误翻转

最终系统在验证集上达到 **89.03% 的准确率**，Oracle 上界显示理论空间超过 **90%**。

### 核心设计原则

```
宁可少翻转，也不随意增加误报。
```

---

## 🏗️ 系统架构

```
输入推文文本
    ↓
文本预处理（URL 归一化 / HTML 解码 / 空白清洗）
    ↓
本地集成模型预测（TF-IDF + 加权线性分类器融合）
    ↓
┌─ 主干判为 1（谣言）→ 直接保留 ──────────────────┐
├─ 主干判为 0（非谣言）→ 进入 FN-risk 评分排序 ───┤
    ↓
选取 Top-K 高风险候选
    ↓
TF-IDF 检索相似训练样本（Few-shot 证据）
    ↓
DeepSeek Reasoner 保守复核
    ↓
规则门控：
  · 高置信谣言门控（confidence ≥ 0.80）
  · 高相似度谣言例外（top1_similarity ≥ 0.35）
  · 客观新闻保护（objective-report 模式拦截）
    ↓
输出最终标签与可解释信息
```

### 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 文本预处理 | `src/text_utils.py` | URL 归一化、HTML entity 解码、空白清洗 |
| 相似样本检索 | `src/retriever.py` | TF-IDF 向量化 + 余弦相似度，检索 Top-K 近邻 |
| 本地集成模型 | `src/hybrid_classifier.py` | 词级/字符级 TF-IDF + 多个线性分类器加权融合 |
| DeepSeek API 客户端 | `src/llm_client.py` | 封装 SJTU OpenAI-compatible API（含重试和限流） |
| LLM 复核器 | `src/llm_reranker.py` | 保守型 Prompt + 结构化 JSON 输出解析 |
| 单条预测入口 | `src/predict.py` | CLI 命令行单条文本预测 |
| FN-aware 批量评测 | `src/fn_aware_selective_correction.py` | 风险排序 + LLM 复核 + 规则门控 + 指标输出 |

---

## 📊 实验结果

### Transformer 主干（基线）

| 指标 | 数值 |
|------|------|
| 准确率 | 88.78% |
| 混淆矩阵 | [[216, 10], [35, 140]] |
| TN / FP / FN / TP | 216 / 10 / 35 / 140 |

### Oracle 上界模拟（理论空间）

| 指标 | 数值 |
|------|------|
| 准确率 | **90.27%** |
| 混淆矩阵 | [[216, 10], [29, 146]] |
| 恢复 FN | 6 |

### 最终方案（保守 DeepSeek + 规则门控）

| 指标 | 数值 |
|------|------|
| 准确率 | **89.03%** |
| 混淆矩阵 | [[216, 10], [34, 141]] |
| 恢复 FN | 1 |
| 新增 FP | 0 |

### 分类报告（最终方案）

| 类别 | 精确率 | 召回率 | F1-score | 支持数 |
|------|-------|-------|---------|-------|
| 非谣言 (0) | 0.864 | 0.956 | 0.908 | 226 |
| 谣言 (1) | 0.934 | 0.806 | 0.865 | 175 |
| **整体** | **0.894** | **0.890** | **0.889** | **401** |

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- Windows / Linux / macOS

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd AIrumor2026
```

### 2. 创建虚拟环境

**Windows PowerShell：**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS：**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
cd rumor_detector_project
pip install -r requirements.txt
```

### 4. 配置 API Key

编辑 `config.json` 或在环境变量中设置：

**方式一：直接编辑 config.json**
```json
{
  "api": {
    "base_url": "https://models.sjtu.edu.cn/api/v1",
    "api_key": "your-api-key-here"
  }
}
```

**方式二：环境变量（推荐，避免泄露）**
```bash
# Windows PowerShell
$env:SJTU_API_KEY="your-api-key-here"

# Linux / macOS
export SJTU_API_KEY="your-api-key-here"
```

### 5. 准备数据和模型

项目已包含以下文件：

| 文件 | 说明 |
|------|------|
| `dataset/train.csv` | 训练数据集（2840 条） |
| `dataset/val.csv` | 验证数据集（401 条） |
| `models/ensemble.joblib` | 本地集成模型 |
| `models/retriever.joblib` | TF-IDF 相似样本检索器 |

如需重新构建检索器：

```bash
python src/retriever.py --train dataset/train.csv --out models/retriever.joblib
```

---

## 💻 使用指南

### 单条文本预测（本地模型）

```bash
python src/predict.py --text "BREAKING: police confirmed the story after multiple reports."
```

输出示例：

```json
{
  "label": 0,
  "label_name": "非谣言",
  "prob_non_rumor": 0.73,
  "prob_rumor": 0.27,
  "confidence": 0.73,
  "reason": "本地集成模型判断为非谣言...",
  "source": "local_ensemble",
  "llm_used": false
}
```

### 单条文本预测（启用 DeepSeek 复核）

```bash
python src/predict.py --use-llm --text "The government is hiding the truth about the virus, this is a massive cover-up!"
```

启用 LLM 后，系统会：
1. 先运行本地模型获取初始预测
2. 对低置信样本检索相似训练样本
3. 调用 DeepSeek Reasoner 复核
4. 返回最终标签 + 可解释信息（含证据类型、翻转状态、LLM 理由）

### 批量评测（Oracle 上界模拟）

```bash
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55
```

### 批量评测（真实 DeepSeek 复核）

```bash
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

#### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|-------|
| `--use-llm` | 使用 DeepSeek 复核 | — |
| `--simulate-oracle` | Oracle 上界模拟（用真实标签代替 LLM） | — |
| `--top-k` | 复核风险最高的 K 个候选样本 | 15 |
| `--review-threshold` | FN-risk 最低复核阈值 | 0.55 |
| `--llm-confidence-min` | LLM 允许翻转的最低置信度 | 0.80 |
| `--min-top1-similarity` | 高相似度例外的最小 Top-1 相似度 | 0.35 |
| `--min-knn5-rumor-ratio` | Top-5 近邻的最小谣言比例 | 0.60 |
| `--strict-objective-protection` | 启用客观新闻保护 | — |
| `--llm-sleep-seconds` | LLM 调用间隔（避免限流） | 7.0 |

---

## 📁 目录结构

```
AIrumor2026/
├── README.md                         # 项目总览（本文件）
├── .codewhale/                       # CodeWhale 配置
│   └── instructions.md
└── rumor_detector_project/
    ├── README.md                     # 子项目详细文档（770 行）
    ├── config.json                   # API 配置文件
    ├── requirements.txt              # Python 依赖
    ├── dataset/
    │   ├── train.csv                 # 训练集（2840 条）
    │   └── val.csv                   # 验证集（401 条）
    ├── models/
    │   ├── ensemble.joblib           # 本地集成模型
    │   └── retriever.joblib          # TF-IDF 检索器
    ├── outputs/
    │   ├── final_fusion_with_transformer/   # 主干模型结果
    │   │   ├── metrics.json
    │   │   └── predictions.jsonl
    │   ├── fn_aware_selective_correction/   # 选择性修正结果
    │   │   ├── metrics.json
    │   │   ├── predictions.jsonl
    │   │   ├── candidate_ranking.jsonl
    │   │   ├── reviewed_candidates.jsonl
    │   │   └── classification_report.txt
    │   ├── confusion_matrix.png      # 混淆矩阵可视化
    │   └── event_accuracy.png        # 事件准确率可视化
    └── src/
        ├── config_utils.py           # 配置加载（支持环境变量覆写）
        ├── text_utils.py             # 文本预处理
        ├── retriever.py              # TF-IDF 相似样本检索
        ├── hybrid_classifier.py      # 混合分类器（本地 + 可选 LLM）
        ├── llm_client.py             # DeepSeek API 客户端
        ├── llm_reranker.py           # LLM 复核器（Prompt + 解析）
        ├── fn_aware_selective_correction.py  # FN-aware 批量评测
        └── predict.py                # CLI 单条预测入口
```

---

## 🧠 技术细节

### FN-aware Risk Score

对主干判为 0 的样本，综合以下信号计算风险分数：

| 信号 | 权重贡献 |
|------|---------|
| Transformer 对谣言类的概率 | 0.3 ~ 0.8 |
| 专用模型对谣言类的概率 | 0.25 ~ 0.9 |
| Top-3 近邻中谣言比例 | 0.65 ~ 1.0 |
| Top-5 近邻中谣言比例 | 0.35 ~ 0.55 |
| Top-1 近邻为谣言且相似度高 | 0.35 ~ 1.0 |
| 本地模型对谣言类的概率 | 0.2 ~ 0.35 |

### DeepSeek 复核 Prompt 设计

复核器采用**保守型二阶段 Prompt**：

1. 系统角色设定为"保守型谣言复核器"
2. 明确要求：**只有文本包含明确未证实断言、阴谋指控或煽动性话术时**才判为谣言
3. 客观新闻报道、官方声明、现场描述优先保持非谣言
4. 输出必须为结构化 JSON，包含 `label`, `confidence`, `evidence_type`, `should_flip`, `reason`

允许的 evidence_type：

| 类型 | 含义 | 是否支持翻转 |
|------|------|------------|
| `objective_report` | 客观新闻报道 | ❌ |
| `unverified_claim` | 未证实的断言 | ✅ |
| `conspiracy_accusation` | 阴谋指控 | ✅ |
| `sensational_claim` | 夸张传播 | ✅ |
| `correction_or_denial` | 澄清或否认 | ❌ |
| `unclear` | 证据不足 | ❌ |

### 翻转门控

要完成 0 → 1 翻转，必须同时满足：
- DeepSeek 判为谣言（label=1）且建议翻转（should_flip=true）
- 置信度 ≥ 0.80
- 证据类型为 `unverified_claim` / `conspiracy_accusation` / `sensational_claim`
- 满足高相似度谣言例外或强谣言语义条件
- 未触发客观新闻保护

---

## 📈 输出文件说明

运行批量评测后，`outputs/fn_aware_selective_correction/` 下生成：

| 文件 | 说明 |
|------|------|
| `metrics.json` | 准确率、混淆矩阵、分类报告、翻转统计 |
| `predictions.jsonl` | 每条样本的最终预测、复核结果和解释字段 |
| `candidate_ranking.jsonl` | 所有主干判 0 样本的 FN-risk 排序 |
| `reviewed_candidates.jsonl` | 实际送入 DeepSeek 复核的候选样本 |
| `classification_report.txt` | sklearn 文本版分类报告 |

---

## ⚠️ 已知局限

1. 真实 DeepSeek 复核效果受 API 稳定性和 Prompt 质量影响
2. 高相似度阈值和客观新闻保护规则需根据数据集校准
3. 仓库当前包含最终方案和评测流程，不包含完整训练流水线
4. 若换到新事件、新平台或新语言，需重新评估 FN-risk 分布

---

## 📚 附录

### 实验复现步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建检索器（如 models/retriever.joblib 不存在）
python src/retriever.py --train dataset/train.csv --out models/retriever.joblib

# 3. 运行 Oracle 上界模拟
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55

# 4. 运行真实 DeepSeek 复核评测
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

### 子项目文档

更详细的技术说明、Motivation 分析和实验过程请参阅：
[`rumor_detector_project/README.md`](rumor_detector_project/README.md)（770 行）

---

## 🧑‍🎓 致谢

《人工智能导论》课程项目 — 感谢课程组和 SJTU 模型 API 平台提供的 DeepSeek 服务。
