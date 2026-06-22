# AIrumer2026

> **人工智能导论** · 课程大作业  
> 基于 Transformer 主干与大模型选择性复核的可解释谣言检测系统

---

## 📖 项目简介

**AIrumer2026** 是一个面向社交媒体短文本的 **可解释谣言检测系统**，针对二分类任务（0 = 非谣言，1 = 谣言）。项目采用“强主干 + 选择性复核”的两阶段架构：

- **Transformer / 本地主干模型** 提供稳定的基础分类，保存主干结果准确率约 **88.78%**
- **FN-aware Selective Correction** 针对主干容易漏检的样本进行风险排序
- **DeepSeek Reasoner** 作为保守型复核器，只对高风险候选样本进行 DeepSeek 复核
- **规则门控**（高相似度谣言例外 + 客观新闻保护）防止误翻转

最终系统在验证集上达到 **89.03% 的准确率**，Oracle 上界模拟结果为 **90.02%**。

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
可选 Transformer voter 概率输出
    ↓
TF-IDF 检索相似训练样本（Few-shot 证据）
    ↓
生成 final-fusion prediction JSONL
    ↓
┌─ 主干判为 1（谣言）→ 直接保留 ──────────────────┐
├─ 主干判为 0（非谣言）→ 进入 FN-risk 评分排序 ───┤
    ↓
选取 Top-K 高风险候选
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

### 关键模块说明

| 模块                | 文件                                          | 职责                                                         |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| 文本预处理          | `src/text_utils.py`                           | URL 归一化、HTML entity 解码、空白清洗                       |
| 相似样本检索        | `src/retriever.py`                            | TF-IDF 向量化 + 余弦相似度，检索 Top-K 近邻                  |
| 本地集成模型        | `src/hybrid_classifier.py`                    | 加载原始 `ensemble.joblib`，并兼容新增 `dedicated_local.joblib` 模型格式 |
| 本地训练模型定义    | `src/dedicated_model.py`                      | 定义多视角 TF-IDF 特征和加权概率集成器                       |
| 本地模型训练        | `src/train_dedicated_local.py`                | 从 `train.csv` 训练 TF-IDF + LR/SGD/NB 本地集成模型          |
| Transformer 训练    | `src/train_transformer_voter.py`              | 微调 Transformer voter，输出多个 seed 的模型清单             |
| Transformer 推理    | `src/predict_transformer_voter.py`            | 使用已训练 Transformer voter 生成验证集概率                  |
| Transformer 导出    | `src/export_transformer_voter_predictions.py` | 导出 `transformer_prob_1` 和 `transformer_label` JSONL，默认不覆盖 final-fusion 文件 |
| Final-fusion 构建   | `src/build_final_fusion_predictions.py`       | 融合本地概率、Transformer 概率和检索特征，生成 `final_fusion_with_transformer/predictions.jsonl` |
| DeepSeek API 客户端 | `src/llm_client.py`                           | 封装 SJTU OpenAI-compatible API（含重试和限流）              |
| LLM 复核器          | `src/llm_reranker.py`                         | 保守型 Prompt + 结构化 JSON 输出解析                         |
| 单条预测入口        | `src/predict.py`                              | CLI 命令行单条文本预测                                       |
| FN-aware 批量评测   | `src/fn_aware_selective_correction.py`        | 风险排序 + LLM 复核 + 规则门控 + 指标输出；兼容 `base_label` 与 `transformer_label` 两类输入字段 |

---

## 📊 实验结果

### 保存主干结果（final-fusion base）

| 指标              | 数值                   |
| ----------------- | ---------------------- |
| 准确率            | 88.78%                 |
| 混淆矩阵          | [[216, 10], [35, 140]] |
| TN / FP / FN / TP | 216 / 10 / 35 / 140    |

### Oracle 上界模拟（理论空间）

| 指标     | 数值                   |
| -------- | ---------------------- |
| 准确率   | **90.02%**             |
| 混淆矩阵 | [[216, 10], [30, 145]] |
| 恢复 FN  | 5                      |

### 最终方案（保守 DeepSeek + 规则门控）

| 指标     | 数值                   |
| -------- | ---------------------- |
| 准确率   | **89.03%**             |
| 混淆矩阵 | [[216, 10], [34, 141]] |
| 恢复 FN  | 1                      |
| 新增 FP  | 0                      |

### 分类报告（最终方案）

| 类别       | 精确率    | 召回率    | F1-score  | 支持数  |
| ---------- | --------- | --------- | --------- | ------- |
| 非谣言 (0) | 0.864     | 0.956     | 0.908     | 226     |
| 谣言 (1)   | 0.934     | 0.806     | 0.865     | 175     |
| **整体**   | **0.894** | **0.890** | **0.889** | **401** |

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- Windows / Linux / macOS

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd AIrumer2026-main/rumor_detector_project
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
pip install -r requirements.txt
```

如需重新训练 Transformer voter，需额外安装 Transformer 训练依赖：

```bash
pip install torch transformers datasets accelerate sentencepiece emoji
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

| 文件                                                      | 说明                             |
| --------------------------------------------------------- | -------------------------------- |
| `dataset/train.csv`                                       | 训练数据集（2840 条）            |
| `dataset/val.csv`                                         | 验证数据集（401 条）             |
| `models/ensemble.joblib`                                  | 本地集成模型                     |
| `models/retriever.joblib`                                 | TF-IDF 相似样本检索器            |
| `outputs/final_fusion_with_transformer/predictions.jsonl` | 保存的 final-fusion 主干预测结果 |
| `outputs/fn_aware_selective_correction/predictions.jsonl` | 保存的最终 DeepSeek 复核结果     |

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
  "label": 1,
  "label_name": "谣言",
  "prob_non_rumor": 0.1551,
  "prob_rumor": 0.8449,
  "confidence": 0.8449,
  "reason": "本地集成模型判断为谣言...",
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

| 参数                            | 含义                                  | 默认值 |
| ------------------------------- | ------------------------------------- | ------ |
| `--use-llm`                     | 使用 DeepSeek 复核                    | —      |
| `--simulate-oracle`             | Oracle 上界模拟（用真实标签代替 LLM） | —      |
| `--top-k`                       | 复核风险最高的 K 个候选样本           | 15     |
| `--review-threshold`            | FN-risk 最低复核阈值                  | 0.55   |
| `--llm-confidence-min`          | LLM 允许翻转的最低置信度              | 0.80   |
| `--min-top1-similarity`         | 高相似度例外的最小 Top-1 相似度       | 0.35   |
| `--min-knn5-rumor-ratio`        | Top-5 近邻的最小谣言比例              | 0.60   |
| `--strict-objective-protection` | 启用客观新闻保护                      | —      |
| `--llm-sleep-seconds`           | LLM 调用间隔（避免限流）              | 7.0    |

---

## 🔁 从头训练与生成 final-fusion JSONL

从头训练流程用于复现实验链路和生成新的中间文件。由于训练随机性、Transformer 版本、硬件环境、API 状态和 LLM 响应解析差异，重新训练或重新调用 LLM 可能导致最终结果与保存的 **89.03%** 存在一定出入。提交复核时建议以仓库中原始保存的 `outputs/final_fusion_with_transformer/predictions.jsonl` 和 `outputs/fn_aware_selective_correction/predictions.jsonl` 为依据。

### 1. 训练本地 dedicated 模型（可选）

快速复现模式：

```bash
python src/train_dedicated_local.py \
  --train dataset/train.csv \
  --val dataset/val.csv \
  --model models/dedicated_local.joblib \
  --out outputs/dedicated_local_metrics.json \
  --quick
```

完整候选搜索模式：

```bash
python src/train_dedicated_local.py \
  --train dataset/train.csv \
  --val dataset/val.csv \
  --model models/dedicated_local.joblib \
  --out outputs/dedicated_local_metrics.json
```

### 2. 重建检索器

```bash
python src/retriever.py \
  --train dataset/train.csv \
  --out models/retriever.joblib
```

### 3. 训练并导出 Transformer voter（可选）

```bash
python src/train_transformer_voter.py \
  --train dataset/train.csv \
  --val dataset/val.csv \
  --model-name vinai/bertweet-base \
  --out-dir models/transformer_voter \
  --seeds 42,43,44
```

```bash
python src/export_transformer_voter_predictions.py \
  --input dataset/val.csv \
  --manifest models/transformer_voter/manifest.json \
  --out outputs/transformer_voter_val.jsonl
```

### 4. 生成 final-fusion prediction JSONL

使用已保存本地模型、检索器和可选 Transformer 输出生成 FN-aware 所需的中间文件：

```bash
python src/build_final_fusion_predictions.py \
  --input dataset/val.csv \
  --model models/ensemble.joblib \
  --dedicated-model models/dedicated_local.joblib \
  --retriever models/retriever.joblib \
  --transformer-predictions outputs/transformer_voter_val.jsonl \
  --out outputs/final_fusion_with_transformer/predictions.jsonl \
  --metrics outputs/final_fusion_with_transformer/metrics.json
```

若未训练 dedicated 模型或 Transformer voter，可省略相应参数；脚本会使用本地 `ensemble.joblib` 概率作为回退值。重新生成的 final-fusion 结果可能与仓库保存的主干结果不同，因此报告中的最终 89.03% 以仓库保存的 final-fusion 和最终复核输出为准。

### 5. 运行 FN-aware 选择性修正

```bash
python src/fn_aware_selective_correction.py \
  --predictions outputs/final_fusion_with_transformer/predictions.jsonl \
  --top-k 20 \
  --review-threshold 0.55
```

如需真实调用 DeepSeek：

```bash
python src/fn_aware_selective_correction.py \
  --predictions outputs/final_fusion_with_transformer/predictions.jsonl \
  --use-llm \
  --top-k 20 \
  --review-threshold 0.55 \
  --llm-confidence-min 0.80 \
  --min-top1-similarity 0.35 \
  --min-knn5-rumor-ratio 0.20 \
  --strict-objective-protection \
  --llm-sleep-seconds 7
```

---

## 📁 目录结构

```
AIrumer2026-main/
├── README.md                         # 项目总览（本文件）
├── report.pdf                        # 课程大作业报告
└── rumor_detector_project/
    ├── README.md                     # 子项目详细文档
    ├── config.json                   # API 配置文件
    ├── requirements.txt              # Python 依赖
    ├── dataset/
    │   ├── train.csv                 # 训练集（2840 条）
    │   └── val.csv                   # 验证集（401 条）
    ├── models/
    │   ├── ensemble.joblib           # 保存的本地集成模型
    │   └── retriever.joblib          # 保存的 TF-IDF 检索器
    ├── outputs/
    │   ├── final_fusion_with_transformer/   # 保存的主干 / final-fusion 中间结果
    │   │   ├── grid_search.json             # final-fusion 实验搜索记录
    │   │   ├── metrics.json                 # 主干预测指标
    │   │   └── predictions.jsonl            # FN-aware 输入预测文件
    │   ├── fn_aware_selective_correction/   # 选择性修正最终结果
    │   │   ├── metrics.json                 # 最终准确率、混淆矩阵与翻转统计
    │   │   ├── predictions.jsonl            # 每条样本最终预测和解释字段
    │   │   ├── candidate_ranking.jsonl      # FN-risk 候选排序
    │   │   ├── reviewed_candidates.jsonl    # 实际复核候选样本
    │   │   └── classification_report.txt    # sklearn 文本版分类报告
    │   ├── confusion_matrix.png             # 混淆矩阵可视化
    │   └── event_accuracy.png               # 事件准确率可视化
    └── src/
        ├── config_utils.py                  # 配置加载，支持环境变量覆写 API Key
        ├── text_utils.py                    # 文本预处理
        ├── retriever.py                     # TF-IDF 相似样本检索器构建与检索
        ├── dedicated_model.py               # 新增：多视角 TF-IDF 与加权概率集成类
        ├── train_dedicated_local.py         # 新增：本地 dedicated 模型训练脚本
        ├── train_transformer_voter.py       # 新增：Transformer voter 微调脚本
        ├── predict_transformer_voter.py     # 新增：Transformer voter 批量推理脚本
        ├── export_transformer_voter_predictions.py # 新增：导出 Transformer 概率 JSONL
        ├── build_final_fusion_predictions.py # 新增：融合本地模型、Transformer 和检索特征
        ├── hybrid_classifier.py             # 修改：支持原始 ensemble 与 dedicated_local 两类模型
        ├── fn_aware_selective_correction.py # 修改：兼容 final-fusion 与 transformer-only JSONL 输入
        ├── llm_client.py                    # DeepSeek API 客户端
        ├── llm_reranker.py                  # LLM 复核器（Prompt + JSON 解析）
        └── predict.py                       # CLI 单条预测入口
```

---

## 🧠 技术细节

### FN-aware Risk Score

对主干判为 0 的样本，综合以下信号计算风险分数：

| 信号                       | 权重贡献    |
| -------------------------- | ----------- |
| Transformer 对谣言类的概率 | 0.3 ~ 0.8   |
| 专用模型对谣言类的概率     | 0.25 ~ 0.9  |
| Top-3 近邻中谣言比例       | 0.65 ~ 1.0  |
| Top-5 近邻中谣言比例       | 0.35 ~ 0.55 |
| Top-1 近邻为谣言且相似度高 | 0.35 ~ 1.0  |
| 本地模型对谣言类的概率     | 0.2 ~ 0.35  |

### Final-fusion JSONL 构建

`build_final_fusion_predictions.py` 负责补齐训练和 FN-aware 复核之间的中间桥接层。该脚本读取 `dataset/val.csv`，依次写入：

| 字段类别           | 主要字段                                                     | 来源                                          |
| ------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| 本地主干预测       | `base_label`, `base_prob_1`, `base_confidence`               | `models/ensemble.joblib`                      |
| dedicated 模型预测 | `dedicated_label`, `dedicated_prob_1`                        | `models/dedicated_local.joblib`，可选         |
| Transformer 预测   | `transformer_label`, `transformer_prob_1`                    | `outputs/transformer_voter_val.jsonl`，可选   |
| 检索特征           | `top1_label`, `top1_similarity`, `knn_rumor_ratio_top3/5/7`, `retrieved_examples` | `models/retriever.joblib`                     |
| 风险排序特征       | `suspicious_score`, `evidence`                               | 本地概率、Transformer 概率和 KNN 统计综合计算 |

默认 `fusion-mode=conservative`，即 `final_label` 保持为 `base_label`，由后续 `fn_aware_selective_correction.py` 负责选择性复核和翻转。

### DeepSeek 复核 Prompt 设计

复核器采用**保守型二阶段 Prompt**：

1. 系统角色设定为“保守型谣言复核器”
2. 明确要求：**只有文本包含明确未证实断言、阴谋指控或煽动性话术时**才判为谣言
3. 客观新闻报道、官方声明、现场描述优先保持非谣言
4. 输出必须为结构化 JSON，包含 `label`, `confidence`, `evidence_type`, `should_flip`, `reason`

允许的 evidence_type：

| 类型                    | 含义         | 是否支持翻转 |
| ----------------------- | ------------ | ------------ |
| `objective_report`      | 客观新闻报道 | ❌            |
| `unverified_claim`      | 未证实的断言 | ✅            |
| `conspiracy_accusation` | 阴谋指控     | ✅            |
| `sensational_claim`     | 夸张传播     | ✅            |
| `correction_or_denial`  | 澄清或否认   | ❌            |
| `unclear`               | 证据不足     | ❌            |

### 翻转门控

要完成 0 → 1 翻转，必须同时满足：

- DeepSeek 判为谣言（label=1）且建议翻转（should_flip=true）
- 置信度 ≥ 0.80
- 证据类型为 `unverified_claim` / `conspiracy_accusation` / `sensational_claim`
- 满足高相似度谣言例外或强谣言语义条件
- 未触发客观新闻保护

---

## 📈 输出文件说明

### final-fusion 输出

运行 `build_final_fusion_predictions.py` 后，`outputs/final_fusion_with_transformer/` 下生成：

| 文件                | 说明                                                |
| ------------------- | --------------------------------------------------- |
| `metrics.json`      | 主干预测准确率、混淆矩阵和分类报告                  |
| `predictions.jsonl` | 本地概率、Transformer 概率、检索特征和 FN-risk 字段 |

### 选择性修正输出

运行批量评测后，`outputs/fn_aware_selective_correction/` 下生成：

| 文件                        | 说明                                   |
| --------------------------- | -------------------------------------- |
| `metrics.json`              | 准确率、混淆矩阵、分类报告、翻转统计   |
| `predictions.jsonl`         | 每条样本的最终预测、复核结果和解释字段 |
| `candidate_ranking.jsonl`   | 所有主干判 0 样本的 FN-risk 排序       |
| `reviewed_candidates.jsonl` | 实际送入 DeepSeek 复核的候选样本       |
| `classification_report.txt` | sklearn 文本版分类报告                 |

---

## ⚠️ 已知局限

1. 真实 DeepSeek 复核效果受 API 稳定性、模型版本和 Prompt 解析稳定性影响
2. 高相似度阈值和客观新闻保护规则需根据数据集校准
3. 从头训练本地模型或 Transformer voter 时，随机种子、依赖版本和硬件环境可能导致结果波动
4. 重新生成 final-fusion JSONL 可能与仓库保存的原始 final-fusion 文件存在差异；课程报告中的 89.03% 以保存的 final-fusion 和最终复核输出为依据
5. 若换到新事件、新平台或新语言，需重新评估 FN-risk 分布

---

## 📚 附录

### 实验复现步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建检索器（如 models/retriever.joblib 不存在）
python src/retriever.py --train dataset/train.csv --out models/retriever.joblib

# 3. 生成 final-fusion JSONL（如需从当前模型重新生成）
python src/build_final_fusion_predictions.py \
  --input dataset/val.csv \
  --model models/ensemble.joblib \
  --retriever models/retriever.joblib \
  --out outputs/final_fusion_with_transformer/predictions.jsonl \
  --metrics outputs/final_fusion_with_transformer/metrics.json

# 4. 运行 Oracle 上界模拟
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55

# 5. 运行真实 DeepSeek 复核评测
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

### 子项目文档

更详细的技术说明、Motivation 分析和实验过程请参阅：
[`rumor_detector_project/README.md`](rumor_detector_project/README.md)

---

## 🧑‍🎓 致谢

《人工智能导论》课程项目 — 感谢课程组和 SJTU 模型 API 平台提供的 DeepSeek 服务。
