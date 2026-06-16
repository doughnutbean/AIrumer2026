# Rumor Detector Project：基于 Transformer 主干与大模型选择性复核的可解释谣言检测系统

本项目是《人工智能导论》课程项目，目标是针对社交媒体短文本进行二分类谣言检测：

- `0`：非谣言（Non-rumor）
- `1`：谣言（Rumor）

项目最终采用的技术路线是：

```text
Transformer 主干模型
+ 本地概率/检索特征辅助
+ FN-aware selective correction
+ DeepSeek 保守型强复核器
+ 高相似度谣言例外放行规则
+ 客观新闻保护规则
```

系统不仅输出预测标签，还输出可解释信息，包括本地主干置信度、相似样本证据、DeepSeek 复核理由、证据类型和最终是否翻转。

---

## 1. 项目背景与问题定义

谣言检测是自然语言处理和社会计算中的典型文本分类任务。社交媒体谣言往往具有以下特点：

1. 文本短，信息稀疏；
2. 包含大量事件名、话题标签、链接和转述；
3. 谣言与非谣言可能共享同一突发事件背景；
4. 简单关键词方法容易误判客观新闻；
5. 仅靠单一分类器容易漏检隐含的未证实断言、阴谋指控和夸张传播。

因此，本项目没有采用单一模型直接完成最终判定，而是采用两阶段结构：

1. 用强主干模型保证整体分类稳定性；
2. 只对主干容易漏检的高风险非谣言预测样本进行选择性复核。

该策略尤其针对当前验证集上的错误结构：主干模型假阳性较少，但假阴性较多，因此优化重点是 **在尽量不新增 FP 的前提下修复 FN**。

---

## 2. 最终技术路线概览

最终系统流程如下：

```text
输入推文文本
    ↓
文本预处理
    ↓
Transformer 主干预测 / 本地主干概率输出
    ↓
如果主干判为 1：直接保留
如果主干判为 0：进入 FN-risk 评分
    ↓
FN-risk 候选排序
    ↓
选取 Top-K 高风险候选
    ↓
TF-IDF 检索相似训练样本
    ↓
DeepSeek reasoner 保守复核
    ↓
高置信谣言门控 + 高相似度例外放行 + 客观新闻保护
    ↓
输出最终标签与解释
```

该路线的核心设计原则是：

```text
宁可少翻转，也不随意增加误报。
```

---

## 3. 主干模型技术说明

### 3.1 Transformer 主干

前期实验中，项目训练并评估了 Transformer voter 作为强语义主干信号。Transformer 模型相比传统 TF-IDF 模型更擅长捕捉：

- 上下文语义；
- 事件相关表达；
- 否定、转述和态度；
- 社交媒体短文本中的隐含语义；
- 谣言中常见的未证实断言和煽动性表达。

当前保留的主干预测结果位于：

```text
outputs/final_fusion_with_transformer/predictions.jsonl
outputs/final_fusion_with_transformer/metrics.json
```

主干模型在验证集上的核心结果为：

```text
accuracy = 0.8877805486
confusion_matrix = [[216, 10], [35, 140]]
```

即：

```text
TN = 216
FP = 10
FN = 35
TP = 140
```

该结果说明主干模型整体较强，且 FP 控制较好；主要问题是漏检了 35 条谣言样本。因此后续改进集中在 FN-aware selective correction，而不是全局降低阈值。

### 3.2 本地集成模型辅助

除 Transformer 主干外，项目还保留了本地集成模型 artifact：

```text
models/ensemble.joblib
```

它主要由传统机器学习文本特征组成，例如：

- word-level TF-IDF；
- char-level TF-IDF；
- 线性分类器概率输出；
- 加权融合概率；
- 可解释词项特征。

该模型在最终系统中主要承担两个作用：

1. 为单条预测提供本地快速分类和解释；
2. 在 DeepSeek 复核时提供本地模型置信度和重要特征提示。

---

## 4. FN-aware Selective Correction

### 4.1 设计动机

主干模型当前错误结构为：

```text
FP = 10
FN = 35
```

如果直接降低分类阈值，虽然可能减少 FN，但往往会显著增加 FP。前期全局融合实验也表明，简单放宽规则会导致准确率下降。

因此，本项目采用 FN-aware selective correction：

```text
只对主干预测为 0 的样本进行风险排序，
只复核其中最像漏检谣言的少量样本，
只在高置信条件下允许 0 → 1 翻转。
```

### 4.2 FN-risk Score

脚本 `src/fn_aware_selective_correction.py` 会为主干判为 `0` 的样本计算风险分数。风险分数综合以下信息：

- `base_prob_1`：主干或本地模型对谣言类的概率；
- `dedicated_prob_1`：辅助模型对谣言类的概率；
- `transformer_prob_1`：Transformer 对谣言类的概率；
- `top1_label`：最相似训练样本的标签；
- `top1_similarity`：最相似训练样本的相似度；
- `knn_rumor_ratio_top5`：Top-5 相似样本中的谣言比例；
- `retrieved_examples`：相似训练样本列表。

该分数不直接决定最终标签，只用于候选排序。

### 4.3 Oracle 上界实验

为了判断该策略是否有理论提升空间，项目提供 oracle 模拟模式：

```bash
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55
```

该模式用真实标签模拟一个理想复核器。验证结果表明，Top-20 候选中确实包含足够的可修正 FN，理论上界为：

```text
accuracy = 0.9027431421
confusion_matrix = [[216, 10], [29, 146]]
```

这说明：

```text
90%+ 在候选层面是有机会的，关键在于真实复核器能否高精度识别这些 FN。
```

---

## 5. DeepSeek 强复核器

### 5.1 使用模型

项目通过 SJTU OpenAI-compatible API 调用 DeepSeek：

```text
default_model = deepseek-reasoner
```

相关代码：

```text
src/llm_client.py
src/llm_reranker.py
```

### 5.2 保守型 Prompt 设计

DeepSeek 复核器不是普通全量分类器，而是保守型二阶段复核器。它的任务是判断：

```text
当前主干判为 0 的样本，是否应当高置信翻转为 1。
```

Prompt 中明确要求：

1. 如果文本是客观报道、视频标题、官方通报、现场描述或澄清，优先保持 `0`；
2. 只有当文本存在明确未证实断言、阴谋指控、夸张传播或煽动性谣言话术时，才允许输出 `1`；
3. 不得仅因为相似样本中谣言比例高就判为谣言；
4. 只有相似样本与待测文本在具体主张层面高度同构时，才可作为高相似度谣言例外。

### 5.3 LLM 输出格式

DeepSeek 复核器必须输出 JSON：

```json
{
  "label": 0,
  "confidence": 0.85,
  "evidence_type": "objective_report",
  "should_flip": false,
  "reason": "中文解释判断依据"
}
```

其中：

- `label`：DeepSeek 判断标签；
- `confidence`：置信度，范围 `[0, 1]`；
- `evidence_type`：证据类型；
- `should_flip`：是否建议把主干的 `0` 翻为 `1`；
- `reason`：可解释说明。

允许的 `evidence_type` 包括：

```text
objective_report
unverified_claim
conspiracy_accusation
sensational_claim
correction_or_denial
unclear
```

---

## 6. 高相似度谣言例外与客观新闻保护

### 6.1 高相似度谣言例外

为了避免保守 prompt 过度抑制召回，系统加入高相似度谣言例外规则。

只有满足以下条件时，才允许翻转：

```text
DeepSeek label = 1
should_flip = true
confidence >= 0.80
evidence_type ∈ {unverified_claim, conspiracy_accusation, sensational_claim}
top1_label = 1
top1_similarity >= 指定阈值
knn_rumor_ratio_top5 >= 指定阈值
```

这保证翻转必须同时得到：

1. LLM 语义判断支持；
2. 高置信度支持；
3. 合法证据类型支持；
4. 相似样本支持。

### 6.2 客观新闻保护规则

为了避免把客观新闻误判为谣言，系统加入 objective-report protection。

以下模式会触发保护：

```text
video:
watch:
raw video
key moments
statement from
issued a statement
prime minister says
police confirm
police say
official says
calling for calm
has accounted for
last position
was travelling from
when it crashed
crowd gathers
stands by the front entrance
turns the lights off
```

触发保护后，即使样本位于高风险候选队列，也不会轻易被翻为谣言。

---

## 7. 当前实验结果

### 7.1 Transformer 主干结果

```text
accuracy = 0.8877805486
confusion_matrix = [[216, 10], [35, 140]]
```

### 7.2 Oracle 上界结果

```text
accuracy = 0.9027431421
confusion_matrix = [[216, 10], [29, 146]]
```

### 7.3 DeepSeek 直接 Top-20 复核结果

```text
accuracy = 0.8877805486
confusion_matrix = [[213, 13], [32, 143]]
```

该实验救回 3 条 FN，但新增 3 条 FP，净收益为 0。

### 7.4 保守 DeepSeek + 高置信门控结果

```text
accuracy = 0.8902743142
confusion_matrix = [[216, 10], [34, 141]]
```

该实验救回 1 条 FN，新增 0 条 FP。

### 7.5 保守 DeepSeek + 高相似度谣言例外 + 客观新闻保护结果

当前保留的最终评测命令为：

```bash
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

真实评测结果为：

```text
accuracy = 0.8902743142
confusion_matrix = [[216, 10], [34, 141]]
recovered_fn = 1
introduced_fp = 0
```

说明最终方案有效降低了误翻风险，但真实准确率尚未超过 90%。

---

## 8. 项目目录结构

```text
rumor_detector_project/
├── config.json
├── requirements.txt
├── README.md
├── dataset/
│   ├── train.csv
│   └── val.csv
├── models/
│   ├── ensemble.joblib
│   └── retriever.joblib
├── outputs/
│   ├── final_fusion_with_transformer/
│   │   ├── metrics.json
│   │   └── predictions.jsonl
│   ├── fn_audit_search/
│   │   └── all_features.jsonl
│   └── fn_aware_selective_correction/
│       ├── metrics.json
│       ├── predictions.jsonl
│       ├── candidate_ranking.jsonl
│       ├── reviewed_candidates.jsonl
│       └── classification_report.txt
└── src/
    ├── config_utils.py
    ├── text_utils.py
    ├── retriever.py
    ├── llm_client.py
    ├── llm_reranker.py
    ├── hybrid_classifier.py
    ├── fn_aware_selective_correction.py
    └── predict.py
```

---

## 9. 环境要求

推荐环境：

```text
Python >= 3.10
Windows / Linux / macOS 均可
```

主要依赖：

```text
numpy
pandas
scikit-learn
joblib
requests
matplotlib
```

依赖文件：

```text
requirements.txt
```

---

## 10. 部署与安装

### 10.1 克隆项目

```bash
git clone <your-repository-url>
cd rumor_detector_project
```

### 10.2 创建虚拟环境

Windows PowerShell：

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 10.3 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. 数据与模型准备

### 11.1 数据文件

需要准备：

```text
dataset/train.csv
dataset/val.csv
```

CSV 至少应包含：

```text
text,label
```

其中：

- `text`：推文文本；
- `label`：标签，`0` 为非谣言，`1` 为谣言。

### 11.2 模型文件

需要准备：

```text
models/ensemble.joblib
models/retriever.joblib
```

其中：

- `ensemble.joblib`：本地主干/辅助集成模型；
- `retriever.joblib`：TF-IDF 相似样本检索器。

如果没有 `retriever.joblib`，可以重新构建：

```bash
python src/retriever.py --train dataset/train.csv --out models/retriever.joblib
```

---

## 12. 配置 DeepSeek API

项目使用 `config.json` 配置 OpenAI-compatible API。

核心字段如下：

```json
{
  "api": {
    "base_url": "https://models.sjtu.edu.cn/api/v1",
    "api_key": "your-api-key",
    "chat_completions_endpoint": "/chat/completions"
  },
  "default_model": "deepseek-reasoner",
  "generation": {
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": false
  }
}
```

也可以使用环境变量覆盖 API key：

Windows PowerShell：

```bash
$env:SJTU_API_KEY="your-api-key"
```

Linux / macOS：

```bash
export SJTU_API_KEY="your-api-key"
```

> 注意：如果将项目上传到 GitHub，不建议提交真实 API key。可以在 `config.json` 中留空，或者使用环境变量。

---

## 13. 如何运行模型

### 13.1 单条文本预测：本地主干

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
  "reason": "本地集成模型判断为非谣言..."
}
```

### 13.2 单条文本预测：启用 DeepSeek 复核

```bash
python src/predict.py --use-llm --text "The police are hiding the truth and the media is lying about the victim."
```

启用 LLM 后，系统会：

1. 先运行本地模型；
2. 对低置信或困难样本检索相似训练样本；
3. 调用 DeepSeek 复核；
4. 返回最终标签和解释。

### 13.3 运行 FN-aware selective correction 评测

推荐最终评测命令：

```bash
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--use-llm` | 使用 DeepSeek 真实复核 |
| `--top-k` | 复核风险最高的 K 个主干判 0 样本 |
| `--review-threshold` | FN-risk 最低复核阈值 |
| `--llm-confidence-min` | LLM 允许翻转所需最低置信度 |
| `--min-top1-similarity` | 高相似度谣言例外的最小 Top-1 相似度 |
| `--min-knn5-rumor-ratio` | Top-5 近邻中的最小谣言比例 |
| `--strict-objective-protection` | 启用客观新闻保护 |
| `--llm-sleep-seconds` | LLM 调用间隔，避免接口限流 |

### 13.4 运行 Oracle 上界模拟

```bash
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55
```

该命令用于验证候选排序是否有理论提升空间，不代表真实系统结果。

---

## 14. 输出文件说明

运行评测后，会生成：

```text
outputs/fn_aware_selective_correction/
├── metrics.json
├── predictions.jsonl
├── candidate_ranking.jsonl
├── reviewed_candidates.jsonl
└── classification_report.txt
```

各文件含义：

| 文件 | 说明 |
|---|---|
| `metrics.json` | 准确率、混淆矩阵、分类报告、翻转统计 |
| `predictions.jsonl` | 每条样本的最终预测、复核结果和解释字段 |
| `candidate_ranking.jsonl` | 所有主干判 0 样本的 FN-risk 排序 |
| `reviewed_candidates.jsonl` | 实际送入 DeepSeek 复核的候选样本 |
| `classification_report.txt` | sklearn 文本版分类报告 |

---

## 15. 关键模块说明

### 15.1 `src/text_utils.py`

负责文本预处理，例如：

- URL 归一化；
- HTML entity 处理；
- 空白字符清洗；
- 社交媒体文本规范化。

### 15.2 `src/retriever.py`

构建和加载 TF-IDF 检索器，用于从训练集中查找与待测文本最相似的样本。

检索结果会提供给 DeepSeek 作为 few-shot 证据。

### 15.3 `src/llm_client.py`

封装 SJTU OpenAI-compatible chat completion API，负责：

- 读取 API 配置；
- 设置模型名；
- 发送请求；
- 处理重试和错误。

### 15.4 `src/llm_reranker.py`

DeepSeek 复核器，负责：

- 构造保守型谣言检测 prompt；
- 要求 LLM 输出结构化 JSON；
- 解析 `label`、`confidence`、`evidence_type`、`should_flip` 和 `reason`。

### 15.5 `src/hybrid_classifier.py`

单条文本预测入口的核心分类器，负责：

- 加载本地 ensemble 模型；
- 输出本地预测概率；
- 提取可解释词项特征；
- 在需要时调用 DeepSeek 复核。

### 15.6 `src/fn_aware_selective_correction.py`

最终评测脚本，负责：

- 读取主干预测结果；
- 构造 FN-risk score；
- 排序候选样本；
- 调用 DeepSeek；
- 应用高置信翻转门控；
- 应用高相似度谣言例外；
- 应用客观新闻保护；
- 输出最终评测指标。

---

## 16. 方法优点与局限

### 16.1 优点

1. **主干稳定**：Transformer 主干已有较高准确率；
2. **错误驱动优化**：针对 FN 多的问题设计补漏策略；
3. **降低误报风险**：不进行全局阈值放宽；
4. **可解释性较强**：提供本地特征、相似样本、LLM 理由和证据类型；
5. **工程可控**：LLM 只用于少量高风险样本，降低成本和不稳定性。

### 16.2 局限

1. 真实 DeepSeek 复核尚未稳定超过 90%；
2. LLM 输出可能受 prompt 和接口稳定性影响；
3. 高相似度阈值和客观新闻保护规则需要根据数据集校准；
4. 如果换到新事件、新平台或新语言，需要重新评估 FN-risk 分布；
5. 当前仓库保留的是最终方案和评测流程，不包含完整训练流水线。

---

## 17. 复现实验建议

推荐复现顺序：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建检索器，如果 models/retriever.joblib 不存在
python src/retriever.py --train dataset/train.csv --out models/retriever.joblib

# 3. 运行 oracle 上界模拟
python src/fn_aware_selective_correction.py --simulate-oracle --top-k 20 --review-threshold 0.55

# 4. 运行真实 DeepSeek 复核评测
python src/fn_aware_selective_correction.py --use-llm --top-k 20 --review-threshold 0.55 --llm-confidence-min 0.80 --min-top1-similarity 0.35 --min-knn5-rumor-ratio 0.20 --strict-objective-protection --llm-sleep-seconds 7
```

---

## 18. GitHub 发布注意事项

上传 GitHub 前建议检查：

1. 不要提交真实 API key；
2. 如果模型文件较大，建议使用 Git LFS 或在 Release 中提供；
3. 保留 `requirements.txt`；
4. 保留 `dataset/` 示例或说明数据来源；
5. 保留 `outputs/fn_aware_selective_correction/metrics.json` 作为结果复现参考；
6. 在 README 中明确当前真实准确率和 oracle 上界，避免夸大结果。

---

## 19. 总结

本项目最终形成了一个面向谣言检测的可解释混合系统。其核心不是简单追求单模型最高分，而是基于错误分析设计选择性修正流程：

```text
强主干负责稳定分类，
FN-aware 策略负责发现疑似漏检，
DeepSeek 负责少量难例复核，
规则门控负责防止客观新闻误翻。
```

当前真实评测结果为：

```text
accuracy = 0.8902743142
confusion_matrix = [[216, 10], [34, 141]]
```

Oracle 上界显示该方向具备超过 90% 的理论空间：

```text
accuracy = 0.9027431421
confusion_matrix = [[216, 10], [29, 146]]
```

因此，该项目展示了从传统文本分类、Transformer 语义建模到大语言模型选择性复核的完整工程实践流程，适合作为课程项目和 GitHub 展示版本。
