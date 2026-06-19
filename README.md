# 基于深度学习模型 + RAG + DeepSeek LLM的完整谣言检测解决方案

## 系统架构

采用三层架构：

1. **深度学习分类器**: 使用DistilBERT预训练模型进行二分类（谣言/非谣言）
2. **RAG检索系统**: 从训练集中检索相似样本，提供参考依据
3. **DeepSeek解释器**: 使用大语言模型生成人类可读的判断依据

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- Pandas
- OpenAI SDK

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集

- `train.csv`: 训练集
- `val.csv`: 验证集

数据格式：
- `id`: 推文ID
- `text`: 推文内容
- `label`: 标签（0=非谣言，1=谣言）
- `event`: 事件ID

## 使用方法

### 1. 训练模型

运行主训练脚本：

```bash
python rumor_detection_system.py
```

这将：
- 加载训练数据
- 训练深度学习分类器（3个epoch）
- 构建RAG向量索引
- 在验证集上评估
- 保存模型到 `rumor_detection_model`

### 2. 推理检测

**单条文本检测：**

```bash
python inference.py "BREAKING: Major news event happened!"
```

**批量检测：**

创建文本文件（每行一条文本），然后运行：

```bash
python inference.py --batch input.txt
```

结果将保存到 `inference_results.json`

## 输出说明

### 输出1: 分类结果
- **prediction**: 0 或 1
- **label_text**: "非谣言" 或 "谣言"
- **confidence**: 模型置信度 (0-1)

### 输出2: 判断依据
- **explanation**: 由DeepSeek生成的文字解释
- **rag_results**: RAG检索到的相似样本及其标签

## 示例输出

```
============================================================
检测结果
============================================================
文本: BREAKING: Museum accepts controversial art collection

预测结果: 非谣言
置信度: 0.892

相似样本参考:
1. [非谣言] Swiss Museum will take part in provenance research...
   相似度: 0.856
2. [非谣言] Where should the Gurlitt collection go?...
   相似度: 0.821

判断依据:
该推文使用了客观的新闻报道语言，没有夸张或煽动性表述。相似样本
显示这是关于博物馆接受艺术收藏的真实新闻事件，语气专业且有事实
依据，因此判定为非谣言。
============================================================
```

## 文件说明

- `rumor_detection_system.py`: 主系统代码，包含训练和评估逻辑
- `inference.py`: 推理脚本，用于实际检测
- `analyze_data.py`: 数据分析脚本
- `requirements.txt`: Python依赖包列表