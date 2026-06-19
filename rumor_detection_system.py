import pandas as pd
import numpy as np
import json
import pickle
from typing import List, Dict, Tuple
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sentence_transformers import SentenceTransformer
import faiss

from openai import OpenAI

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class RumorClassifier(nn.Module):
    """基于预训练模型的谣言分类器"""
    def __init__(self, model_name='distilbert-base-uncased', dropout=0.3):
        super(RumorClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


class RAGRetriever:
    """RAG检索器：从训练集中检索相似样本"""
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        print(f"初始化RAG检索器，使用模型: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []
        self.labels = []
        self.ids = []

    def build_index(self, texts: List[str], labels: List[int], ids: List[str]):
        """构建FAISS索引"""
        print("构建向量索引...")
        self.texts = texts
        self.labels = labels
        self.ids = ids

        # 生成embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # 归一化以便使用余弦相似度
        faiss.normalize_L2(embeddings)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (余弦相似度)
        self.index.add(embeddings)

        print(f"索引构建完成，共 {len(texts)} 条记录")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索最相似的top_k条记录"""
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'text': self.texts[idx],
                'label': self.labels[idx],
                'id': self.ids[idx],
                'similarity': float(distances[0][i])
            })

        return results

    def save(self, path: str):
        """保存索引"""
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'labels': self.labels,
                'ids': self.ids
            }, f)
        faiss.write_index(self.index, f"{path}_index.faiss")
        print(f"RAG索引已保存到 {path}")

    def load(self, path: str):
        """加载索引"""
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.texts = metadata['texts']
            self.labels = metadata['labels']
            self.ids = metadata['ids']
        self.index = faiss.read_index(f"{path}_index.faiss")
        print(f"RAG索引已从 {path} 加载")


class DeepSeekExplainer:
    """使用DeepSeek生成判断依据"""
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate_explanation(
        self,
        text: str,
        prediction: int,
        rag_results: List[Dict],
        model: str = "deepseek-chat"
    ) -> str:
        """生成判断依据"""

        # 构建RAG上下文
        rag_context = "相似样本参考：\n"
        for i, result in enumerate(rag_results[:3], 1):
            label_text = "谣言" if result['label'] == 1 else "非谣言"
            rag_context += f"{i}. [{label_text}] {result['text']} (相似度: {result['similarity']:.3f})\n"

        # 构建prompt
        prediction_text = "谣言" if prediction == 1 else "非谣言"

        system_prompt = f"""你是一个专业的谣言检测分析专家。你需要根据提供的信息，解释为什么一条推文被判定为{prediction_text}。

你的判断依据应该基于：
1. 文本内容的特征（情绪化、夸张、缺乏证据等）
2. 相似样本的参考（这些样本已被标注为谣言或非谣言）
3. 语言特征（如绝对性表述、紧迫性、煽动性等）

{rag_context}

请给出简洁、专业的判断依据（2-3句话），说明为什么这条推文是{prediction_text}。"""

        user_prompt = f"待检测推文：{text}\n\n请分析并给出判断依据。"

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            explanation = response.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            return f"生成解释时出错: {str(e)}"


class RumorDetectionSystem:
    """完整的谣言检测系统"""
    def __init__(
        self,
        classifier_model_name='distilbert-base-uncased',
        rag_model_name='all-MiniLM-L6-v2',
        api_key='sk-80zR-Twna32QTLkqGCW5Zg',
        base_url='https://models.sjtu.edu.cn/api/v1'
    ):
        print("初始化谣言检测系统...")

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 初始化分类器
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        self.classifier = RumorClassifier(classifier_model_name).to(self.device)

        # 初始化RAG检索器
        self.rag_retriever = RAGRetriever(rag_model_name)

        # 初始化DeepSeek解释器
        self.explainer = DeepSeekExplainer(api_key, base_url)

        print("系统初始化完成")

    def train_classifier(
        self,
        train_texts,
        train_labels,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    ):
        """训练分类器"""
        print("\n开始训练分类器...")

        # 创建数据集
        dataset = RumorDataset(train_texts, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                logits = self.classifier(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })

            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    def build_rag_index(self, texts, labels, ids):
        """构建RAG索引"""
        print("\n构建RAG知识库...")
        self.rag_retriever.build_index(texts, labels, ids)

    def predict(self, text: str, return_explanation: bool = True) -> Dict:
        """预测单条文本"""
        self.classifier.eval()

        # 1. 使用分类器预测
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        result = {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'label_text': '谣言' if prediction == 1 else '非谣言'
        }

        # 2. 使用RAG检索相似样本
        rag_results = self.rag_retriever.retrieve(text, top_k=5)
        result['rag_results'] = rag_results

        # 3. 使用DeepSeek生成解释
        if return_explanation:
            explanation = self.explainer.generate_explanation(
                text, prediction, rag_results
            )
            result['explanation'] = explanation

        return result

    def evaluate(self, test_texts, test_labels, test_ids, save_results=True):
        """评估模型性能"""
        print("\n开始评估模型...")

        predictions = []
        results = []

        for i, (text, true_label, text_id) in enumerate(tqdm(
            zip(test_texts, test_labels, test_ids),
            total=len(test_texts),
            desc="评估中"
        )):
            # 只在前10个样本生成解释（节省时间和成本）
            return_explanation = (i < 10)
            result = self.predict(text, return_explanation=return_explanation)

            predictions.append(result['prediction'])
            results.append({
                'id': text_id,
                'text': text,
                'true_label': true_label,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'explanation': result.get('explanation', '')
            })

        # 计算准确率
        accuracy = accuracy_score(test_labels, predictions)
        print(f"\n准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            test_labels,
            predictions,
            target_names=['非谣言', '谣言']
        ))

        # 保存结果
        if save_results:
            results_df = pd.DataFrame(results)
            results_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8')
            print("\n评估结果已保存到 evaluation_results.csv")

        return accuracy, results

    def save_model(self, path='model_checkpoint'):
        """保存模型"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
        }, f"{path}.pt")
        self.rag_retriever.save(path)
        print(f"模型已保存到 {path}")

    def load_model(self, path='model_checkpoint'):
        """加载模型"""
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.rag_retriever.load(path)
        print(f"模型已从 {path} 加载")


def main():
    """主函数"""
    print("=" * 60)
    print("谣言检测系统 - 训练和评估")
    print("=" * 60)

    # 1. 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"训练集谣言比例: {train_df['label'].sum() / len(train_df):.2%}")
    print(f"验证集谣言比例: {val_df['label'].sum() / len(val_df):.2%}")

    # 2. 初始化系统
    system = RumorDetectionSystem(
        classifier_model_name='distilbert-base-uncased',
        rag_model_name='all-MiniLM-L6-v2'
    )

    # 3. 训练分类器
    system.train_classifier(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        epochs=3,
        batch_size=16
    )

    # 4. 构建RAG索引
    system.build_rag_index(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        ids=train_df['id'].astype(str).tolist()
    )

    # 5. 保存模型
    system.save_model('rumor_detection_model')

    # 6. 评估模型
    accuracy, results = system.evaluate(
        test_texts=val_df['text'].tolist(),
        test_labels=val_df['label'].tolist(),
        test_ids=val_df['id'].astype(str).tolist()
    )

    # 7. 展示一些示例结果
    print("\n" + "=" * 60)
    print("检测示例 (前5条):")
    print("=" * 60)
    for i, result in enumerate(results[:5]):
        print(f"\n[示例 {i+1}]")
        print(f"文本: {result['text'][:100]}...")
        print(f"真实标签: {'谣言' if result['true_label'] == 1 else '非谣言'}")
        print(f"预测结果: {'谣言' if result['prediction'] == 1 else '非谣言'}")
        print(f"置信度: {result['confidence']:.3f}")
        if result['explanation']:
            print(f"判断依据: {result['explanation']}")

    print("\n" + "=" * 60)
    print("训练和评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
