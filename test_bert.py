"""验证DistilBERT和Ensemble模型"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch, config
from bert_model import DistilBERTRumorDetector

# 检查 DistilBERT
model = DistilBERTRumorDetector().to(config.DEVICE)
sd = torch.load('distilbert_rumor.pt', map_location=config.DEVICE, weights_only=True)
model.load_state_dict(sd)
model.eval()
print('DistilBERT 模型加载成功!')
print(f'分类器weight: {sd["bert.classifier.weight"].shape}')
print(f'分类器bias: {sd["bert.classifier.bias"].shape}')

# 简单测试
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
test = tokenizer("Swiss museum confirms it will take on Gurlitt collection", return_tensors="pt").to(config.DEVICE)
logits = model(test["input_ids"], test["attention_mask"])
probs = torch.softmax(logits, dim=1)
print(f'\n测试样本 - 非谣言概率: {probs[0][0]:.4f}, 谣言概率: {probs[0][1]:.4f}')
print(f'DistilBERT 检测完成!')
