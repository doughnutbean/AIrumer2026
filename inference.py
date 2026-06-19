import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from model import RumorDetector
from sjtu_llm import SJTULLMExplainer
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path='models/best_model.pth', model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained('models/tokenizer')
    model = RumorDetector(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer, max_len=128):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def main():
    # 加载模型
    model, tokenizer = load_model()
    explainer = SJTULLMExplainer()

    # 读取验证集
    val_df = pd.read_csv('data/val.csv')
    texts = val_df['text'].tolist()
    true_labels = val_df['label'].tolist() if 'label' in val_df.columns else None

    # 批量预测 + 生成解释
    results = []
    for text in tqdm(texts, desc='Processing'):
        pred, conf = predict(text, model, tokenizer)
        explanation = explainer.generate_explanation(text, pred, conf)
        results.append({
            'text': text,
            'prediction': pred,
            'confidence': round(conf, 4),
            'explanation': explanation
        })

    # 保存结果
    os.makedirs('outputs', exist_ok=True)
    result_df = pd.DataFrame(results)
    if true_labels:
        result_df['true_label'] = true_labels
        accuracy = (result_df['prediction'] == result_df['true_label']).mean()
        print(f"验证集准确率: {accuracy:.4f}")
    result_df.to_csv('outputs/predictions.csv', index=False, encoding='utf-8-sig')
    print("结果已保存至 outputs/predictions.csv")

if __name__ == '__main__':
    main()