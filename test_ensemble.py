"""验证集成模型效果"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ensemble_model import EnsembleRumorDetector

detector = EnsembleRumorDetector()

test_texts = [
    ("Swiss museum confirms it will take on Gurlitt collection", "训练集-谣言"),
    ("BREAKING: Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen", "验证集-谣言"),
    ("Just had a great lunch today with my friends", "日常-非谣言"),
]

print("\n" + "=" * 60)
print("集成模型测试")
print("=" * 60)

for text, desc in test_texts:
    result = detector.analyze(text)
    print(f"\n[{desc}]")
    print(f"  文本: {text[:55]}...")
    print(f"  BiGRU: {result['prob_bigru']:.4f} | BERT: {result['prob_bert']:.4f}")
    print(f"  融合: {result['prediction']} (概率={result['probability']:.4f})")
