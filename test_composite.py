"""测试复合谣言检测模型"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from composite_model import CompositeRumorDetector

# 初始化（不需要LLM API Key也能跑检测部分）
detector = CompositeRumorDetector()

# 测试样本
tests = [
    ("Swiss museum confirms it will take on Gurlitt collection", 0, "训练集-谣言"),
    ("BREAKING: Ferguson police chief just announced that officer Darren Wilson shot unarmed teen Mike Brown", 1, "验证集-谣言"),
    ("Just had a great lunch today with my friends", None, "日常-非谣言"),
    ("Shoot unarmed kid. Conceal evidence. Impose martial law. Harass reporters. Smear the victim. Worst. Police. Ever.", 1, "情绪化-谣言"),
]

print("\n" + "=" * 60)
print("复合谣言检测模型测试")
print("=" * 60)

for text, event, desc in tests:
    result = detector.analyze(text, event=event)
    print(f"\n[{desc}]")
    print(f"  文本: {text[:60]}...")
    print(f"  检测: {result['prediction']} (概率={result['probability']:.4f})")
    print(f"  判断依据: {result['judgment_basis']}")
    print()

# 测试RAG检索
print("=" * 60)
print("RAG检索测试")
print("=" * 60)
from rag import get_retriever
retriever = get_retriever()
similar = retriever.format_retrieved("Shoot unarmed kid. Conceal evidence.")
print(f"\n检索结果:\n{similar}")
