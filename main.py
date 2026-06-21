"""
主入口：训练BiGRU谣言检测模型
"""

import torch
import config
from data import load_data, Vocab
from model import BiGRUModel
from train import run_training


def main():
    print("=" * 50)
    print("社交媒体谣言检测 - BiGRU基线模型")
    print(f"设备: {config.DEVICE}")
    print(f"模型配置: 事件嵌入={config.USE_EVENT_EMB}, 注意力={config.USE_ATTENTION}")
    print(f"          Embedding={config.EMBEDDING_DIM}, Hidden={config.HIDDEN_DIM}, Layers={config.NUM_LAYERS}")
    print("=" * 50)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    train_loader, val_loader, vocab = load_data()
    print(f"词表大小: {len(vocab)}")

    # 2. 创建模型
    print("\n[2/3] 创建模型...")
    model = BiGRUModel(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
    ).to(config.DEVICE)

    # 打印模型结构
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,} | 可训练参数: {trainable_params:,}")

    # 3. 训练
    print("\n[3/3] 开始训练...")
    model, history = run_training(model, train_loader, val_loader, config.DEVICE)

    # 4. 保存词表
    vocab.save(config.TOKENIZER_SAVE_PATH)
    print(f"\n词表已保存至: {config.TOKENIZER_SAVE_PATH}")
    print(f"模型已保存至: {config.MODEL_SAVE_PATH}")
    print("\n训练完成!")


if __name__ == "__main__":
    main()
