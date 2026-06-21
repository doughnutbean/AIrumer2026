"""
模型定义
- BiGRU 基线模型
- 可选：事件类别嵌入 + 注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Attention(nn.Module):
    """加性注意力机制，对BiGRU所有时间步的输出计算加权和"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            gru_outputs: (batch, seq_len, hidden_dim*2)  双向GRU输出
            mask: (batch, seq_len)  padding mask, 1表示有效位置, 0表示padding
        Returns:
            context: (batch, hidden_dim*2)  加权后的上下文向量
        """
        # (batch, seq_len, 1)
        score = self.attention_weights(gru_outputs)

        if mask is not None:
            # 将padding位置的分数设为负无穷
            score = score.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # (batch, seq_len, 1)
        attn_weights = F.softmax(score, dim=1)

        # (batch, hidden_dim*2)
        context = (gru_outputs * attn_weights).sum(dim=1)
        return context


class BiGRUModel(nn.Module):
    """BiGRU谣言检测模型（支持事件嵌入和注意力机制）"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        use_event_emb: bool = config.USE_EVENT_EMB,
        event_emb_dim: int = config.EVENT_EMB_DIM,
        num_events: int = config.NUM_EVENTS,
        use_attention: bool = config.USE_ATTENTION,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.use_event_emb = use_event_emb
        self.use_attention = use_attention

        # 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        # 双向GRU
        self.bigru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 注意力机制
        if use_attention:
            self.attention = Attention(hidden_dim * 2)

        # 事件嵌入
        if use_event_emb:
            self.event_embedding = nn.Embedding(num_events, event_emb_dim)

        # 计算分类器输入维度
        classifier_input_dim = hidden_dim * 2  # 双向GRU最后一层输出
        if use_attention:
            pass  # 注意力输出也是 hidden_dim*2
        if use_event_emb:
            classifier_input_dim += event_emb_dim

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 分类器
        self.fc = nn.Linear(classifier_input_dim, 1)

    def forward(self, input_ids, event=None):
        """
        Args:
            input_ids: (batch, seq_len)
            event: (batch,) 可选的事件类别
        Returns:
            logits: (batch,)
        """
        # 词嵌入 (batch, seq_len, embedding_dim)
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # 创建padding mask
        mask = input_ids != 0  # (batch, seq_len)

        # BiGRU编码
        # outputs: (batch, seq_len, hidden_dim*2)
        # hidden: (num_layers*2, batch, hidden_dim)
        outputs, hidden = self.bigru(emb)

        if self.use_attention:
            # 注意力池化
            text_feat = self.attention(outputs, mask)
        else:
            # 取最后一层双向的最终隐状态拼接
            # hidden[-2] 是正向最后一层, hidden[-1] 是反向最后一层
            h_fwd = hidden[-2]  # (batch, hidden_dim)
            h_bwd = hidden[-1]  # (batch, hidden_dim)
            text_feat = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden_dim*2)

        text_feat = self.dropout(text_feat)

        # 拼接事件嵌入
        if self.use_event_emb:
            if event is not None:
                event_feat = self.event_embedding(event)  # (batch, event_emb_dim)
            else:
                # 推理时未提供事件，用零向量填充
                event_feat = torch.zeros(
                    text_feat.size(0), config.EVENT_EMB_DIM, device=text_feat.device
                )
            combined = torch.cat([text_feat, event_feat], dim=1)
        else:
            combined = text_feat

        # 分类
        logits = self.fc(combined).squeeze(1)  # (batch,)
        return logits

    def predict(self, input_ids, event=None) -> list[int]:
        """预测二分类标签（0或1）"""
        logits = self.forward(input_ids, event)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().tolist()
        return preds
