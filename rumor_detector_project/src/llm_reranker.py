"""LLM-based reranking and explanation for difficult rumor samples."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .llm_client import SJTUModelClient
except ImportError:
    from llm_client import SJTUModelClient


LABEL_NAMES = {0: "非谣言", 1: "谣言"}


class LLMReranker:
    """Use DeepSeek reasoner to review low-confidence local predictions."""

    def __init__(self, config_path: str | Path = "config.json", model: Optional[str] = None) -> None:
        self.client = SJTUModelClient(config_path=config_path, model=model)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, object]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"LLM output does not contain JSON: {text[:300]}")
        return json.loads(match.group(0))

    @staticmethod
    def _normalize_evidence_type(value: object) -> str:
        text = str(value or "").strip().lower()
        allowed = {
            "objective_report",
            "unverified_claim",
            "conspiracy_accusation",
            "sensational_claim",
            "correction_or_denial",
            "unclear",
        }
        if text in allowed:
            return text
        return "unclear"

    @staticmethod
    def _normalize_confidence(value: object, default: float = 0.5) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return default
        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return conf

    @staticmethod
    def build_prompt(
        text: str,
        local_label: int,
        local_confidence: float,
        feature_cues: List[str],
        similar_examples: List[Dict[str, object]],
    ) -> List[Dict[str, str]]:
        examples_text = []
        for i, ex in enumerate(similar_examples, start=1):
            sample_text = str(ex.get("text", "")).replace("\n", " ")[:500]
            label = int(ex.get("label", 0))
            sim = float(ex.get("similarity", 0.0))
            examples_text.append(
                f"样本{i}:\n文本: {sample_text}\n标签: {label}（{LABEL_NAMES[label]}）\n相似度: {sim:.4f}"
            )

        cues = "、".join(feature_cues) if feature_cues else "无明显高权重特征"
        system = (
            "你是一个用于课程项目的保守型谣言复核器。"
            "任务是判断一条英文推文是否应从非谣言翻转为谣言。"
            "标签定义：0=非谣言，1=谣言。"
            "你的首要目标不是覆盖更多样本，而是避免误报。"
            "只有当文本包含明确的、未证实的、具有传播煽动性的谣言特征时，才允许输出 label=1。"
            "如果文本只是客观新闻报道、现场描述、视频标题、官方声明、转述、更新、澄清或证据不足，必须输出 label=0。"
            "必须只输出合法 JSON，不要输出 Markdown 或额外说明。"
        )
        user = f"""下面是训练集中与待检测文本最相似的若干样本：

{chr(10).join(examples_text)}

本地模型预测：
label = {local_label}（{LABEL_NAMES[local_label]}）
confidence = {local_confidence:.4f}
重要特征 = {cues}

待检测文本：
{text}

请只输出合法 JSON，格式如下：
{{
  "label": 0 或 1,
  "confidence": 0 到 1 之间的小数，表示你对该标签的置信度，只有当你对 label=1 非常确定时才给出 0.80 以上，
  "evidence_type": "objective_report | unverified_claim | conspiracy_accusation | sensational_claim | correction_or_denial | unclear",
  "should_flip": true 或 false，只有当你建议把本地模型的 0 翻成 1 时才为 true，
  "reason": "用中文解释判断依据，说明文本里哪些表达支持该类型"
}}

判定要求：
1. 如果文本主要是客观报道、视频/图片/直播标题、官方通报、现场状态描述、转述或澄清，优先输出 0。
2. 只有明显存在未经证实的事件断言、阴谋指控、夸张传播或煽动性谣言话术，才输出 1。
3. 如果最相似样本为谣言且与待测文本表达的是同一个具体断言，而不是仅共享同一新闻事件背景，可以作为高相似度谣言例外；此时 evidence_type 应选择 unverified_claim 或 conspiracy_accusation，而不是 objective_report。
4. 不要因为相似样本里很多 1 就判 1；只有当相似样本和待测文本在具体主张层面高度同构时才可以支持翻转。
5. 仅在 confidence >= 0.80 且 evidence_type 属于 unverified_claim、conspiracy_accusation、sensational_claim 时，should_flip 才能为 true。
6. 如果你犹豫、证据不足、只是普通新闻链接或只是共享同一事件背景，应保持 0。
"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def review(
        self,
        text: str,
        local_label: int,
        local_confidence: float,
        feature_cues: List[str],
        similar_examples: List[Dict[str, object]],
    ) -> Dict[str, object]:
        messages = self.build_prompt(text, local_label, local_confidence, feature_cues, similar_examples)
        content = self.client.chat(messages)
        parsed = self._extract_json(content)
        label = int(parsed.get("label", local_label))
        if label not in (0, 1):
            label = local_label
        confidence = self._normalize_confidence(parsed.get("confidence"), default=0.5)
        evidence_type = self._normalize_evidence_type(parsed.get("evidence_type"))
        should_flip = bool(parsed.get("should_flip", False))
        reason = str(parsed.get("reason", "大模型未返回有效解释。"))
        return {
            "label": label,
            "confidence": confidence,
            "evidence_type": evidence_type,
            "should_flip": should_flip,
            "reason": reason,
            "raw": content,
        }
