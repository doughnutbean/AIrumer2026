import os
from openai import OpenAI

class SJTULLMExplainer:
    """
    交大“致远一号”LLM解释器
    需先在交我办申请 API 密钥：https://my.sjtu.edu.cn/
    """
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.environ.get('SJTU_API_KEY')
        self.base_url = base_url or os.environ.get('SJTU_BASE_URL', 'https://models.sjtu.edu.cn/api/v1')
        if not self.api_key:
            print("⚠️ 未设置 SJTU_API_KEY，将使用降级解释")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_explanation(self, text: str, prediction: int, confidence: float = None) -> str:
        label = "谣言" if prediction == 1 else "真实信息"
        conf_str = f"，置信度 {confidence:.2f}" if confidence else ""

        if self.client is None:
            return self._fallback_explanation(text, prediction)

        prompt = f"""请分析以下社交媒体文本是否构成谣言，并给出具体的判断依据。

文本内容：
"{text}"

检测结果：该文本被判定为{label}{conf_str}。

请按以下格式输出判断依据（200字以内）：
1. 关键语言特征分析
2. 内容逻辑自洽性
3. 信息可信度评估
4. 综合结论"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "system", "content": "你是一个专业的谣言检测分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self._fallback_explanation(text, prediction)

    def _fallback_explanation(self, text: str, prediction: int) -> str:
        if prediction == 1:
            return "该文本被判定为谣言：包含夸张表述或未经证实的信息，缺乏可靠来源支持。"
        else:
            return "该文本被判定为真实信息：内容表达客观，逻辑连贯，未发现明显谣言特征。"