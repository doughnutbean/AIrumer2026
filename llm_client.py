"""
SJTU LLM API 客户端
- OpenAI 兼容格式调用
- 支持 deepseek-chat / deepseek-reasoner 等模型
"""

import json
import urllib.request
import urllib.error
import config


class SJTU_LLM_Client:
    """SJTU 大模型 API 客户端"""

    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        api_key: str = config.LLM_API_KEY,
        model: str = config.LLM_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = f"{self.base_url}/chat/completions"

    def _call(self, messages: list[dict], max_tokens: int = config.LLM_MAX_TOKENS,
              temperature: float = config.LLM_TEMPERATURE) -> str:
        """调用 LLM API"""
        if not self.api_key:
            return "[错误] 未配置LLM_API_KEY，请在config.py中填入SJTU API密钥"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return content.strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            return f"[HTTP错误 {e.code}] {error_body[:300]}"
        except Exception as e:
            return f"[调用失败] {str(e)}"

    def chat(self, system_prompt: str, user_message: str) -> str:
        """发送对话请求"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._call(messages)

    def generate_judgment(
        self, text: str, label: int, confidence: float, similar_cases: str = ""
    ) -> str:
        """生成谣言检测的判断依据"""
        label_name = "谣言" if label == 1 else "非谣言"

        system_prompt = (
            "你是一个专业的社交媒体信息分析师。给定一条推文和检测结果，"
            "你需要用中文输出一段清晰、有逻辑的判断依据，解释为什么这条推文是或不是谣言。\n\n"
            "要求：\n"
            "1. 从文本内容、语言风格、信息来源等角度分析\n"
            "2. 结合常见的谣言特征（情绪化表述、缺乏可信来源、夸张表述等）\n"
            "3. 输出控制在150-300字，简洁有力\n"
            "4. 不要重复输入文本，只输出判断依据"
        )

        user_message = (
            f"推文内容：\n\"{text}\"\n\n"
            f"检测结果：{label_name}（置信度：{confidence:.2%}）\n"
        )
        if similar_cases:
            user_message += (
                f"\n训练集中相似的已标注样本：\n{similar_cases}\n"
            )

        user_message += "\n请给出判断依据："

        return self.chat(system_prompt, user_message)

    def list_models(self) -> list[str]:
        """列出可用的模型"""
        if not self.api_key:
            return ["未配置API_KEY"]
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            return [f"获取失败: {e}"]
