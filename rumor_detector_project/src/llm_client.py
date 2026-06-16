"""Client for SJTU local large model API.

The project uses the OpenAI-compatible endpoint:
https://models.sjtu.edu.cn/api/v1/chat/completions
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from .config_utils import get_api_key, load_config
except ImportError:
    from config_utils import get_api_key, load_config


Message = Dict[str, str]


class SJTUModelClient:
    """Small wrapper around the SJTU OpenAI-compatible chat completion API."""

    def __init__(self, config_path: str | Path = "config.json", model: Optional[str] = None) -> None:
        self.config = load_config(config_path)
        api_cfg = self.config.get("api", {})
        gen_cfg = self.config.get("generation", {})

        self.base_url = str(api_cfg.get("base_url", "https://models.sjtu.edu.cn/api/v1")).rstrip("/")
        self.endpoint = str(api_cfg.get("chat_completions_endpoint", "/chat/completions"))
        self.api_key = get_api_key(self.config)
        self.model = model or str(self.config.get("default_model", "deepseek-reasoner"))
        self.temperature = float(gen_cfg.get("temperature", 0.0))
        self.max_tokens = int(gen_cfg.get("max_tokens", 512))
        self.stream = bool(gen_cfg.get("stream", False))
        self.timeout: Tuple[int, int] = (
            int(gen_cfg.get("timeout_connect_seconds", 30)),
            int(gen_cfg.get("timeout_read_seconds", 600)),
        )
        self.max_retries = int(gen_cfg.get("max_retries", 5))
        self.retry_sleep_seconds = float(gen_cfg.get("retry_sleep_seconds", 2))

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the chat completion API and return assistant content.

        DeepSeek V3.2 requires at least one user-role message. Callers should
        always include a user message in addition to any system message.
        """
        if not self.api_key:
            raise RuntimeError(
                "SJTU API key is empty. Fill api.api_key in config.json or set SJTU_API_KEY."
            )
        if not any(msg.get("role") == "user" for msg in messages):
            raise ValueError("SJTU V3.2 API requires at least one user-role message.")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": self.stream,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=data, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"request exception on attempt {attempt}: {exc}"
                time.sleep(self.retry_sleep_seconds)
                continue

            if resp.status_code == 200:
                payload = resp.json()
                return str(payload["choices"][0]["message"].get("content", "")).strip()

            last_error = f"HTTP {resp.status_code} on attempt {attempt}: {resp.text[:500]}"
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
            time.sleep(self.retry_sleep_seconds)

        raise RuntimeError(f"SJTU model request failed: {last_error}")
