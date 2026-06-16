"""Configuration helpers for the rumor detector project."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load JSON config and allow environment variables to override secrets."""
    path = Path(config_path)
    if not path.is_absolute():
        project_path = PROJECT_ROOT / path
        path = project_path if project_path.exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find config file: {path}")

    config = json.loads(path.read_text(encoding="utf-8-sig"))

    env_api_key = os.getenv("SJTU_API_KEY")
    if env_api_key:
        config.setdefault("api", {})["api_key"] = env_api_key

    env_base_url = os.getenv("SJTU_API_BASE_URL")
    if env_base_url:
        config.setdefault("api", {})["base_url"] = env_base_url

    env_model = os.getenv("SJTU_LLM_MODEL")
    if env_model:
        config["default_model"] = env_model

    return config


def get_api_key(config: Dict[str, Any]) -> str:
    """Return configured API key or an empty string."""
    return str(config.get("api", {}).get("api_key", "")).strip()
