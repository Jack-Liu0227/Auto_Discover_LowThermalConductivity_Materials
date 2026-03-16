# -*- coding: utf-8 -*-
"""
LLM model configuration.

This project uses OpenAI-compatible endpoints for both workflow evaluation and
the theory-document updater. The only task-specific differences are model name,
API key, and base URL, all sourced from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ENV_PATH = PROJECT_ROOT / ".env"
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.deepseek.com/v1"


def _normalize_base_url(value: str | None, default: str) -> str:
    text = (value or "").strip()
    return text or default


def _resolve_actual_model(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return normalized
    if "/" in normalized:
        return normalized
    return f"openai/{normalized}"


def _model_entry(
    entry_id: str,
    name: str,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "description": f"{name} model configuration",
        "provider": "openai",
        "model_name": model_name,
        "model": _resolve_actual_model(model_name),
        "api_key": api_key,
        "base_url": base_url,
        "default_temperature": temperature,
        "enabled": True,
    }


def get_llm_models_config() -> Dict[str, Any]:
    temperature = float(os.getenv("TEMPERATURE", "0.3"))

    workflow_model_name = os.getenv("WORKFLOW_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    workflow_api_key = os.getenv("WORKFLOW_API_KEY", "").strip()
    workflow_base_url = _normalize_base_url(
        os.getenv("WORKFLOW_BASE_URL"),
        DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    )

    theory_update_model_name = os.getenv("THEORY_UPDATE_MODEL", workflow_model_name).strip() or workflow_model_name
    theory_update_api_key = os.getenv("THEORY_UPDATE_API_KEY", workflow_api_key).strip()
    theory_update_base_url = _normalize_base_url(
        os.getenv("THEORY_UPDATE_BASE_URL"),
        workflow_base_url,
    )

    models: List[Dict[str, Any]] = [
        _model_entry(
            entry_id="workflow",
            name="Workflow",
            model_name=workflow_model_name,
            api_key=workflow_api_key,
            base_url=workflow_base_url,
            temperature=temperature,
        ),
        _model_entry(
            entry_id="theory_update",
            name="Theory Update",
            model_name=theory_update_model_name,
            api_key=theory_update_api_key,
            base_url=theory_update_base_url,
            temperature=temperature,
        ),
    ]

    return {
        "default_model": "workflow",
        "workflow_model": "workflow",
        "theory_update_model": "theory_update",
        "models": models,
        "alternative_models": [],
        "temperature": {
            "evaluation": temperature,
            "theory_update": temperature,
        },
        "max_tokens": {
            "evaluation": None,
            "theory_update": None,
        },
    }


if __name__ == "__main__":
    llm_config = get_llm_models_config()
    print("=== LLM model config ===")
    for model in llm_config["models"]:
        print(
            f"{model['id']}: model_name={model['model_name']} "
            f"actual={model['model']} base_url={model['base_url']}"
        )
