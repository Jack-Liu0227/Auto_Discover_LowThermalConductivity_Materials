# -*- coding: utf-8 -*-
"""
LLM model configuration.
Loads environment variables from project .env and exposes a small, explicit API.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def _normalize_base_url(value: str | None, default: str) -> str:
    text = (value or "").strip()
    return text or default


def _model_entry(
    entry_id: str,
    name: str,
    actual_model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "description": f"{name} model configuration",
        "model": actual_model,
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
        DEFAULT_DEEPSEEK_BASE_URL,
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
            actual_model=workflow_model_name,
            api_key=workflow_api_key,
            base_url=workflow_base_url,
            temperature=temperature,
        ),
        _model_entry(
            entry_id="theory_update",
            name="Theory Update",
            actual_model=theory_update_model_name,
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


def get_file_paths_config(iteration_num: int = 1, version: int = 1, results_root: str = "results") -> Dict[str, Any]:
    return {
        "evaluation": {
            "input": {"dir": f"{results_root}/iteration_{iteration_num}/reports", "filename": "llm_evaluation_input.md"},
            "output": {"dir": f"{results_root}/iteration_{iteration_num}/reports", "filename": "llm_evaluation_output.md"},
            "selected_csv": {"dir": f"{results_root}/iteration_{iteration_num}/selected_results", "filename": "ai_selected_materials.csv"},
        },
        "theory_update": {
            "input": {"dir": f"{results_root}/iteration_{iteration_num}/reports", "filename": "llm_theory_update_input.md"},
            "output": {"dir": f"{results_root}/iteration_{iteration_num}/reports", "filename": "llm_theory_update_output.md"},
            "updated_doc": {"dir": f"llm/doc/v0.0.{version}", "filename": "Theoretical_principle_document.md"},
        },
        "documents": {
            "initial_theory_doc": "llm/doc/v0.0.1/Theoretical_principle_document.md",
            "versioned_theory_doc": f"llm/doc/v0.0.{version}/Theoretical_principle_document.md",
        },
        "success": {
            "csv_dir": f"{results_root}/iteration_{iteration_num}/success_examples",
            "csv_filename": "success_materials.csv",
        },
    }


def get_evaluation_params() -> Dict[str, Any]:
    return {"n_select": 5, "n_candidates": 20}


def get_iteration_config() -> Dict[str, Any]:
    return {"max_rounds": 10, "convergence_threshold": 0.95}


def get_naming_config() -> Dict[str, Any]:
    return {"use_timestamp": True, "timestamp_format": "%Y%m%d_%H%M%S"}


class LLMConfig:
    def __init__(self):
        self._models_config = get_llm_models_config()
        self._file_paths_default = get_file_paths_config()
        self._eval_params = get_evaluation_params()
        self._iteration = get_iteration_config()
        self._naming = get_naming_config()

    @property
    def default_model(self) -> str:
        return self._models_config["default_model"]

    @property
    def workflow_model(self) -> str:
        return self._models_config["workflow_model"]

    @property
    def theory_update_model(self) -> str:
        return self._models_config["theory_update_model"]

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self._models_config["models"]

    @property
    def alternative_models(self) -> List[str]:
        return self._models_config["alternative_models"]

    def get_model(self, model_id: str) -> Dict[str, Any]:
        for model in self.models:
            if model["id"] == model_id:
                return model
        raise ValueError(f"Model ID not found: {model_id}")

    def get_file_paths(self, iteration_num: int = 1, version: int = 1, results_root: str = "results") -> Dict[str, Any]:
        return get_file_paths_config(iteration_num, version, results_root)

    @property
    def evaluation_params(self) -> Dict[str, Any]:
        return self._eval_params

    @property
    def iteration_config(self) -> Dict[str, Any]:
        return self._iteration

    @property
    def naming_config(self) -> Dict[str, Any]:
        return self._naming


config = LLMConfig()


def get_config() -> LLMConfig:
    return config


if __name__ == "__main__":
    llm_config = get_llm_models_config()
    print("=== LLM model config ===")
    print(f"Workflow model: {llm_config['models'][0]['model']}")
    print(f"Theory update model: {llm_config['models'][1]['model']}")
