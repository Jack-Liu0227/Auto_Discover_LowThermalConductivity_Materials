from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, create_model, field_validator


class WorkflowInput(BaseModel):
    # Runtime overrides
    max_iterations: Optional[int | str] = Field(default=5, description="Max iterations for this run (1-20)")
    samples: Optional[int | str] = Field(default=100, description="Bayesian sample size (>0)")
    n_structures: Optional[int | str] = Field(default=5, description="Structures per candidate (>0)")
    top_k_bayes: Optional[int | str] = Field(default=20, description="Top-K after Bayes (>0)")
    top_k_screen: Optional[int | str] = Field(default=10, description="Top-K for AI screening (>0)")
    db_candidate_workers: Optional[int | str] = Field(default=8, description="Parallel workers for candidate dedup (>0)")
    websearch_enabled: Optional[bool] = Field(default=True, description="Enable web search enrichment")
    websearch_top_n: Optional[int | str] = Field(default=2, description="Top-N candidates for web search (>=0)")

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _coerce_max_iterations(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        parsed = int(value)
        if not 1 <= parsed <= 20:
            raise ValueError("max_iterations must be between 1 and 20")
        return parsed

    @field_validator("samples", "n_structures", "top_k_bayes", "top_k_screen", "db_candidate_workers", mode="before")
    @classmethod
    def _coerce_positive_int(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("value must be > 0")
        return parsed

    @field_validator("websearch_top_n", mode="before")
    @classmethod
    def _coerce_websearch_top_n(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError("websearch_top_n must be >= 0")
        return parsed


def build_workflow_input_schema(defaults: dict[str, Any] | None = None) -> type[WorkflowInput]:
    defaults = defaults or {}
    field_defs: dict[str, tuple[Any, Any]] = {}
    key_map = {
        "max_iterations": "agentos_default_iterations",
        "samples": "samples",
        "n_structures": "n_structures",
        "top_k_bayes": "top_k_bayes",
        "top_k_screen": "top_k_screen",
        "db_candidate_workers": "db_candidate_workers",
        "websearch_enabled": "websearch_enabled",
        "websearch_top_n": "websearch_top_n",
    }
    for key, config_key in key_map.items():
        f = WorkflowInput.model_fields[key]
        default_value = defaults.get(config_key, defaults.get(key, f.default))
        field_defs[key] = (
            f.annotation,
            Field(default=default_value, description=f.description),
        )
    return create_model("WorkflowInputRuntime", __base__=WorkflowInput, **field_defs)
