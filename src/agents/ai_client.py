# -*- coding: utf-8 -*-
"""AI client using LiteLLM with configurable fallback behavior."""

import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm import completion

try:
    from .llm_models import get_llm_models_config  # type: ignore
except ImportError:
    from llm_models import get_llm_models_config  # type: ignore


class AIClient:
    """AI client wrapper."""

    def __init__(self, config_file: str | None = None):
        _ = config_file  # legacy arg kept for compatibility
        config = get_llm_models_config()
        self.default_model_id: str = config.get("default_model", "")
        self.workflow_model_id: str = config.get("workflow_model", self.default_model_id)
        self.theory_update_model_id: str = config.get("theory_update_model", self.workflow_model_id)
        self.alternative_models: List[str] = list(config.get("alternative_models", []))
        self.temperature_config: Dict[str, Any] = dict(config.get("temperature", {}))

        self.models: Dict[str, Dict[str, Any]] = {}
        for model in config.get("models", []):
            self.models[model["id"]] = model

        self.request_timeout_sec: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "120"))
        self.num_retries: int = int(os.getenv("LLM_NUM_RETRIES", "2"))

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self.models.values())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_id)

    def get_default_model(self, task: str = "workflow") -> str:
        if task == "theory_update":
            return self.theory_update_model_id or self.workflow_model_id or self.default_model_id
        return self.workflow_model_id or self.default_model_id

    def get_default_temperature(self, task: str = "workflow") -> Optional[float]:
        if task == "theory_update":
            return self.temperature_config.get("theory_update")
        return self.temperature_config.get("evaluation")

    def _candidate_models(self, primary_model_id: str) -> List[str]:
        candidates: List[str] = [primary_model_id]

        for fallback_id in self.alternative_models:
            if fallback_id != primary_model_id and fallback_id in self.models:
                candidates.append(fallback_id)

        for fallback_id in self.models.keys():
            if fallback_id not in candidates:
                candidates.append(fallback_id)

        return candidates

    def chat(
        self,
        prompt: str,
        model_id: str = "",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        auto_fallback: bool = True,
        **kwargs,
    ) -> str:
        if not model_id:
            model_id = self.default_model_id
        if model_id not in self.models:
            raise ValueError(f"模型 ID 不存在: {model_id}. 可用模型: {list(self.models.keys())}")

        candidates = self._candidate_models(model_id)
        if not auto_fallback:
            candidates = candidates[:1]

        errors: List[str] = []
        for idx, candidate in enumerate(candidates):
            try:
                if idx > 0:
                    print(f"[fallback] primary model failed, switching to: {candidate}")
                return self._call_model(
                    model_id=candidate,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                errors.append(f"{candidate}: {e}")

        raise Exception("AI 调用失败。" + " | ".join(errors))

    def _call_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        model_config = self.models[model_id]
        actual_model = model_config["model"]

        if temperature is None:
            temperature = model_config.get("default_temperature", 0.7)

        if not model_config.get("api_key"):
            raise ValueError(f"模型 {model_id} 未配置 API Key")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        import litellm

        litellm.request_timeout = self.request_timeout_sec
        litellm.num_retries = self.num_retries
        litellm.retry_on_timeout = True

        response = self._completion_with_provider_retry(
            actual_model=actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=model_config.get("api_key") or None,
            api_base=model_config.get("base_url") or None,
            **kwargs,
        )
        return response.choices[0].message["content"]

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        auto_fallback: bool = True,
        **kwargs,
    ) -> str:
        if not model_id:
            model_id = self.default_model_id
        if model_id not in self.models:
            raise ValueError(f"模型 ID 不存在: {model_id}")

        candidates = self._candidate_models(model_id)
        if not auto_fallback:
            candidates = candidates[:1]

        errors: List[str] = []
        for idx, candidate in enumerate(candidates):
            try:
                if idx > 0:
                    print(f"[fallback] primary model failed, switching to: {candidate}")
                return self._call_model_with_history(
                    model_id=candidate,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                errors.append(f"{candidate}: {e}")

        raise Exception("AI 调用失败。" + " | ".join(errors))

    def _call_model_with_history(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        model_config = self.models[model_id]
        actual_model = model_config["model"]

        if temperature is None:
            temperature = model_config.get("default_temperature", 0.7)

        if not model_config.get("api_key"):
            raise ValueError(f"模型 {model_id} 未配置 API Key")

        import litellm

        litellm.request_timeout = self.request_timeout_sec
        litellm.num_retries = self.num_retries
        litellm.retry_on_timeout = True

        response = self._completion_with_provider_retry(
            actual_model=actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=model_config.get("api_key") or None,
            api_base=model_config.get("base_url") or None,
            **kwargs,
        )
        return response.choices[0].message["content"]

    def _completion_with_provider_retry(
        self,
        actual_model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        api_key: Optional[str],
        api_base: Optional[str],
        **kwargs,
    ):
        request_kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "api_base": api_base,
            "timeout": self.request_timeout_sec,
            **kwargs,
        }

        try:
            return completion(model=actual_model, **request_kwargs)
        except Exception as exc:
            if self._should_retry_with_openai_provider(actual_model, api_base, exc):
                retried_model = f"openai/{actual_model}"
                print(f"[provider-fallback] retrying with provider-qualified model: {retried_model}")
                return completion(model=retried_model, **request_kwargs)
            raise

    @staticmethod
    def _should_retry_with_openai_provider(actual_model: str, api_base: Optional[str], exc: Exception) -> bool:
        if not api_base:
            return False
        if "/" in actual_model:
            return False
        error_text = str(exc)
        return "LLM Provider NOT provided" in error_text


if __name__ == "__main__":
    client = AIClient()
    print("可用模型:")
    for model in client.list_models():
        print(f"  - {model['id']}: {model['name']} ({model['description']})")
