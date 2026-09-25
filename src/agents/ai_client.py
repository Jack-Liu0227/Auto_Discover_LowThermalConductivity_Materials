# -*- coding: utf-8 -*-
"""AI client using LiteLLM with configurable fallback behavior."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm import completion

logger = logging.getLogger(__name__)

try:
    from .llm_models import get_llm_models_config  # type: ignore
except ImportError:
    from llm_models import get_llm_models_config  # type: ignore


class LLMResponseTruncatedError(RuntimeError):
    """Raised when the provider stops a response at the output token limit."""


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

    def get_default_max_tokens(self, task: str = "workflow") -> Optional[int]:
        """Return the configured output budget for a task.

        Theory updates must return a complete document. The previous fixed
        8000-token budget was too small for the current prompt/document pair,
        so the budget is configurable without changing workflow calls.
        """
        if task != "theory_update":
            return None

        raw_value = os.getenv("THEORY_UPDATE_MAX_TOKENS", "32000").strip()
        if raw_value.lower() in {"", "none", "null", "unlimited"}:
            return None
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("THEORY_UPDATE_MAX_TOKENS must be a positive integer or 'none'") from exc
        if value <= 0:
            raise ValueError("THEORY_UPDATE_MAX_TOKENS must be a positive integer or 'none'")
        return value

    def _build_theory_update_proxy_client(
        self,
        model_config: Dict[str, Any],
    ) -> Any:
        """Build an isolated OpenAI client for the optional theory proxy.

        LiteLLM's OpenAI-compatible path accepts an OpenAI client. Using an
        isolated httpx transport keeps the theory proxy from changing the
        workflow model's networking and disables inherited proxy variables for
        this explicit transport to avoid double proxying.
        """
        proxy = os.getenv("THEORY_UPDATE_PROXY", "").strip()
        if not proxy:
            return None

        try:
            import httpx
            from openai import OpenAI

            http_client = httpx.Client(
                proxy=proxy,
                timeout=self.request_timeout_sec,
                follow_redirects=True,
                trust_env=False,
            )
            try:
                return OpenAI(
                    api_key=model_config.get("api_key") or None,
                    base_url=model_config.get("base_url") or None,
                    timeout=self.request_timeout_sec,
                    max_retries=0,
                    http_client=http_client,
                )
            except Exception:
                http_client.close()
                raise
        except Exception as exc:
            # Do not include the proxy URL in an exception or log: it may
            # contain credentials. Keep the failure actionable but secret-safe.
            raise RuntimeError("Failed to initialize theory-update proxy transport") from exc

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
            except LLMResponseTruncatedError:
                raise
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
        # Keep retry ownership in this client; do not multiply LiteLLM and
        # provider retries for one logical workflow request.
        litellm.num_retries = 0
        litellm.retry_on_timeout = False

        request_client = None
        if model_id == "theory_update":
            request_client = self._build_theory_update_proxy_client(model_config)
            if request_client is not None:
                logger.info("Using dedicated proxy for theory-update model")
                kwargs = {**kwargs, "client": request_client}

        try:
            response = self._completion_with_provider_retry(
            actual_model=actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=model_config.get("api_key") or None,
            api_base=model_config.get("base_url") or None,
            **kwargs,
        )
        finally:
            if request_client is not None:
                request_client.close()

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason", finish_reason)
        if str(finish_reason or "").lower() in {"length", "max_tokens"}:
            raise LLMResponseTruncatedError(
                f"LLM response truncated by output limit (finish_reason={finish_reason})"
            )

        message = getattr(choice, "message", None)
        if isinstance(choice, dict):
            message = choice.get("message", message)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response content is empty; workflow aborted")
        return content

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
        # Keep retry ownership in this client; do not multiply LiteLLM and
        # provider retries for one logical workflow request.
        litellm.num_retries = 0
        litellm.retry_on_timeout = False

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
            # OpenAI-compatible clients otherwise add their own retries.
            "max_retries": 0,
            **kwargs,
        }

        attempts = max(0, self.num_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                return completion(model=actual_model, **request_kwargs)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if self._should_retry_with_openai_provider(actual_model, api_base, exc):
                    retried_model = f"openai/{actual_model}"
                    logger.warning(
                        "LLM provider was unspecified; retrying with provider-qualified model "
                        "model=%s endpoint=%s",
                        retried_model,
                        api_base or "<default>",
                    )
                    try:
                        return completion(model=retried_model, **request_kwargs)
                    except Exception as provider_exc:
                        exc = provider_exc
                        status_code = getattr(exc, "status_code", status_code)

                logger.warning(
                    "LLM request failed model=%s endpoint=%s attempt=%d/%d status=%s error=%s",
                    actual_model,
                    api_base or "<default>",
                    attempt,
                    attempts,
                    status_code if status_code is not None else "unknown",
                    str(exc)[:500],
                )
                if attempt >= attempts or not self._is_retryable_error(exc):
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))

        raise RuntimeError("LLM request retry loop exited unexpectedly")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {400, 401, 403, 404, 422}:
                return False
            return status_code in {408, 429} or status_code >= 500

        error_text = str(exc).lower()
        return any(
            marker in error_text
            for marker in (
                "timeout",
                "timed out",
                "connecterror",
                "connection error",
                "connection reset",
                "temporarily unavailable",
                "server error",
                "service unavailable",
                "rate limit",
                "too many requests",
            )
        )

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
