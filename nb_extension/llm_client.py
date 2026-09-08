from __future__ import annotations

import re
import time
from typing import Callable


def retry_with_suggested_backoff(func: Callable, max_retries: int = 5):
    """Retry helper for transient LLM errors."""

    try:
        from openai import RateLimitError
    except Exception:
        class RateLimitError(Exception):
            pass

    def wrapper(*args, **kwargs):
        num_retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                num_retries += 1
                is_rate_limit = isinstance(exc, RateLimitError)
                if not is_rate_limit:
                    if hasattr(exc, "http_status") and getattr(exc, "http_status") == 429:
                        is_rate_limit = True
                    elif "rate limit" in str(exc).lower() or "please try again" in str(exc).lower():
                        is_rate_limit = True

                if num_retries > max_retries:
                    raise Exception(
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                    # raise Exception(f"Maximum number of retries ({max_retries}) exceeded.") from exc

                if not is_rate_limit:
                    raise
                    # print(f"LLM call failed with error: {exc}. Retrying.")
                    # continue

                delay = None
                headers = getattr(exc, "headers", None)
                if headers:
                    retry_hdr = headers.get("retry-after") or headers.get("Retry-After")
                    if retry_hdr is not None:
                        try:
                            delay = float(retry_hdr)
                        except Exception:
                            delay = None

                if delay is None:
                    match = re.search(r"(\d+(?:\.\d+)?)(ms|s|m)\b", str(exc), flags=re.IGNORECASE)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2).lower()
                        if unit == "ms":
                            delay = value / 1000.0
                        elif unit == "s":
                            delay = value
                        elif unit == "m":
                            delay = value * 60.0

                if delay is None:
                    delay = min(2 ** num_retries, 60)

                delay = float(delay) + 1.0
                print(f"Rate limit encountered: retrying in {delay:.1f}s (attempt {num_retries}/{max_retries})")
                time.sleep(delay)

    return wrapper


class OpenAILLMClient:
    def __init__(self, model: str, system_prompt: str, max_output_tokens: int = 1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens

    @retry_with_suggested_backoff
    def _call(self, client, prompt: str) -> str:
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            max_output_tokens=self.max_output_tokens,
        )
        return response.output_text

    def run(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI import failed: {type(exc).__name__}: {exc}"
            ) from exc

        client = OpenAI()
        return self._call(client, prompt)


def _resolve_model_name(model: str | None) -> str:
    from llms.config_llms import config

    if isinstance(model, str):
        resolved = model.strip()
        if resolved:
            return resolved
    return config.openai_llm_model


def default_openai_client(model: str | None = None) -> OpenAILLMClient:
    from llms.config_llms import config

    return OpenAILLMClient(
        model=_resolve_model_name(model),
        system_prompt=config.system_prompt_crane_llm_enforcejson,
        max_output_tokens=config.max_output_tokens,
    )
