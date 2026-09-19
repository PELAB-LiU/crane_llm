from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


from llms.retry import retry_on_rate_limit


_DOTENV_LOADED = False


def _load_env_once() -> None:
    """Pick up API keys from the repository ``.env``.

    The batch runner calls ``load_dotenv()`` at import time, so anyone who
    followed the project README and put ``OPENAI_API_KEY`` in ``.env`` expects
    it to work here too.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    try:
        from dotenv import load_dotenv
    except Exception:
        return

    try:
        load_dotenv()
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass


def _supports_reasoning(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def _supports_verbosity(model: str) -> bool:
    return model.startswith("gpt-5")


class OpenAILLMClient:
    def __init__(self, model: str, system_prompt: str, max_output_tokens: int = 1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _request_kwargs(self, prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": self.max_output_tokens,
        }

        # These are specific to the reasoning models. Sending them to, say,
        # gpt-4o is an API error, and the model name is user-supplied.
        if _supports_reasoning(self.model):
            kwargs["reasoning"] = {"effort": "minimal"}
        if _supports_verbosity(self.model):
            kwargs["text"] = {"verbosity": "low"}

        return kwargs

    @retry_on_rate_limit(max_retries=5, retry_other_errors=False)
    def _call(self, client, prompt: str) -> str:
        response = client.responses.create(**self._request_kwargs(prompt))
        return self._read_output(response)

    def _read_output(self, response) -> str:
        text = (getattr(response, "output_text", None) or "").strip()

        status = getattr(response, "status", None)
        if status is not None and status != "completed":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "max_output_tokens":
                raise RuntimeError(
                    "The model hit the output token limit "
                    f"(max_output_tokens={self.max_output_tokens}) before finishing. "
                    "Raise `max_output_tokens` in llms/config_llms.py and try again."
                    + (f" Partial output: {text}" if text else "")
                )
            raise RuntimeError(
                f"The model returned status {status!r}"
                + (f" (reason: {reason})" if reason else "")
                + (f". Partial output: {text}" if text else " with no output.")
            )

        if not text:
            raise RuntimeError("The model returned an empty response.")

        return text

    def _get_client(self):
        if self._client is None:
            _load_env_once()
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError(
                    f"OpenAI import failed: {type(exc).__name__}: {exc}"
                ) from exc
            self._client = OpenAI()
        return self._client

    def run(self, prompt: str) -> str:
        return self._call(self._get_client(), prompt)


def _resolve_model_name(model: Optional[str]) -> str:
    from llms.config_llms import config

    if isinstance(model, str):
        resolved = model.strip()
        if resolved:
            return resolved
    return config.openai_llm_model


def default_openai_client(
    model: Optional[str] = None,
    include_runinfo: bool = True,
) -> OpenAILLMClient:
    """Build a client whose system prompt matches the prompt being sent.

    With runtime information switched off the prompt has no
    [Current relevant runtime information] section, so the system prompt must
    not tell the model to expect one.
    """

    from llms.config_llms import config

    system_prompt = (
        config.system_prompt_crane_llm_enforcejson
        if include_runinfo
        else config.system_prompt_crane_llm_enforcejson_code_only
    )

    return OpenAILLMClient(
        model=_resolve_model_name(model),
        system_prompt=system_prompt,
        max_output_tokens=config.max_output_tokens,
    )
