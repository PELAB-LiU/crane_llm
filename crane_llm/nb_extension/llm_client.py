from __future__ import annotations

from typing import Any, Dict, List, Optional


from ..llms.retry import retry_on_rate_limit
from . import settings as settings_module


def _supports_reasoning(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def _supports_verbosity(model: str) -> bool:
    return model.startswith("gpt-5")


class _BaseLLMClient:
    """Shared plumbing for the two OpenAI SDK call styles.

    Both styles use the same SDK object, differing only in which endpoint they
    call and how the reply is shaped, so construction and error handling live
    here and the subclasses supply the two halves that differ.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError(
                    f"OpenAI import failed: {type(exc).__name__}: {exc}"
                ) from exc

            if not self.api_key:
                raise RuntimeError(settings_module.missing_key_message())

            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _truncation_error(self, text: str) -> RuntimeError:
        return RuntimeError(
            "The model hit the output token limit "
            f"(max_output_tokens={self.max_output_tokens}) before finishing. "
            "Raise `max_output_tokens` in crane_llm/llms/config_llms.py and try again."
            + (f" Partial output: {text}" if text else "")
        )

    def run(self, prompt: str) -> str:
        return self._call(self._get_client(), prompt)


class OpenAILLMClient(_BaseLLMClient):
    """The OpenAI Responses API, as used for the experiments in the paper."""

    def _request_kwargs(self, prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": self._messages(prompt),
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
                raise self._truncation_error(text)
            raise RuntimeError(
                f"The model returned status {status!r}"
                + (f" (reason: {reason})" if reason else "")
                + (f". Partial output: {text}" if text else " with no output.")
            )

        if not text:
            raise RuntimeError("The model returned an empty response.")

        return text


class ChatCompletionsLLMClient(_BaseLLMClient):
    """Chat Completions, for any OpenAI-compatible endpoint.

    This is the path for everything that is not a plain OpenAI account:
    OpenRouter (and through it Claude and Gemini), Groq, Together, DeepSeek,
    Azure OpenAI, vLLM and Ollama. They implement ``/chat/completions`` and
    almost none implement ``/responses``, which is why this exists separately
    rather than as a flag on the class above.

    The reasoning and verbosity parameters are deliberately not sent here. They
    are OpenAI-specific and a compatible server will usually reject an unknown
    field outright.
    """

    def _request_kwargs(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._messages(prompt),
            "max_tokens": self.max_output_tokens,
        }

    @retry_on_rate_limit(max_retries=5, retry_other_errors=False)
    def _call(self, client, prompt: str) -> str:
        response = client.chat.completions.create(**self._request_kwargs(prompt))
        return self._read_output(response)

    def _read_output(self, response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("The model returned no choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "").strip()

        # Reasoning models served this way put the answer in `content` and the
        # chain of thought in a separate field, but some gateways return only
        # the latter when the budget runs out mid-thought.
        if not text:
            text = (getattr(message, "reasoning_content", None) or "").strip()

        if getattr(choice, "finish_reason", None) == "length":
            raise self._truncation_error(text)

        if not text:
            raise RuntimeError("The model returned an empty response.")

        return text


def _resolve_model_name(model: Optional[str] = None) -> str:
    """The model that would be used for ``model``.

    ``api.py`` calls this to decide whether a request needs a rebuilt client,
    so it has to apply the same resolution order as ``default_client``.
    """

    return settings_module.resolve(model=model).model


def default_client(
    model: Optional[str] = None,
    include_runinfo: bool = True,
) -> _BaseLLMClient:
    """Build a client whose system prompt matches the prompt being sent.

    With runtime information switched off the prompt has no
    [Current relevant runtime information] section, so the system prompt must
    not tell the model to expect one.
    """

    from ..llms.config_llms import config

    resolved = settings_module.resolve(model=model)

    system_prompt = (
        config.system_prompt_crane_llm_enforcejson
        if include_runinfo
        else config.system_prompt_crane_llm_enforcejson_code_only
    )

    client_class = (
        OpenAILLMClient if resolved.api_style == "responses" else ChatCompletionsLLMClient
    )

    return client_class(
        model=resolved.model,
        system_prompt=system_prompt,
        max_output_tokens=config.max_output_tokens,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )


# The name this module exposed before it supported anything but OpenAI. Kept so
# that a notebook or script written against the earlier version still runs.
default_openai_client = default_client
