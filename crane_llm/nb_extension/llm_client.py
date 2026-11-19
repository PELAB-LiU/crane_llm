from __future__ import annotations

from typing import Any, Dict, List, Optional


from ..llms.retry import retry_on_rate_limit
from . import settings as settings_module
from .chatgpt_auth import ChatGPTError
from .texts import text


def _error(message: str, output: Optional[str] = "") -> RuntimeError:
    """A failed call, with the partial output if there was any.

    ``output=None`` says there was none at all, which is worth saying.
    """

    if output:
        message += " " + text("errors.model_partial_output", output=output)
    elif output is None:
        message += " " + text("errors.model_no_output")
    return RuntimeError(message)


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
                    text("errors.openai_import", error=f"{type(exc).__name__}: {exc}")
                ) from exc

            if not self.api_key:
                raise RuntimeError(settings_module.missing_key_message())

            kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                # Without this the HTTP client offers brotli whenever a brotli
                # module is importable, and a hosted image can pair a new HTTP
                # client with an old or impostor one (Kaggle ships brotlipy
                # under the name ``brotli``). The reply then fails to decode
                # with "Decompressor.decompress() got an unexpected keyword
                # argument 'output_buffer_limit'". The replies are small JSON,
                # so gzip costs nothing.
                "default_headers": {"Accept-Encoding": "gzip, deflate"},
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _truncation_error(self, output: str) -> RuntimeError:
        return _error(text("errors.model_truncated", limit=self.max_output_tokens), output)

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
        output = (getattr(response, "output_text", None) or "").strip()

        status = getattr(response, "status", None)
        if status is not None and status != "completed":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "max_output_tokens":
                raise self._truncation_error(output)
            message = (
                text("errors.model_status_reason", status=status, reason=reason)
                if reason
                else text("errors.model_status", status=status)
            )
            raise _error(message, output or None)

        if not output:
            raise RuntimeError(text("errors.model_empty"))

        return output


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
            raise RuntimeError(text("errors.model_no_choices"))

        choice = choices[0]
        message = getattr(choice, "message", None)
        output = (getattr(message, "content", None) or "").strip()

        # Reasoning models served this way put the answer in `content` and the
        # chain of thought in a separate field, but some gateways return only
        # the latter when the budget runs out mid-thought.
        if not output:
            output = (getattr(message, "reasoning_content", None) or "").strip()

        if getattr(choice, "finish_reason", None) == "length":
            raise self._truncation_error(output)

        if not output:
            raise RuntimeError(text("errors.model_empty"))

        return output


# Plan-usage error codes from the docs, and what to tell the user.
_CHATGPT_ERRORS = {
    "subscription_sharing_usage_limit_exceeded": "errors.chatgpt_usage_limit",
    "subscription_sharing_usage_unavailable": "errors.chatgpt_unavailable",
    "subscription_sharing_user_unavailable": "errors.chatgpt_unavailable",
    "subscription_sharing_user_not_eligible": "errors.chatgpt_not_eligible",
    "subscription_sharing_invalid_user": "errors.chatgpt_session_expired",
}


def chatgpt_error_message(code: str, detail: str = "") -> str:
    key = _CHATGPT_ERRORS.get(code or "")
    if key:
        return text(key)
    return text("errors.chatgpt_request_failed", code=code or "-", detail=detail or "-")


class ChatGPTSubscriptionLLMClient(OpenAILLMClient):
    """The public Responses API, paid for by a ChatGPT plan. EXPERIMENTAL.

    Same endpoint as ``OpenAILLMClient``, with the access token from the
    ChatGPT sign-in as the bearer token and the request shape the docs set for
    plan usage: ``store`` false, always streamed, the system prompt in
    ``instructions``, and no ``max_output_tokens`` or ``reasoning``. A reply
    counts only once ``response.completed`` arrives. Not retried on a 429: that
    is the plan's usage limit, which a retry cannot lift.

    The SDK client is rebuilt for every call because the access token is
    short-lived and may have been refreshed.
    """

    def _get_client(self):
        from . import chatgpt_auth

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                text("errors.openai_import", error=f"{type(exc).__name__}: {exc}")
            ) from exc

        return OpenAI(
            api_key=chatgpt_auth.access_token(),
            base_url=chatgpt_auth.BASE_URL,
            max_retries=0,
            default_headers={"Accept-Encoding": "gzip, deflate"},
        )

    def _request_kwargs(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": [{"role": "user", "content": prompt}],
            "store": False,
            "stream": True,
        }

    def _call(self, client, prompt: str) -> str:
        try:
            stream = client.responses.create(**self._request_kwargs(prompt))
            deltas: List[str] = []
            final = None
            for event in stream:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    deltas.append(getattr(event, "delta", "") or "")
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    final = getattr(event, "response", None)
                elif kind == "error":
                    raise ChatGPTError(
                        chatgpt_error_message(getattr(event, "code", "") or "", getattr(event, "message", "") or "")
                    )
        except ChatGPTError:
            raise
        except Exception as exc:
            body = getattr(exc, "body", None)
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                body = body["error"]
            code = body.get("code") if isinstance(body, dict) else None
            if code:
                raise ChatGPTError(chatgpt_error_message(code, body.get("message", ""))) from exc
            raise

        output = "".join(deltas).strip()
        status = getattr(final, "status", None)
        if status == "failed":
            error = getattr(final, "error", None)
            raise ChatGPTError(
                chatgpt_error_message(getattr(error, "code", "") or "", getattr(error, "message", "") or "")
            )
        if status != "completed":
            reason = getattr(getattr(final, "incomplete_details", None), "reason", None)
            message = (
                text("errors.model_status_reason", status=status, reason=reason)
                if reason
                else text("errors.model_status", status=status or "no response.completed")
            )
            raise _error(message, output or None)
        if not output:
            raise RuntimeError(text("errors.model_empty"))
        return output


def default_client(
    model: Optional[str] = None,
    include_runinfo: bool = True,
    resolved: Optional["settings_module.Settings"] = None,
) -> _BaseLLMClient:
    """Build a client whose system prompt matches the prompt being sent.

    With runtime information switched off the prompt has no
    [Current relevant runtime information] section, so the system prompt must
    not tell the model to expect one.

    ``resolved`` lets a caller that has already resolved the settings pass them
    in instead of looking them up a second time.
    """

    from ..llms.config_llms import config

    if resolved is None:
        resolved = settings_module.resolve(model=model)

    system_prompt = (
        config.system_prompt_crane_llm_enforcejson
        if include_runinfo
        else config.system_prompt_crane_llm_enforcejson_code_only
    )

    client_class = {
        "responses": OpenAILLMClient,
        settings_module.CHATGPT_STYLE: ChatGPTSubscriptionLLMClient,
    }.get(resolved.api_style, ChatCompletionsLLMClient)

    return client_class(
        model=resolved.model,
        system_prompt=system_prompt,
        max_output_tokens=config.max_output_tokens,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
    )

