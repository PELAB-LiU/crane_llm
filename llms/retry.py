"""Shared retry helper for LLM API calls.

Used by both the batch experiment runner (``llms.llm_executor``) and the
notebook extension (``nb_extension.llm_client``). The two differ only in how
they treat non-rate-limit failures, which is exposed as a parameter.
"""

from __future__ import annotations

import functools
import re
import time
from typing import Callable, Optional


def _rate_limit_error_type():
    try:
        from openai import RateLimitError

        return RateLimitError
    except Exception:
        return ()


def _is_rate_limit(exc: BaseException) -> bool:
    rate_limit_type = _rate_limit_error_type()
    if rate_limit_type and isinstance(exc, rate_limit_type):
        return True
    if getattr(exc, "http_status", None) == 429:
        return True
    if getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "rate limit" in message or "please try again" in message


def _suggested_delay(exc: BaseException, attempt: int) -> float:
    """Best-effort retry delay: response header, then message, then backoff."""

    headers = getattr(exc, "headers", None)
    if headers:
        try:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            retry_after = None
        if retry_after is not None:
            try:
                return float(retry_after) + 1.0
            except (TypeError, ValueError):
                pass

    match = re.search(r"(\d+(?:\.\d+)?)(ms|s|m)\b", str(exc), flags=re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "ms":
            return value / 1000.0 + 1.0
        if unit == "s":
            return value + 1.0
        if unit == "m":
            return value * 60.0 + 1.0

    return float(min(2 ** attempt, 60)) + 1.0


def retry_on_rate_limit(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 5,
    retry_other_errors: bool = False,
    other_error_delay: float = 1.0,
) -> Callable:
    """Retry ``func`` when the provider reports a rate limit.

    ``retry_other_errors`` controls what happens for every other exception.
    The batch runner retries them (long unattended jobs should survive a blip);
    the notebook extension re-raises immediately so the user sees the real
    error instead of waiting through silent retries.
    """

    def decorate(target: Callable) -> Callable:
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return target(*args, **kwargs)
                except Exception as exc:
                    attempt += 1
                    rate_limited = _is_rate_limit(exc)

                    if attempt > max_retries:
                        raise RuntimeError(
                            f"Maximum number of retries ({max_retries}) exceeded. "
                            f"Last error: {type(exc).__name__}: {exc}"
                        ) from exc

                    if not rate_limited:
                        if not retry_other_errors:
                            raise
                        print(f"LLM call failed with error: {exc}. Retrying.")
                        time.sleep(other_error_delay)
                        continue

                    delay = _suggested_delay(exc, attempt)
                    print(
                        f"Rate limit encountered: retrying in {delay:.1f}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(delay)

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
