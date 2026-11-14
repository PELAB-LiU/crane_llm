"""Sign in with a ChatGPT subscription (Plus, Pro, Team) instead of an API key.

EXPERIMENTAL. This uses the same OAuth client and backend as OpenAI's Codex
CLI, the way other third-party agents (e.g. Hermes Agent) do. That backend is
not a documented public API: OpenAI can change or restrict it at any time, the
models are those of your ChatGPT plan, and usage counts against that plan's
limits rather than an API bill.

How it works:

1. ``login()`` runs OpenAI's device-code flow: it prints a URL and a one-time
   code, the user approves it in a browser, and the code is exchanged for an
   access token and a refresh token.
2. The tokens are stored in ``chatgpt_auth.json`` next to the configuration
   file (``~/.crane_llm/`` by default), owner-readable only. They are kept
   apart from ``~/.codex/auth.json`` on purpose: every refresh rotates the
   refresh token, so two programs sharing one session would sign each other
   out.
3. ``access_token()`` returns a valid token, refreshing it shortly before it
   expires.
4. Requests go to ``BASE_URL`` with the Responses API, plus the account
   headers built by ``request_headers()``.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .texts import text


ISSUER = "https://auth.openai.com"
# The public OAuth client of OpenAI's Codex CLI. There is no other client a
# third party can use for ChatGPT sign-in.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = f"{ISSUER}/oauth/token"
DEVICE_CODE_URL = f"{ISSUER}/api/accounts/deviceauth/usercode"
DEVICE_POLL_URL = f"{ISSUER}/api/accounts/deviceauth/token"
DEVICE_VERIFY_URL = f"{ISSUER}/codex/device"
DEVICE_REDIRECT_URI = f"{ISSUER}/deviceauth/callback"

BASE_URL = "https://chatgpt.com/backend-api/codex"
MODELS_URL = f"{BASE_URL}/models"

# Identifies this client to the backend, which asks third-party harnesses to
# name themselves rather than pose as the Codex CLI.
ORIGINATOR = "crane-llm"

AUTH_FILE_NAME = "chatgpt_auth.json"

# Refresh this long before the access token expires.
REFRESH_SKEW_SECONDS = 120
# Give up waiting for the user to approve the device code after this long.
LOGIN_TIMEOUT_SECONDS = 15 * 60

# Offered when the model catalogue cannot be fetched. Only slugs that the ChatGPT
# backend accepts; it rejects the public API's "-pro" variants.
FALLBACK_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]
DEFAULT_MODEL = FALLBACK_MODELS[0]


class ChatGPTAuthError(RuntimeError):
    """Signing in, or keeping the session alive, failed."""


# ---------------------------------------------------------------------------
# Token store


def auth_path() -> Path:
    """Where the tokens are kept: beside the configuration file.

    Following ``settings.config_path()`` means ``CRANE_LLM_CONFIG`` moves both,
    which is what keeps the tests away from the developer's real session.
    """

    from .settings import config_path

    return config_path().parent / AUTH_FILE_NAME


def load_tokens() -> Optional[Dict[str, Any]]:
    """The stored tokens, or ``None`` when not signed in."""

    try:
        data = json.loads(auth_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("access_token") or not data.get("refresh_token"):
        return None
    return data


def is_logged_in() -> bool:
    return load_tokens() is not None


def save_tokens(tokens: Dict[str, Any]) -> Path:
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tokens, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.replace(tmp, path)
    return path


def logout() -> bool:
    """Forget the stored session. True if there was one."""

    try:
        auth_path().unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# JWT helpers. The token is only decoded, never verified: the backend does that.


def _jwt_claims(token: str) -> Dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _is_expiring(token: str, skew_seconds: int = REFRESH_SKEW_SECONDS) -> bool:
    exp = _jwt_claims(token).get("exp")
    if not isinstance(exp, (int, float)):
        # No expiry to go by: assume it is still valid and let a 401 say otherwise.
        return False
    return exp - skew_seconds <= time.time()


def account_email(token: str) -> Optional[str]:
    claims = _jwt_claims(token)
    profile = claims.get("https://api.openai.com/profile")
    if isinstance(profile, dict) and isinstance(profile.get("email"), str):
        return profile["email"]
    email = claims.get("email")
    return email if isinstance(email, str) else None


def request_headers(token: str) -> Dict[str, str]:
    """Headers the backend needs besides the bearer token.

    ``ChatGPT-Account-ID`` selects the workspace; without it the backend
    answers 401 or an empty model list. Residency-enforced workspaces also
    need their region.
    """

    from .. import __version__

    headers = {"originator": ORIGINATOR, "User-Agent": f"crane-llm/{__version__}"}
    auth = _jwt_claims(token).get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        account = auth.get("chatgpt_account_id")
        if isinstance(account, str) and account:
            headers["ChatGPT-Account-ID"] = account
        residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
        if isinstance(residency, str) and residency.strip():
            headers["x-openai-internal-codex-residency"] = residency.strip()
    return headers


# ---------------------------------------------------------------------------
# HTTP


def _http():
    # httpx comes with the OpenAI SDK, which is already a dependency.
    import httpx

    return httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _token_response(response) -> Dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict) or not data.get("access_token"):
        raise ChatGPTAuthError(text("setup.chatgpt.bad_token_response", status=response.status_code))
    return data


def refresh(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """Exchange the refresh token for a new access token and store the result."""

    with _http() as client:
        response = client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": CLIENT_ID,
            },
        )
    if response.status_code in (400, 401, 403):
        # The refresh token was revoked or already rotated by another client.
        raise ChatGPTAuthError(text("errors.chatgpt_session_expired"))
    if response.status_code != 200:
        raise ChatGPTAuthError(text("setup.chatgpt.refresh_failed", status=response.status_code))

    data = _token_response(response)
    updated = dict(tokens)
    updated["access_token"] = data["access_token"]
    if data.get("refresh_token"):
        updated["refresh_token"] = data["refresh_token"]
    if data.get("id_token"):
        updated["id_token"] = data["id_token"]
    updated["last_refresh"] = _now()
    save_tokens(updated)
    return updated


def access_token() -> str:
    """A valid access token, refreshed if it is about to expire."""

    tokens = load_tokens()
    if tokens is None:
        raise ChatGPTAuthError(text("errors.chatgpt_not_logged_in"))
    if _is_expiring(tokens["access_token"]):
        tokens = refresh(tokens)
    return tokens["access_token"]


def login(
    output: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = LOGIN_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run the device-code sign-in and store the tokens.

    ``output`` and ``sleep`` are parameters so the flow can be tested without
    a terminal or a real wait.
    """

    with _http() as client:
        response = client.post(DEVICE_CODE_URL, json={"client_id": CLIENT_ID})
        if response.status_code != 200:
            raise ChatGPTAuthError(text("setup.chatgpt.device_code_failed", status=response.status_code))
        device = response.json()
        user_code = device.get("user_code")
        device_auth_id = device.get("device_auth_id")
        if not user_code or not device_auth_id:
            raise ChatGPTAuthError(text("setup.chatgpt.device_code_failed", status=response.status_code))
        try:
            interval = max(3, int(device.get("interval", 5)))
        except (TypeError, ValueError):
            interval = 5

        output(text("setup.chatgpt.device_instructions", url=DEVICE_VERIFY_URL, code=user_code))

        # Poll until the user approves. 403/404 mean "not yet".
        deadline = time.monotonic() + timeout_seconds
        code_response: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            sleep(interval)
            poll = client.post(
                DEVICE_POLL_URL, json={"device_auth_id": device_auth_id, "user_code": user_code}
            )
            if poll.status_code == 200:
                code_response = poll.json()
                break
            if poll.status_code not in (403, 404):
                raise ChatGPTAuthError(text("setup.chatgpt.poll_failed", status=poll.status_code))
        if code_response is None:
            raise ChatGPTAuthError(text("setup.chatgpt.login_timeout"))

        authorization_code = code_response.get("authorization_code")
        code_verifier = code_response.get("code_verifier")
        if not authorization_code or not code_verifier:
            raise ChatGPTAuthError(text("setup.chatgpt.bad_token_response", status=200))

        exchange = client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": DEVICE_REDIRECT_URI,
                "client_id": CLIENT_ID,
                "code_verifier": code_verifier,
            },
        )
        if exchange.status_code != 200:
            raise ChatGPTAuthError(text("setup.chatgpt.bad_token_response", status=exchange.status_code))
        data = _token_response(exchange)
        if not data.get("refresh_token"):
            raise ChatGPTAuthError(text("setup.chatgpt.bad_token_response", status=200))

    tokens = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "last_refresh": _now(),
    }
    if data.get("id_token"):
        tokens["id_token"] = data["id_token"]
    save_tokens(tokens)
    return tokens


def fetch_models() -> Optional[List[str]]:
    """The models this ChatGPT account may use, best first, or ``None``."""

    try:
        token = access_token()
        headers = {"Authorization": f"Bearer {token}", **request_headers(token)}
        with _http() as client:
            # The catalogue is versioned by client; the high version asks for
            # everything the account may use.
            response = client.get(MODELS_URL, params={"client_version": "99.0.0"}, headers=headers)
        entries = response.json().get("models") if response.status_code == 200 else None
    except Exception:
        entries = None
    if not isinstance(entries, list):
        return None

    ranked = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str):
            continue
        if str(entry.get("visibility", "")).lower() in ("hide", "hidden"):
            continue
        priority = entry.get("priority")
        ranked.append((priority if isinstance(priority, (int, float)) else 10_000, entry["slug"]))
    return list(dict.fromkeys(slug for _, slug in sorted(ranked))) or None
