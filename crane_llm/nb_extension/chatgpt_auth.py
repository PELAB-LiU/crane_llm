"""Sign in with ChatGPT: use a ChatGPT plan (Plus, Pro, Business) instead of an API key.

EXPERIMENTAL. Follows OpenAI's documented flow for open-source, locally hosted
tools ("Sign in with ChatGPT", token sharing for open source):
https://developers.openai.com/siwc/token-sharing-open-source/sign-in

How it works:

1. **Registration.** The first sign-in registers CRANE-LLM as its own OAuth
   client for the user's account: the authorize request carries
   ``client_id=dynamic_agent_client``, the agent name and this machine's host
   ID, and the callback returns the issued client ID (``oaiapp_...``). That ID
   is saved and used for every later sign-in, token exchange and refresh.
2. **Loopback redirect.** The browser is sent back to
   ``http://127.0.0.1:<port>/auth/callback``, where ``login()`` listens. This
   only works when the browser runs on the same machine as the kernel (local
   JupyterLab), or reaches it through an SSH tunnel. It cannot work on
   Kaggle or Colab, and ``login()`` refuses there.
3. **Tokens.** Access tokens last an hour and are refreshed shortly before
   they expire; each refresh rotates the refresh token. Everything is stored
   in ``chatgpt_auth.json`` beside the configuration file (``~/.crane_llm/``),
   owner-readable only, written atomically, and refreshed under a file lock.
4. **Requests** go to the public API, ``https://api.openai.com/v1/responses``
   and ``/v1/models``, with the access token as the bearer token. Nothing
   calls ChatGPT's ``backend-api``.

Usage is charged to the user's ChatGPT plan. For Plus the five-hour limit is
shared with every other app using the plan; see https://chatgpt.com/settings/usage.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import secrets
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from .texts import text


ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"

# Only for the first, registering authorize request. Never stored, and never
# used for the token exchange or a refresh.
REGISTRATION_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "CRANE-LLM"

PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
RESOURCE = "https://api.openai.com/v1"

BASE_URL = "https://api.openai.com/v1"
MODELS_URL = f"{BASE_URL}/models"

CALLBACK_HOST = "127.0.0.1"  # the docs require this literal host, not "localhost"
CALLBACK_PATH = "/auth/callback"
# The registering sign-in must use this port; later sign-ins may use any free one.
DEFAULT_CALLBACK_PORT = 1455

AUTH_FILE_NAME = "chatgpt_auth.json"
HOST_ID_FILE_NAME = "host_id"

REFRESH_SKEW_SECONDS = 120
LOGIN_TIMEOUT_SECONDS = 10 * 60

# Refresh errors after which the stored tokens can never work again.
_TERMINAL_REFRESH_ERRORS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}

# Offered when the model list cannot be fetched.
FALLBACK_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]
DEFAULT_MODEL = FALLBACK_MODELS[0]


class ChatGPTError(RuntimeError):
    """A ChatGPT plan request or sign-in failed. Worded for the user."""


class ChatGPTAuthError(ChatGPTError):
    """Signing in, or keeping the session alive, failed."""


# ---------------------------------------------------------------------------
# Storage


def _config_dir() -> Path:
    # Following settings.config_path() means CRANE_LLM_CONFIG moves these too,
    # which keeps the tests away from the developer's real session.
    from .settings import config_path

    return config_path().parent


def auth_path() -> Path:
    return _config_dir() / AUTH_FILE_NAME


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp, path)


def host_id() -> str:
    """This machine's ``ext_agent_host_id``: a UUID made once and kept.

    Opaque and not derived from anything about the user, as the docs require.
    """

    path = _config_dir() / HOST_ID_FILE_NAME
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value.startswith("urn:uuid:"):
            return value
    except OSError:
        pass
    value = f"urn:uuid:{uuid.uuid4()}"
    _write_private(path, value + "\n")
    return value


def load_record() -> Dict[str, Any]:
    """Everything stored about the ChatGPT sign-in, or ``{}``.

    Survives sign-out without its tokens, so the issued client ID is reused
    for the next sign-in instead of registering the tool again.
    """

    try:
        data = json.loads(auth_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_tokens() -> Optional[Dict[str, Any]]:
    """The stored record if it holds a usable session, else ``None``."""

    record = load_record()
    if record.get("client_id") and record.get("access_token") and record.get("refresh_token"):
        return record
    return None


def is_logged_in() -> bool:
    return load_tokens() is not None


def save_record(record: Dict[str, Any]) -> Path:
    path = auth_path()
    _write_private(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def _forget_tokens(record: Dict[str, Any]) -> None:
    """Drop the tokens but keep the registration, for the next sign-in."""

    kept = {k: v for k, v in record.items() if k in ("client_id", "ext_agent_host_id", "email", "subject")}
    save_record(kept)


_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _refresh_lock() -> Iterator[None]:
    """Serialise refreshes: refresh tokens rotate, so two at once sign out one.

    A file lock covers several kernels on one machine; where ``fcntl`` is
    missing (Windows) the in-process lock still covers one kernel.
    """

    with _THREAD_LOCK:
        path = auth_path().with_name(AUTH_FILE_NAME + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:
            yield
            return
        with open(path, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Tokens


def _jwt_claims(token: str) -> Dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _check_id_token(id_token: str, client_id: str, nonce: Optional[str]) -> Dict[str, Any]:
    """The ID token's claims, after checking issuer, audience, expiry and nonce.

    The token comes straight from the token endpoint over TLS, which OpenID
    Connect Core (3.1.3.7, rule 6) accepts in place of checking its signature.
    """

    claims = _jwt_claims(id_token)
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    problems = []
    if claims.get("iss") != ISSUER:
        problems.append("issuer")
    if client_id not in audiences:
        problems.append("audience")
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)) or exp < time.time() - 60:
        problems.append("expiry")
    if nonce is not None and claims.get("nonce") != nonce:
        problems.append("nonce")
    if problems:
        raise ChatGPTAuthError(text("setup.chatgpt.bad_id_token", problems=", ".join(problems)))
    return claims


def _email(claims: Dict[str, Any]) -> Optional[str]:
    email = claims.get("email")
    if isinstance(email, str):
        return email
    profile = claims.get("https://api.openai.com/profile")
    if isinstance(profile, dict) and isinstance(profile.get("email"), str):
        return profile["email"]
    return None


def account_email() -> Optional[str]:
    return load_record().get("email")


def _http():
    # httpx comes with the OpenAI SDK, which is already a dependency.
    import httpx

    return httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_code(response) -> str:
    try:
        body = response.json()
    except Exception:
        return ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "")
        return str(error or body.get("code") or "")
    return ""


def _error_detail(response) -> str:
    """The server's reason for a failed OAuth request, for the error message.

    OAuth errors carry no tokens, so the reason is safe to show.
    """

    code = _error_code(response)
    try:
        body = response.json()
    except Exception:
        body = None
    description = ""
    if isinstance(body, dict):
        error = body.get("error")
        description = str(
            body.get("error_description")
            or (error.get("message") if isinstance(error, dict) else "")
            or body.get("detail")
            or ""
        )
    return ": ".join(part for part in (code, description) if part) or "no reason given"


def _store_token_response(record: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(record)
    updated["access_token"] = data["access_token"]
    if data.get("refresh_token"):
        updated["refresh_token"] = data["refresh_token"]
    if data.get("id_token"):
        updated["id_token"] = data["id_token"]
    updated["token_type"] = data.get("token_type", "Bearer")
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        updated["expires_at"] = int(time.time() + expires_in)
    if data.get("scope"):
        updated["scopes"] = str(data["scope"]).split()
    updated["saved_at"] = _now()
    save_record(updated)
    return updated


def _expiring(record: Dict[str, Any]) -> bool:
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = _jwt_claims(record.get("access_token", "")).get("exp")
    if not isinstance(expires_at, (int, float)):
        return False
    return expires_at - REFRESH_SKEW_SECONDS <= time.time()


def refresh(record: Dict[str, Any]) -> Dict[str, Any]:
    """New access token from the refresh token, stored before it is returned."""

    with _http() as client:
        response = client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": record["client_id"],
                "refresh_token": record["refresh_token"],
                "resource": RESOURCE,
            },
        )
    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get("access_token"):
            return _store_token_response(record, data)
        raise ChatGPTAuthError(text("setup.chatgpt.refresh_failed", status=response.status_code))

    code = _error_code(response)
    if code in _TERMINAL_REFRESH_ERRORS:
        _forget_tokens(record)
        raise ChatGPTAuthError(text("errors.chatgpt_session_expired"))
    if code == "invalid_client":
        raise ChatGPTAuthError(text("errors.chatgpt_invalid_client"))
    # Anything else may be passing: keep the tokens and let the user retry.
    raise ChatGPTAuthError(text("setup.chatgpt.refresh_failed", status=response.status_code))


def access_token() -> str:
    """A valid access token, refreshed first if it is about to expire."""

    record = load_tokens()
    if record is None:
        raise ChatGPTAuthError(text("errors.chatgpt_not_logged_in"))
    if not _expiring(record):
        return record["access_token"]
    with _refresh_lock():
        # Another kernel may have refreshed while this one waited.
        record = load_tokens()
        if record is None:
            raise ChatGPTAuthError(text("errors.chatgpt_not_logged_in"))
        if _expiring(record):
            record = refresh(record)
    return record["access_token"]


# ---------------------------------------------------------------------------
# Sign-in


def _pkce_pair() -> tuple:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


class _CallbackServer:
    """A one-shot HTTP listener on 127.0.0.1 for the OAuth redirect."""

    def __init__(self, port: int):
        result: Dict[str, Any] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server naming)
                parsed = urlparse(self.path)
                if parsed.path != CALLBACK_PATH:
                    self.send_response(404)
                    self.end_headers()
                    return
                result.update({k: v[0] for k, v in parse_qs(parsed.query).items()})
                body = text("setup.chatgpt.browser_done").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep the notebook output clean
                pass

        self.result = result
        self.server = HTTPServer((CALLBACK_HOST, port), Handler)
        self.server.timeout = 1.0
        self.port = self.server.server_address[1]

    @property
    def redirect_uri(self) -> str:
        return f"http://{CALLBACK_HOST}:{self.port}{CALLBACK_PATH}"

    def wait(self, timeout_seconds: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        try:
            while not self.result and time.monotonic() < deadline:
                self.server.handle_request()
        finally:
            self.server.server_close()
        return self.result


def _open_listener(registering: bool) -> _CallbackServer:
    try:
        return _CallbackServer(DEFAULT_CALLBACK_PORT)
    except OSError:
        if registering:
            raise ChatGPTAuthError(text("setup.chatgpt.port_busy", port=DEFAULT_CALLBACK_PORT))
    # Only the port may change after registration.
    return _CallbackServer(0)


def _open_browser(url: str) -> None:
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass


def login(
    output: Callable[[str], None] = print,
    open_browser: Callable[[str], None] = _open_browser,
    timeout_seconds: float = LOGIN_TIMEOUT_SECONDS,
    listener: Optional[Callable[[bool], Any]] = None,
) -> Dict[str, Any]:
    """Sign in through the browser and store the session.

    The first time, this also registers CRANE-LLM for the account. Later
    sign-ins reuse the issued client ID. ``open_browser`` and ``listener`` are
    parameters so the flow can be tested without a browser or a socket.
    """

    from .settings import hosted_platform

    if hosted_platform():
        raise ChatGPTAuthError(text("errors.chatgpt_hosted_unsupported"))

    record = load_record()
    client_id = record.get("client_id")
    registering = not client_id

    server = (listener or _open_listener)(registering)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    ext_host_id = record.get("ext_agent_host_id") or host_id()

    params = {
        "client_id": REGISTRATION_CLIENT_ID if registering else client_id,
        "ext_agent_host_id": ext_host_id,
        "response_type": "code",
        "redirect_uri": server.redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if registering:
        params["agent_name_hint"] = AGENT_NAME
    else:
        if record.get("id_token"):
            params["id_token_hint"] = record["id_token"]
        if record.get("email"):
            params["login_hint"] = record["email"]
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    output(text("setup.chatgpt.browser_instructions", url=url))
    open_browser(url)

    callback = server.wait(timeout_seconds)
    if not callback:
        raise ChatGPTAuthError(text("setup.chatgpt.login_timeout"))
    if not secrets.compare_digest(str(callback.get("state", "")), state):
        raise ChatGPTAuthError(text("setup.chatgpt.bad_callback"))
    if callback.get("error"):
        if callback["error"] == "access_denied":
            raise ChatGPTAuthError(text("setup.chatgpt.access_denied"))
        raise ChatGPTAuthError(
            text("setup.chatgpt.authorize_failed", error=callback.get("error_description") or callback["error"])
        )
    code = callback.get("code")
    if registering:
        client_id = callback.get("client_id")
    if not code or not client_id:
        raise ChatGPTAuthError(text("setup.chatgpt.bad_callback"))

    with _http() as client:
        response = client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": server.redirect_uri,
                "resource": RESOURCE,
            },
        )
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code != 200 or not isinstance(data, dict) or not data.get("access_token"):
        if _error_code(response) == "invalid_client":
            raise ChatGPTAuthError(text("errors.chatgpt_invalid_client"))
        raise ChatGPTAuthError(
            text("setup.chatgpt.exchange_failed", status=response.status_code, detail=_error_detail(response))
        )

    granted = str(data.get("scope") or callback.get("scope") or "").split()
    claims = _check_id_token(data.get("id_token", ""), client_id, nonce)

    new_record = {
        "client_id": client_id,
        "ext_agent_host_id": ext_host_id,
        "issuer": ISSUER,
        "subject": claims.get("sub"),
        "email": _email(claims),
    }
    if not data.get("refresh_token") or PLAN_SCOPE not in granted:
        # Signed in, but plan usage was not granted. Keep the registration so
        # the next attempt reuses it.
        save_record({k: v for k, v in new_record.items() if v})
        raise ChatGPTAuthError(text("setup.chatgpt.plan_not_granted"))
    return _store_token_response(new_record, data)


def logout() -> bool:
    """Revoke the refresh token and forget the tokens. True if signed in.

    The registration (issued client ID, host ID) is kept for the next
    sign-in. A failed revocation does not stop the local sign-out.
    """

    record = load_tokens()
    if record is None:
        return False
    try:
        with _http() as client:
            endpoint = client.get(DISCOVERY_URL).json().get("revocation_endpoint")
            if endpoint:
                client.post(
                    endpoint,
                    data={
                        "token": record["refresh_token"],
                        "token_type_hint": "refresh_token",
                        "client_id": record["client_id"],
                    },
                )
    except Exception:
        pass
    _forget_tokens(record)
    return True


# ---------------------------------------------------------------------------
# Models


def fetch_models() -> Optional[List[str]]:
    """The models this ChatGPT plan may use, or ``None`` if not fetched.

    The plan's ``/v1/models`` lists ``models`` entries with a ``slug`` and a
    ``visibility``; only ``visibility == "list"`` ones are offered.
    """

    try:
        token = access_token()
        with _http() as client:
            response = client.get(MODELS_URL, headers={"Authorization": f"Bearer {token}"})
        body = response.json() if response.status_code == 200 else None
    except Exception:
        return None
    if not isinstance(body, dict):
        return None

    if isinstance(body.get("models"), list):
        slugs = [
            m["slug"]
            for m in body["models"]
            if isinstance(m, dict) and isinstance(m.get("slug"), str) and m.get("visibility", "list") == "list"
        ]
    elif isinstance(body.get("data"), list):
        slugs = [m["id"] for m in body["data"] if isinstance(m, dict) and isinstance(m.get("id"), str)]
    else:
        return None
    return list(dict.fromkeys(slugs)) or None
