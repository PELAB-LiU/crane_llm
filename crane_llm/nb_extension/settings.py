"""Where the extension gets its API key, model and endpoint from.

Someone who installed the wheel has no repository and therefore no ``.env`` at
a repository root, which is the only place the batch pipeline ever looked. This
module defines one resolution order that works for both audiences:

1. an explicit argument (``run_crane_llm(model=...)``, ``%%crane_llm gpt-5``)
2. a ``CRANE_LLM_*`` environment variable
3. the provider's own environment variable, e.g. ``OPENAI_API_KEY``
4. the user configuration file, ``~/.crane_llm/config.json``
5. for the API key only: a Kaggle or Colab secret named ``CRANE_LLM_API_KEY``
   or ``OPENAI_API_KEY``, on those platforms
6. the built-in default from ``crane_llm.llms.config_llms``

Step 5 exists because a hosted notebook has no durable home directory, so the
configuration file is gone next session, and the alternative, a key typed into
a cell, is published along with the notebook.

A ``.env`` file is deliberately **not** a step here. ``_load_dotenv_once`` runs
before the lookup and loads it into the environment without overriding what is
already set, so its values are found at step 2 or step 3 under whatever
variable name they were written with. That is what makes an exported shell
variable beat the same name in ``.env``, and makes ``OPENAI_API_KEY`` in
``.env`` beat a key stored in the configuration file. Describing ``.env`` as a
rank below step 3 predicts the wrong winner whenever the file and the shell use
different variable names.

Step 4 is what ``crane_llm.set_api_key(...)`` writes, and is the setup path we
document for users of the installed package: it survives restarts and is not
attached to any one notebook or directory.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


CONFIG_ENV_VAR = "CRANE_LLM_CONFIG"
DEFAULT_CONFIG_DIR = Path.home() / ".crane_llm"
DEFAULT_CONFIG_NAME = "config.json"

# Recognised keys in the configuration file. Anything else is preserved on
# write but ignored on read, so a newer version's settings survive an older
# version touching the file.
_CONFIG_KEYS = ("api_key", "base_url", "model", "api_style", "scan_limit")

# The most values a built-in check may read from the user's data, for checks
# that have to look at every value, such as whether a column converts to int.
# Above it the check is skipped rather than guessed. 0 turns those checks off.
DEFAULT_SCAN_LIMIT = 1_000_000

_DOTENV_LOADED = False


def config_path() -> Path:
    """The user configuration file, overridable with ``CRANE_LLM_CONFIG``."""

    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CONFIG_DIR / DEFAULT_CONFIG_NAME


def _load_dotenv_once() -> None:
    """Pick up ``.env`` from the working directory or an ancestor of it.

    ``python-dotenv`` searches upwards from the caller by default, which finds
    the repository ``.env`` for anyone working inside a checkout. Loaded once
    per process, and never overriding a variable that is already set.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass


def read_config() -> Dict[str, Any]:
    """The user configuration file as a dict, or ``{}`` if unusable.

    A missing file is the normal case before setup. A corrupt one is reported
    rather than swallowed, because silently ignoring it would surface later as
    a confusing "no API key" error.
    """

    path = config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(
            f"{path} is not valid JSON ({exc}). Fix or delete it, then retry."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object, not {type(data).__name__}.")
    return data


def write_config(**values: Optional[str]) -> Path:
    """Merge ``values`` into the user configuration file and return its path.

    Passing ``None`` for a key removes it, so a setting can be unset without
    editing the file by hand. The file is created with owner-only permissions
    where the platform supports it, since it holds an API key.
    """

    path = config_path()
    try:
        current = read_config()
    except RuntimeError:
        # An unreadable file is being replaced wholesale, which is the only way
        # out of a corrupt one without asking the user to delete it manually.
        current = {}

    for key, value in values.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # Windows and some network filesystems do not honour this. The file is
        # still under the user's home directory, so this is a hardening step
        # rather than the only protection.
        pass

    return path


def set_api_key(
    api_key: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_style: Optional[str] = None,
) -> Path:
    """Save credentials to ``~/.crane_llm/config.json``.

    Run once from any notebook cell::

        import crane_llm
        crane_llm.set_api_key("sk-...")

    Any OpenAI-compatible endpoint works by adding ``base_url`` and ``model``,
    for example OpenRouter (which reaches Claude and Gemini through one key) or
    a local Ollama server::

        crane_llm.set_api_key(
            "sk-or-...",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4.5",
        )

    Only the arguments you pass are changed; the rest of the file is left as it
    is. Pass ``""`` to clear one.
    """

    normalised = {}
    for key, value in (
        ("api_key", api_key),
        ("base_url", base_url),
        ("model", model),
        ("api_style", api_style),
    ):
        if value is None:
            continue
        value = value.strip()
        normalised[key] = value or None

    if not normalised:
        raise ValueError("Nothing to save: pass at least one of api_key, base_url, model, api_style.")

    return write_config(**normalised)


def scan_limit() -> int:
    """The most data values a built-in check may read, from the first of:
    ``CRANE_LLM_SCAN_LIMIT``, ``scan_limit`` in the configuration file, or
    ``DEFAULT_SCAN_LIMIT``. Read on every check, so a change applies at once.
    """

    _load_dotenv_once()
    try:
        config_value = read_config().get("scan_limit")
    except RuntimeError:
        config_value = None
    for candidate in (os.environ.get("CRANE_LLM_SCAN_LIMIT"), config_value):
        if candidate is None or candidate == "":
            continue
        try:
            return max(0, int(candidate))
        except (TypeError, ValueError):
            continue
    return DEFAULT_SCAN_LIMIT


def set_scan_limit(limit: Optional[int]) -> Path:
    """Save the most data values a built-in check may read.

    Checks that must look at every value of a column or array -- whether it
    converts to int, whether a target is continuous -- are skipped for data
    larger than this, rather than guessed. ``0`` turns those checks off and
    ``None`` restores the default::

        import crane_llm
        crane_llm.set_scan_limit(5_000_000)
    """

    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError("The scan limit must be a whole number of values, 0 or more, or None.")
    return write_config(scan_limit=limit)


def _first(*candidates: Optional[str]) -> Optional[str]:
    for candidate in candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
    return None


def _is_local(base_url: Optional[str]) -> bool:
    """Whether ``base_url`` points at something running on this machine.

    Local inference servers (Ollama, vLLM, llama.cpp) accept any key or none at
    all, so requiring one would block the setup that needs no account.
    """

    if not base_url:
        return False
    lowered = base_url.lower()
    return any(
        host in lowered
        for host in ("//localhost", "//127.0.0.1", "//0.0.0.0", "//[::1]", "//host.docker.internal")
    )


# Looked up in a hosted notebook's secret store, in this order, when no key was
# found anywhere else.
HOSTED_SECRET_NAMES = ("CRANE_LLM_API_KEY", "OPENAI_API_KEY")

# A secret, once read, is kept for the rest of the process: each read is a
# request to the platform's backend. A miss is not kept, so a secret attached
# to the notebook after the first attempt is still found without a restart.
_HOSTED_SECRET_CACHE: Dict[str, str] = {}


def hosted_platform() -> Optional[str]:
    """``"kaggle"`` or ``"colab"`` when running in one, else ``None``."""

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    return None


def _read_hosted_secret(name: str) -> Optional[str]:
    """One secret from Kaggle Secrets or Colab's Secrets panel, if available.

    Neither module exists outside its platform, and both raise when the secret
    is missing or not shared with this notebook, so every failure means "not
    found" rather than an error of its own.
    """

    platform = hosted_platform()
    try:
        if platform == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        if platform == "colab":
            from google.colab import userdata

            return userdata.get(name)
    except Exception:
        return None
    return None


def hosted_secret_api_key() -> Optional[str]:
    """The API key from the hosted platform's secret store, if one is shared."""

    for name in HOSTED_SECRET_NAMES:
        if name in _HOSTED_SECRET_CACHE:
            return _HOSTED_SECRET_CACHE[name]
        value = _first(_read_hosted_secret(name))
        if value:
            _HOSTED_SECRET_CACHE[name] = value
            return value
    return None


class Settings(NamedTuple):
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    api_style: str  # "responses" or "chat"


def resolve(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_style: Optional[str] = None,
) -> Settings:
    """Combine arguments, environment, ``.env`` and the config file.

    The API style is derived rather than asked for. A bare OpenAI account gets
    the Responses API, which is what the experiments in the paper used. As soon
    as a ``base_url`` is set the default becomes Chat Completions, because
    almost no OpenAI-compatible server implements ``/responses`` -- OpenRouter,
    Groq, Together, vLLM and Ollama all expose ``/chat/completions`` only.
    Azure OpenAI is the exception, hence the explicit override.
    """

    _load_dotenv_once()
    stored = read_config()

    resolved_base_url = _first(
        base_url,
        os.environ.get("CRANE_LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        stored.get("base_url"),
    )

    resolved_api_key = _first(
        api_key,
        os.environ.get("CRANE_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        stored.get("api_key"),
    )
    if resolved_api_key is None and _is_local(resolved_base_url):
        resolved_api_key = "local"
    if resolved_api_key is None:
        resolved_api_key = hosted_secret_api_key()

    from ..llms.config_llms import config

    resolved_model = _first(
        model,
        os.environ.get("CRANE_LLM_MODEL"),
        stored.get("model"),
        config.openai_llm_model,
    )
    assert resolved_model is not None  # config always supplies a default

    resolved_style = _first(
        api_style,
        os.environ.get("CRANE_LLM_API_STYLE"),
        stored.get("api_style"),
    )
    if resolved_style is None:
        resolved_style = "chat" if resolved_base_url else "responses"
    resolved_style = resolved_style.lower()
    if resolved_style not in ("responses", "chat"):
        raise RuntimeError(
            f"Unknown API style {resolved_style!r}. Use 'responses' (OpenAI) or "
            "'chat' (any OpenAI-compatible endpoint)."
        )

    return Settings(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        api_style=resolved_style,
    )


def missing_key_message() -> str:
    """The message shown when no API key could be found anywhere.

    Worth spelling out in full: this is the single most likely thing to go
    wrong for someone who has just installed the wheel, and the sidebar shows
    it verbatim.
    """

    platform = hosted_platform()
    if platform == "kaggle":
        return (
            "No API key found. On Kaggle, store it as a secret:\n"
            "\n"
            "  1. In the notebook editor, open Add-ons > Secrets.\n"
            "  2. Add a secret with the label CRANE_LLM_API_KEY and your key as the value.\n"
            "  3. Tick the checkbox next to it, so this notebook may read it.\n"
            "\n"
            "Then run the %%crane_llm cell again; no restart is needed. Keep the key\n"
            "out of the notebook itself, which others can read once it is shared."
        )
    if platform == "colab":
        return (
            "No API key found. On Colab, store it as a secret:\n"
            "\n"
            "  1. Click the key icon in the left sidebar (Secrets).\n"
            "  2. Add a secret named CRANE_LLM_API_KEY with your key as the value.\n"
            "  3. Switch on 'Notebook access' for it.\n"
            "\n"
            "Then run the %%crane_llm cell again; no restart is needed."
        )

    return (
        "No API key found. Set one up in any of these ways, then retry.\n"
        "\n"
        "  1. From a notebook cell, saved to {path} for future sessions:\n"
        "         import crane_llm\n"
        '         crane_llm.set_api_key("sk-...")\n'
        "\n"
        "  2. As an environment variable before starting Jupyter:\n"
        "         CRANE_LLM_API_KEY=sk-...\n"
        "\n"
        "  3. In a .env file in the directory you started Jupyter from:\n"
        "         CRANE_LLM_API_KEY=sk-...\n"
        "\n"
        "Any OpenAI-compatible provider works. For one that is not OpenAI, add a\n"
        "base_url and a model, for example OpenRouter (Claude, Gemini and others\n"
        "through a single key):\n"
        '         crane_llm.set_api_key("sk-or-...",\n'
        '                               base_url="https://openrouter.ai/api/v1",\n'
        '                               model="anthropic/claude-sonnet-4.5")\n'
        "\n"
        "A local server needs no key at all, only a base_url:\n"
        '         crane_llm.set_api_key(base_url="http://localhost:11434/v1",\n'
        '                               model="qwen2.5-coder:32b")'
    ).format(path=config_path())
