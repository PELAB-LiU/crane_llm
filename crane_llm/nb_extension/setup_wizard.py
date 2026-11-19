"""``crane_llm.setup()``: choose a model provider interactively.

Asks which provider to use, then only what that provider needs (an API key, a
server address, a model, or a ChatGPT sign-in), and saves the answers to the
configuration file that ``settings.resolve()`` reads. Runs in any notebook
cell or Python prompt; the key is read with ``getpass`` so it is neither echoed
nor left in the cell's source.

Every prompt can be answered in advance with an argument, so
``setup("ollama", model="qwen2.5-coder:32b")`` asks nothing.
"""

from __future__ import annotations

import getpass as getpass_module
import re
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from . import settings
from .texts import text


class Provider(NamedTuple):
    id: str
    label: str
    base_url: Optional[str]
    # Offered first in the model list, best first. Shown as they are when the
    # provider's own list cannot be fetched; otherwise only those it has.
    suggested: Tuple[str, ...]
    # "required", "optional" or "none"
    key: str
    # Ask for the server address, offering ``base_url`` as the default.
    ask_base_url: bool = False

    @property
    def default_model(self) -> str:
        return self.suggested[0] if self.suggested else ""


# The providers of the user guide's "Set up a model" section, plus the ChatGPT
# subscription and a catch-all for any other OpenAI-compatible endpoint.
# The suggestions date quickly; the live list fetched from each provider is
# what keeps the picker current.
PROVIDERS: List[Provider] = [
    Provider("openai", "OpenAI (API key)", None,
             ("gpt-5", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5-mini"), "required"),
    Provider("chatgpt", "ChatGPT plan: Continue with ChatGPT (local JupyterLab only, experimental)", None,
             (), "none"),
    Provider("openrouter", "OpenRouter (Claude, Gemini, GPT, DeepSeek, ...)", "https://openrouter.ai/api/v1",
             ("anthropic/claude-sonnet-5.5", "openai/gpt-5.5", "google/gemini-3.1-pro-preview",
              "google/gemini-3.8-flash", "deepseek/deepseek-v4-pro", "anthropic/claude-haiku-5.5"), "required"),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
             ("gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro"), "required"),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1",
             ("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"), "required"),
    Provider("ollama", "Ollama (local model, no key)", "http://localhost:11434/v1",
             ("qwen2.5-coder:32b",), "none", ask_base_url=True),
    Provider("custom", "Other OpenAI-compatible endpoint", None, (), "optional", ask_base_url=True),
]

_BY_ID: Dict[str, Provider] = {p.id: p for p in PROVIDERS}

# Up to this many models are listed in full; a longer list (OpenRouter has
# hundreds) shows the suggestions and offers to search the rest.
SHOW_ALL_LIMIT = 20

# Models in a provider's /models list that cannot answer a chat prompt.
_NOT_CHAT = re.compile(
    r"embed|tts|whisper|dall-e|gpt-image|imagen|image-|moderation|audio|realtime|"
    r"transcribe|search|davinci|babbage|computer-use|sora|rerank|guard|veo|aqa|"
    r"text-to|speech|-vision-preview$",
    re.IGNORECASE,
)


class _Prompter:
    """Input and output, swappable so the wizard can be tested."""

    def __init__(self, input_fn, getpass_fn, output_fn):
        self.input = input_fn
        self.getpass = getpass_fn
        self.output = output_fn

    def ask(self, question: str, default: Optional[str] = None) -> str:
        suffix = f" [{default}]" if default else ""
        answer = self.input(f"{question}{suffix}: ").strip()
        return answer or (default or "")

    def yes(self, question: str, default: bool) -> bool:
        hint = "Y/n" if default else "y/N"
        answer = self.input(f"{question} [{hint}]: ").strip().lower()
        if not answer:
            return default
        return answer.startswith(("y", "s"))  # "sí" works too

    def choose(self, question: str, options: List[str], default_index: int = 0) -> int:
        for number, option in enumerate(options, start=1):
            self.output(f"  {number}. {option}")
        while True:
            answer = self.ask(question, str(default_index + 1))
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            if answer in options:
                return options.index(answer)
            self.output(text("setup.invalid_choice", max=len(options)))


def _stored() -> Dict[str, object]:
    try:
        return settings.read_config()
    except RuntimeError:
        return {}


def _current_provider(stored: Dict[str, object]) -> Optional[Provider]:
    """The provider the configuration file is set up for, if it can be told."""

    name = stored.get("provider")
    if isinstance(name, str) and name in _BY_ID:
        return _BY_ID[name]
    if stored.get("api_style") == settings.CHATGPT_STYLE:
        return _BY_ID["chatgpt"]
    base_url = stored.get("base_url")
    if base_url:
        for provider in PROVIDERS:
            if provider.base_url and provider.base_url == base_url:
                return provider
        return _BY_ID["custom"]
    if stored.get("api_key"):
        return _BY_ID["openai"]
    return None


def _describe_current(stored: Dict[str, object]) -> Optional[str]:
    provider = _current_provider(stored)
    if provider is None:
        return None
    model = stored.get("model") or provider.default_model or "-"
    return text("setup.current", provider=provider.label, model=model)


def _setup_chatgpt(prompt: _Prompter, model: Optional[str], relogin: Optional[bool]) -> Dict[str, Optional[str]]:
    from . import chatgpt_auth

    prompt.output(text("setup.chatgpt.warning"))

    signed_in = chatgpt_auth.is_logged_in()
    if signed_in and relogin is None:
        who = chatgpt_auth.account_email() or "?"
        relogin = prompt.yes(text("setup.chatgpt.already_signed_in", account=who), default=False)
    if not signed_in or relogin:
        chatgpt_auth.login(output=prompt.output)
        prompt.output(text("setup.chatgpt.signed_in", account=chatgpt_auth.account_email() or "?"))

    if model is None:
        model = _pick_model(prompt, _BY_ID["chatgpt"], None, None, current=None)

    # The stored API key, if any, is kept: the "chatgpt" style ignores it, and
    # it is still there when switching back to that provider.
    return {
        "provider": "chatgpt",
        "api_style": settings.CHATGPT_STYLE,
        "base_url": None,
        "model": model,
    }


def _setup_with_key(
    prompt: _Prompter,
    provider: Provider,
    stored: Dict[str, object],
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> Dict[str, Optional[str]]:
    same_provider = _current_provider(stored) == provider

    if base_url is None:
        if provider.ask_base_url:
            default = (stored.get("base_url") if same_provider else None) or provider.base_url
            while True:
                base_url = prompt.ask(text("setup.ask_base_url"), default)
                if base_url:
                    break
                prompt.output(text("setup.base_url_required"))
        else:
            base_url = provider.base_url

    existing_key = stored.get("api_key") if same_provider else None
    if api_key is None and provider.key != "none":
        while True:
            question = text("setup.ask_key_keep") if existing_key else text("setup.ask_key")
            api_key = prompt.getpass(question + ": ").strip()
            if api_key or existing_key or provider.key == "optional":
                break
            prompt.output(text("setup.key_required"))
        api_key = api_key or existing_key or None

    if model is None:
        current = stored.get("model") if same_provider else None
        model = _pick_model(prompt, provider, api_key, base_url, current=current if isinstance(current, str) else None)

    return {
        "provider": provider.id,
        "api_key": api_key if provider.key != "none" else None,
        "base_url": base_url or None,
        "model": model,
        # Derived from base_url by settings.resolve(): Responses for OpenAI,
        # Chat Completions for everything else.
        "api_style": None,
    }


def _list_models(
    provider: Provider, api_key: Optional[str], base_url: Optional[str]
) -> Tuple[Optional[List[str]], Optional[str]]:
    """The provider's own model list, as ``(models, None)``, or ``(None, why)``.

    Every supported provider serves the OpenAI ``GET /models`` endpoint;
    the ChatGPT backend has its own catalogue. Models that cannot chat
    (embeddings, speech, images) are left out.
    """

    if provider.id == "chatgpt":
        from . import chatgpt_auth

        models = chatgpt_auth.fetch_models()
        return (models, None) if models else (None, "catalogue unavailable")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or "none",
            base_url=base_url or None,
            timeout=10.0,
            max_retries=0,
        )
        ids = [model.id for model in client.models.list()]
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        return None, (f"HTTP {status}" if status else type(exc).__name__)

    # Gemini names its models "models/gemini-...", but expects them without the prefix.
    ids = [i[len("models/"):] if i.startswith("models/") else i for i in ids if isinstance(i, str)]
    models = sorted({i for i in ids if i and not _NOT_CHAT.search(i)})
    return (models, None) if models else (None, "empty list")


def _pick_model(
    prompt: _Prompter,
    provider: Provider,
    api_key: Optional[str],
    base_url: Optional[str],
    current: Optional[str],
) -> str:
    """A numbered list of models to choose from.

    The suggestions come first, then the rest of what the provider lists. A
    long list (OpenRouter has hundreds) shows only the suggestions, with an
    entry to search the others. A name can always be typed instead.
    """

    prompt.output(text("setup.fetching_models", provider=provider.label.split(" (")[0].split(":")[0]))
    live, error = _list_models(provider, api_key, base_url)

    if provider.id == "chatgpt":
        from . import chatgpt_auth

        suggested = list(chatgpt_auth.FALLBACK_MODELS)
    else:
        suggested = list(provider.suggested)

    if live is None:
        prompt.output(text("setup.models_unavailable", error=error))
        head, rest = suggested, []
    elif provider.id == "chatgpt":
        # Already ranked by the backend.
        head, rest = list(live), []
    else:
        head = [m for m in suggested if m in live]
        rest = [m for m in live if m not in head]

    if current and current not in head and (live is None or current in live):
        head.insert(0, current)
        rest = [m for m in rest if m != current]
    if len(head) + len(rest) <= SHOW_ALL_LIMIT:
        head, rest = head + rest, []

    while True:
        options = list(head)
        if rest:
            options.append(text("setup.show_all_models", count=len(head) + len(rest)))
        options.append(text("setup.other_model"))
        default_index = head.index(current) if current in head else 0

        prompt.output("")
        index = prompt.choose(text("setup.choose_model"), options, default_index)
        if index < len(head):
            return head[index]

        if rest and index == len(head):
            pattern = prompt.ask(text("setup.filter_models")).lower()
            matches = [m for m in head + rest if pattern in m.lower()]
            if not matches:
                prompt.output(text("setup.no_match", pattern=pattern))
                continue
            prompt.output("")
            choice = prompt.choose(text("setup.choose_model"), matches + [text("setup.back")])
            if choice < len(matches):
                return matches[choice]
            continue

        while True:
            name = prompt.ask(text("setup.ask_model_name"))
            if name:
                return name
            prompt.output(text("setup.model_required"))


def setup(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    relogin: Optional[bool] = None,
    test: Optional[bool] = None,
    input: Callable[[str], str] = input,
    getpass: Callable[[str], str] = getpass_module.getpass,
    output: Callable[[str], None] = print,
):
    """Choose the model provider for CRANE-LLM and save it.

    Run once in any notebook cell::

        import crane_llm
        crane_llm.setup()

    It lists the supported providers, then asks only what the chosen one
    needs. The answers are saved to ``~/.crane_llm/config.json`` and used by
    every notebook from then on. Run it again at any time to switch.

    ``provider`` is one of ``openai``, ``chatgpt``, ``openrouter``,
    ``gemini``, ``groq``, ``ollama`` or ``custom``. Any argument given is not
    asked for. ``test`` sends one short request at the end to check the setup
    works (asked if not given). ``input``, ``getpass`` and ``output`` replace
    the console, for tests.
    """

    prompt = _Prompter(input, getpass, output)
    stored = _stored()

    output(text("setup.title"))
    current = _describe_current(stored)
    if current:
        output(current)
    if settings.hosted_platform():
        output(text("setup.hosted_note"))

    # The ChatGPT sign-in needs a browser on the kernel's own machine, which a
    # hosted notebook never has.
    offered = [p for p in PROVIDERS if not (p.id == "chatgpt" and settings.hosted_platform())]

    if provider is None:
        output("")
        current_provider = _current_provider(stored)
        default_index = offered.index(current_provider) if current_provider in offered else 0
        chosen = offered[prompt.choose(text("setup.choose_provider"), [p.label for p in offered], default_index)]
    else:
        chosen = _BY_ID.get(provider.strip().lower())
        if chosen is None:
            raise ValueError(text("setup.unknown_provider", provider=provider, known=", ".join(_BY_ID)))
        if chosen not in offered:
            raise ValueError(text("errors.chatgpt_hosted_unsupported"))

    if chosen.id == "chatgpt":
        values = _setup_chatgpt(prompt, model, relogin)
    else:
        values = _setup_with_key(prompt, chosen, stored, api_key, model, base_url)

    path = settings.write_config(**values)
    output(text("setup.saved", provider=chosen.label, model=values["model"], path=path))

    if test is None:
        test = prompt.yes(text("setup.ask_test"), default=True)
    if test:
        _test_connection(output)
    return path


def _test_connection(output: Callable[[str], None]) -> bool:
    """One short request with the saved settings. Reports, never raises."""

    from .assistant import describe_error, key_rejected
    from .llm_client import default_client

    resolved = settings.resolve()
    output(text("setup.testing", model=resolved.model))
    try:
        client = default_client(include_runinfo=False, resolved=resolved)
        client.run("# Executed Cells:\n\n# Target Cell:\nx = 1\n")
    except Exception as exc:
        rejected = key_rejected(exc, resolved)
        output(text("setup.test_failed", error=describe_error(rejected or exc)))
        return False
    output(text("setup.test_ok"))
    return True


def logout_chatgpt() -> bool:
    """Forget the stored ChatGPT session. The provider choice is left as is."""

    from . import chatgpt_auth

    return chatgpt_auth.logout()
