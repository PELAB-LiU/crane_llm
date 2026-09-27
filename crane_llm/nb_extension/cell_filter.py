"""Single source of truth for what of the notebook may reach the prompt.

Two things are decided here so the tracker and the prompt builder cannot drift
apart:

- Which cells are the extension's own rather than the user's code. The frontend
  drives the backend by running short helper snippets in the user's kernel, and
  the user drives it with ``%%crane_llm`` and ``%load_ext crane_llm``. All of
  those reach the ``post_run_cell`` hook like any other cell, but none of them
  changed the kernel state, and a ``%%crane_llm`` cell in particular must not be
  listed as having "run successfully": its body was analysed, not executed.
- Which text must never be sent to the model: API keys. A key set with
  ``crane_llm.set_api_key("sk-...")`` or assigned to an environment variable
  sits in an executed cell's source like any other code.
"""

from __future__ import annotations

import re


# Matched against the *first* statement of a cell rather than anywhere in it, so
# that ordinary user code merely mentioning the extension is still treated as
# notebook content.
_INTERNAL_CELL_RE = re.compile(
    r"^\s*(?:"
    r"from\s+crane_llm(?:\.\w+)*\s+import\b"
    r"|import\s+crane_llm\b"
    r"|crane_llm\.\w+"
    r"|%%crane_llm\b"
    r"|%(?:load|reload|unload)_ext\s+crane_llm\b"
    r")",
)

_SET_API_KEY_CALL_RE = re.compile(r"\bset_api_key\s*\(")

# Key formats of the providers the README documents: OpenAI and OpenRouter
# (``sk-``, ``sk-proj-``, ``sk-or-``), Groq (``gsk_``) and Google (``AIza``).
_KEY_LITERAL_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{30,})"
)

REDACTED = "[REDACTED]"


def is_internal_helper_cell(source: str) -> bool:
    """True for empty cells and for cells that only drive the extension."""

    if not source or not source.strip():
        return True

    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return bool(_INTERNAL_CELL_RE.match(line))

    return True


def redact_secrets(text: str) -> str:
    """Remove API keys from text that is about to be sent to the model.

    ``set_api_key(...)`` calls are dropped whole, including arguments continued
    on later lines, because a key need not follow any recognisable format.
    Anything else that looks like a key is replaced by ``[REDACTED]``.
    """

    kept = []
    depth = 0
    for line in text.splitlines(keepends=True):
        if depth > 0:
            depth += line.count("(") - line.count(")")
            continue

        match = _SET_API_KEY_CALL_RE.search(line)
        if match is None:
            kept.append(line)
            continue

        tail = line[match.start():]
        depth = tail.count("(") - tail.count(")")
        indent = line[: len(line) - len(line.lstrip())]
        newline = "\n" if line.endswith("\n") else ""
        kept.append(f"{indent}# set_api_key(...) removed by CRANE-LLM{newline}")

    return _KEY_LITERAL_RE.sub(REDACTED, "".join(kept))
