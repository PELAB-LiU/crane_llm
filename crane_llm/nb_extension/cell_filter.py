"""Single source of truth for what of the notebook may reach the prompt.

Two things are decided here so the tracker and the prompt builder cannot drift
apart:

- Which cells are the extension's own rather than the user's code. The frontend
  drives the backend by running short helper snippets in the user's kernel, and
  the user drives it with ``%%crane_llm``, ``%crane_llm guard on`` and
  ``%load_ext crane_llm``. All of
  those reach the ``post_run_cell`` hook like any other cell, but none of them
  changed the kernel state, and a ``%%crane_llm`` cell in particular must not be
  listed as having "run successfully": its body was analysed, not executed.
- Which text must never be sent to the model: API keys. A key set with
  ``crane_llm.set_api_key("sk-...")`` or assigned to an environment variable
  sits in an executed cell's source like any other code.
"""

from __future__ import annotations

import ast
import builtins
import re


# A ``%%crane_llm`` cell is analysed, not run, whatever its body holds.
_CELL_MAGIC_RE = re.compile(r"^\s*%%crane_llm\b")

# Line magics that only drive the extension.
_DRIVER_MAGIC_RE = re.compile(r"^\s*%(?:crane_llm\b|(?:load|reload|unload)_ext\s+crane_llm\s*$)")

_SET_API_KEY_CALL_RE = re.compile(r"\bset_api_key\s*\(")

# Key formats of the providers the README documents: OpenAI and OpenRouter
# (``sk-``, ``sk-proj-``, ``sk-or-``), Groq (``gsk_``) and Google (``AIza``).
_KEY_LITERAL_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{30,})"
)

REDACTED = "[REDACTED]"


def is_internal_helper_cell(source: str) -> bool:
    """True for empty cells and for cells that only drive the extension.

    Every statement of the cell must drive the extension: a ``%%crane_llm``
    cell, ``%load_ext crane_llm``, ``%crane_llm guard on``, an import from
    ``crane_llm``, or a call made only with what was imported from it, such as
    ``crane_llm.set_api_key(...)``. A cell that also does anything else, such
    as ``%load_ext crane_llm`` followed by loading the data, is the user's
    code: it changes the kernel state like any other cell.
    """

    if not source or not source.strip():
        return True

    code_lines = [line for line in source.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not code_lines:
        return True
    if _CELL_MAGIC_RE.match(code_lines[0]):
        return True

    python_lines = []
    for line in source.splitlines():
        if _DRIVER_MAGIC_RE.match(line):
            python_lines.append("")
        elif line.lstrip().startswith(("%", "!")):
            # Another magic or a shell command: something the user runs.
            return False
        else:
            python_lines.append(line)

    try:
        tree = ast.parse("\n".join(python_lines))
    except SyntaxError:
        return False

    # Names bound by this cell's imports from crane_llm.
    driver_names = {"crane_llm"}
    return all(_drives_extension(statement, driver_names) for statement in tree.body)


def _is_crane_module(name: str) -> bool:
    return name == "crane_llm" or name.startswith("crane_llm.")


def _drives_extension(statement: ast.stmt, driver_names: set) -> bool:
    """Whether a statement only imports from crane_llm or calls what it imported."""

    if isinstance(statement, ast.Import):
        if not all(_is_crane_module(alias.name) for alias in statement.names):
            return False
        driver_names.update((alias.asname or alias.name).split(".")[0] for alias in statement.names)
        return True
    if isinstance(statement, ast.ImportFrom):
        if statement.level or not _is_crane_module(statement.module or ""):
            return False
        driver_names.update(alias.asname or alias.name for alias in statement.names)
        return True
    if isinstance(statement, ast.Expr):
        # ``print(run_crane_llm_payload(...))``, ``crane_llm.set_api_key(...)``.
        used = {node.id for node in ast.walk(statement) if isinstance(node, ast.Name)}
        return bool(used & driver_names) and all(
            name in driver_names or name in vars(builtins) for name in used
        )
    return False


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
