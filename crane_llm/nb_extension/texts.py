"""The texts users see, read from ``src/ui_texts.json``.

That one file holds every user-facing text of both the JupyterLab frontend,
which imports it when it is built, and the backend and ``%%crane_llm`` magic,
which read it here. Edit the wording there, not in code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

TEXTS_PATH = Path(__file__).with_name("src") / "ui_texts.json"

_TEXTS: Dict[str, Any] = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def text(key: str, **values: Any) -> str:
    """The text at a dotted ``key``, with its ``{placeholders}`` filled in.

    Mirrors ``t`` in src/index.ts. A placeholder with no value is left as
    written, so a typo in the file shows up on screen instead of raising.
    """

    node: Any = _TEXTS
    for part in key.split("."):
        node = node[part]
    return _PLACEHOLDER.sub(
        lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), node
    )
