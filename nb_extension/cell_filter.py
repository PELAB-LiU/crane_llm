"""Single source of truth for cells the extension must not treat as user code.

The frontend drives the backend by running short helper snippets in the user's
kernel. Those executions reach the ``post_run_cell`` hook like any other cell,
so they have to be filtered out of the prompt. The filter lives here so the
tracker and the prompt builder cannot drift apart.
"""

from __future__ import annotations

import re


# Matched against the *first* statement of a cell rather than anywhere in it, so
# that ordinary user code merely mentioning the extension is still treated as
# notebook content.
_INTERNAL_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+nb_extension(?:\.\w+)*\s+import\b|import\s+nb_extension\b)",
)


def is_internal_helper_cell(source: str) -> bool:
    """True for empty cells and for the helper snippets the frontend injects."""

    if not source or not source.strip():
        return True

    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return bool(_INTERNAL_IMPORT_RE.match(line))

    return True
