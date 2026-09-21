"""Assert that `jupyter labextension list` reports crane-llm-jlab as enabled.

Used by both workflows to check that a freshly installed wheel is actually
discovered by JupyterLab, which is the one thing a wheel can get wrong while
still installing perfectly.

This is a script rather than a `grep` in the workflow because the output is
harder to match than it looks. `jupyter_server` colours its status words on
POSIX::

    GREEN_ENABLED = "\\033[32menabled\\033[0m" if os.name != "nt" else "enabled"
    GREEN_OK      = "\\033[32mOK\\033[0m"      if os.name != "nt" else "ok"

so on Linux the bytes read ``...enabled\\033[0m \\033[32mOK...``: the phrase
"enabled OK" is never contiguous, and a grep for it cannot match however the
casing is handled. GitHub's log viewer renders the escapes away, so the log
shows a tidy "enabled OK" and the failure looks like a casing problem instead.
On Windows the same command emits plain lowercase "enabled ok", so a developer
checking locally sees neither issue.

Stripping the escapes first removes both traps, and failing here prints the
listing so the reason is visible in the log.
"""

from __future__ import annotations

import re
import sys

EXTENSION = "crane-llm-jlab"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ENABLED_OK = re.compile(r"enabled\s+ok", re.IGNORECASE)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to captured listing>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
        listing = ANSI.sub("", handle.read())

    matches = [line for line in listing.splitlines() if EXTENSION in line]

    if not matches:
        print(f"FAIL: {EXTENSION} does not appear in the listing at all.")
        print("The wheel installed but JupyterLab did not find the frontend bundle.")
        print("--- listing ---")
        print(listing)
        return 1

    line = matches[0].strip()
    print(f"found: {line}")

    # "disabled" does not contain "enabled", so this catches an extension that
    # is present but switched off, which would otherwise read as a pass.
    if "disabled" in line or not ENABLED_OK.search(line):
        print(f"FAIL: {EXTENSION} is listed but not reported as enabled ok.")
        print("--- listing ---")
        print(listing)
        return 1

    print(f"OK: {EXTENSION} is installed and enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
