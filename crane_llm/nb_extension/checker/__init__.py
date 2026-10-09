"""The built-in checker: crashes it can report as certain, without the LLM.

``walker.py`` goes through the target cell against the live kernel state and
offers each operation to the rules in ``rules/``, which are written with what
``sites.py`` provides. ``pure.py`` lists the operations the walker may go
past without losing track of the values.
"""

from .walker import CheckFinding, run_checks

__all__ = ["CheckFinding", "run_checks"]
