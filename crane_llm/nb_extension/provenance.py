"""Which cells produced a variable's current state.

When a crash is predicted, the cell under analysis is rarely where the
mistake was made. ``df.head()`` fails because some earlier cell ran
``df = df.dropna(inplace=True)``; ``model.predict(X_test)`` fails because the
cell that built ``X_test`` dropped a column. Notebooks make that earlier cell
hard to find, since cells run in any order and any number of times, and the
``.ipynb`` file records none of it.

The kernel does see it. Around every cell this module records, for the names
the cell mentions, whether the cell

- **assigned** the name: it now refers to a different object,
- **modified** it in place: same object, but its shape, columns, dtypes,
  length or fitted state changed, or
- **possibly modified** it: the code mutates it (``df[c] = ...``,
  ``lst.append(...)``, ``inplace=True``) but no change was visible in those
  summaries,
- **deleted** it.

``locate_origins`` then answers, for each variable a verdict blames, which
cells made it what it is: the last cell that assigned it and every cell that
changed it since.

Cells that ran before tracking started are known only from IPython's input
history, so for those the writes are read from the code alone.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .texts import text


# IPython's own output-caching names, which change on every cell, and dunder
# names such as ``__builtins__``, which no user variable is.
_IPYTHON_NAME_RE = re.compile(r"^(?:_+|_i+|_i?\d+|_oh|_ih|_dh|In|Out|exit|quit|get_ipython|__\w+__)$")

# Method calls that change their receiver.
_MUTATING_METHODS = frozenset(
    {
        "append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault",
        "popitem", "sort", "reverse", "add", "discard", "fit", "partial_fit",
        "fit_transform", "fit_predict", "set_params", "add_module", "load_state_dict",
        "compile", "train", "eval", "to", "cuda", "cpu", "apply_", "fill_", "zero_",
        "resize", "itemset", "put",
    }
)

_MAX_EVENTS = 2000


@dataclass
class CellEvent:
    """What one execution of one cell did to the namespace."""

    cell_id: str
    source: str
    execution_count: Optional[int]
    order: int
    succeeded: bool
    # name -> line in ``source`` where the cell writes it (1-based, or 0)
    assigned: Dict[str, int] = field(default_factory=dict)
    modified: Dict[str, int] = field(default_factory=dict)
    possibly_modified: Dict[str, int] = field(default_factory=dict)
    deleted: Dict[str, int] = field(default_factory=dict)
    # True when the writes were read from the code alone, because the cell ran
    # before tracking started.
    from_code_only: bool = False


@dataclass
class _Pending:
    names: Set[str]
    keys_before: Set[str]
    snapshot: Dict[str, Tuple[int, Any]]


class ProvenanceLog:
    """Execution-ordered record of which cell wrote which variable."""

    def __init__(self):
        self.events: List[CellEvent] = []
        # The latest run of every cell, including runs that wrote nothing, so
        # a cell that raised before its assignment is known to have run.
        self.last_runs: Dict[str, CellEvent] = {}
        self._pending: Optional[_Pending] = None
        self._order = 0

    def reset(self) -> None:
        self.events.clear()
        self.last_runs.clear()
        self._pending = None
        self._order = 0

    # --- recording around a cell ---------------------------------------

    def before_cell(self, source: str, namespace: Dict[str, Any]) -> None:
        names = mentioned_names(source)
        snapshot = {}
        for name in names:
            if name in namespace:
                value = namespace[name]
                snapshot[name] = (id(value), fingerprint(value))
        self._pending = _Pending(names=names, keys_before=set(namespace), snapshot=snapshot)

    def after_cell(
        self,
        source: str,
        cell_id: str,
        execution_count: Optional[int],
        succeeded: bool,
        namespace: Dict[str, Any],
    ) -> Optional[CellEvent]:
        pending, self._pending = self._pending, None
        if pending is None:
            return None

        writes = static_writes(source)
        keys_after = set(namespace)
        assigned: Dict[str, int] = {}
        modified: Dict[str, int] = {}
        deleted: Dict[str, int] = {}

        for name in pending.names | (keys_after - pending.keys_before):
            if _IPYTHON_NAME_RE.match(name):
                continue
            before = pending.snapshot.get(name)
            if name not in namespace:
                if before is not None:
                    deleted[name] = writes.line(name)
                continue
            value = namespace[name]
            if before is None or before[0] != id(value):
                assigned[name] = writes.line(name)
                continue
            after_print = fingerprint(value)
            if before[1] is not None and after_print is not None and before[1] != after_print:
                modified[name] = writes.line(name)

        for name in pending.keys_before - keys_after:
            if not _IPYTHON_NAME_RE.match(name):
                deleted.setdefault(name, writes.line(name))

        possibly = {
            name: line
            for name, line in writes.mutations.items()
            if name in namespace and name not in assigned and name not in modified
        }

        event = CellEvent(
            cell_id=cell_id,
            source=source,
            execution_count=execution_count,
            order=0,
            succeeded=succeeded,
            assigned=assigned,
            modified=modified,
            possibly_modified=possibly,
            deleted=deleted,
        )
        self.last_runs[cell_id] = event
        if not (assigned or modified or possibly or deleted):
            return None
        return self._append(event)

    def record_from_code(self, source: str, cell_id: str, execution_count: Optional[int]) -> None:
        """Record a cell known only from history, reading its writes from the code."""

        writes = static_writes(source)
        event = CellEvent(
            cell_id=cell_id,
            source=source,
            execution_count=execution_count,
            order=0,
            succeeded=True,
            assigned=dict(writes.stores),
            possibly_modified={n: l for n, l in writes.mutations.items() if n not in writes.stores},
            deleted=dict(writes.deletes),
            from_code_only=True,
        )
        self.last_runs[cell_id] = event
        if writes.stores or writes.mutations or writes.deletes:
            self._append(event)

    def _append(self, event: CellEvent) -> CellEvent:
        self._order += 1
        event.order = self._order
        self.events.append(event)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]
        return event


# --- locating origins -----------------------------------------------------


@dataclass
class OriginStep:
    """One cell that contributed to a variable's current state."""

    cell_id: str
    execution_count: Optional[int]
    # assigned | modified | possibly_modified | deleted | defines. Shown to
    # users as the words under origins.roles in ui_texts.json.
    role: str
    line: int
    line_text: str
    source: str
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "execution_count": self.execution_count,
            "role": self.role,
            "line": self.line,
            "line_text": self.line_text,
            "source": self.source,
            "note": self.note,
        }


@dataclass
class Origin:
    variable: str
    steps: List[OriginStep]
    summary: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "summary": self.summary,
            "steps": [step.to_json() for step in self.steps],
        }


def locate_origins(
    variables: Iterable[str],
    log: ProvenanceLog,
    namespace: Dict[str, Any],
    notebook_cells: Optional[List[Dict[str, str]]] = None,
    target_cell_id: Optional[str] = None,
) -> List[Origin]:
    """For each variable, the cells that made it what it is now.

    ``notebook_cells`` is every code cell of the notebook as ``{"id", "source"}``,
    when the frontend can supply them. It is what lets a variable that does
    not exist be traced to a cell that would define it but has not run.
    """

    origins = []
    for variable in dict.fromkeys(variables):
        if not variable or not variable.isidentifier():
            continue
        steps = _trace(variable, log)
        if variable not in namespace and not any(s.role == "deleted" for s in steps):
            steps = _definers(variable, log, notebook_cells or [], target_cell_id)
        _name_by_last_run(steps, log)
        if steps:
            origins.append(Origin(variable=variable, steps=steps, summary=_summarise(variable, steps)))
    return origins


def _trace(variable: str, log: ProvenanceLog) -> List[OriginStep]:
    """The last assignment or deletion of ``variable`` and every change after it."""

    steps: List[OriginStep] = []
    for event in reversed(log.events):
        if variable in event.deleted:
            steps.append(_step(event, "deleted", event.deleted[variable]))
            break
        if variable in event.assigned:
            note = text("origins.notes.code_only") if event.from_code_only else ""
            steps.append(_step(event, "assigned", event.assigned[variable], note))
            break
        if variable in event.modified:
            steps.append(_step(event, "modified", event.modified[variable]))
        elif variable in event.possibly_modified:
            steps.append(_step(event, "possibly_modified", event.possibly_modified[variable]))
    steps.reverse()
    return _merge_reruns(steps)


_CHANGES = ("modified", "possibly_modified")


def _merge_reruns(steps: List[OriginStep]) -> List[OriginStep]:
    """One step per cell: a cell run three times is one cell.

    Runs that changed the variable merge whether the change was visible or
    not, as "modified" if any run visibly changed it. The latest run is kept,
    where it falls in the order, since that run is the one that left the
    variable as it is.
    """

    runs: Dict[Tuple[str, str], int] = {}
    latest: Dict[Tuple[str, str], OriginStep] = {}
    visible: Dict[Tuple[str, str], bool] = {}
    for step in steps:
        key = (step.cell_id, "changed" if step.role in _CHANGES else step.role)
        runs[key] = runs.get(key, 0) + 1
        visible[key] = visible.get(key, False) or step.role == "modified"
        latest.pop(key, None)
        latest[key] = step
    merged = []
    for key, step in latest.items():
        if step.role in _CHANGES:
            step.role = "modified" if visible[key] else "possibly_modified"
        if runs[key] > 1:
            count = text("origins.notes.runs", count=runs[key])
            step.note = f"{step.note}; {count}" if step.note else count
        merged.append(step)
    return merged


def _name_by_last_run(steps: List[OriginStep], log: "ProvenanceLog") -> None:
    """Give every step the execution count of its cell's last run.

    That is the number the notebook shows next to the cell, also when the
    cell has run again since without changing the variable.
    """

    for step in steps:
        run = log.last_runs.get(step.cell_id)
        if run is not None and run.execution_count:
            step.execution_count = run.execution_count


def _definers(variable, log, notebook_cells, target_cell_id) -> List[OriginStep]:
    """Cells of the notebook that would define a variable that does not exist."""

    ran = log.last_runs
    steps = []
    for cell in notebook_cells:
        cell_id, source = cell.get("id", ""), cell.get("source", "")
        if cell_id == target_cell_id:
            continue
        writes = static_writes(source)
        if variable not in writes.stores:
            continue
        event = ran.get(cell_id)
        if event is None:
            note = text("origins.notes.not_run")
        elif not event.succeeded:
            note = text("origins.notes.raised_before")
        else:
            note = text(
                "origins.notes.edited" if event.source != source else "origins.notes.ran_no_define"
            )
        steps.append(
            OriginStep(
                cell_id=cell_id,
                execution_count=event.execution_count if event else None,
                role="defines",
                line=writes.stores[variable],
                line_text=_line(source, writes.stores[variable]),
                source=source,
                note=note,
            )
        )
    return steps


def _step(event: CellEvent, role: str, line: int, note: str = "") -> OriginStep:
    if not event.succeeded and not note:
        note = text("origins.notes.partial_run")
    return OriginStep(
        cell_id=event.cell_id,
        execution_count=event.execution_count,
        role=role,
        line=line,
        line_text=_line(event.source, line),
        source=event.source,
        note=note,
    )


def _summarise(variable: str, steps: List[OriginStep]) -> str:
    if steps[0].role == "defines":
        if len(steps) == 1:
            return text("origins.summary_undefined_one", variable=variable)
        return text("origins.summary_undefined_many", variable=variable, count=len(steps))

    def where(step: OriginStep) -> str:
        if step.execution_count:
            return text("origins.cell_with_count", count=step.execution_count)
        return text("origins.cell_unknown")

    parts = [
        text("origins.summary_step", role=text(f"origins.roles.{step.role}"), cell=where(step))
        for step in steps
    ]
    return text("origins.summary", variable=variable, steps=text("origins.summary_join").join(parts))


def _line(source: str, line: int) -> str:
    lines = source.splitlines()
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


# --- reading writes from code ---------------------------------------------


@dataclass
class _Writes:
    stores: Dict[str, int] = field(default_factory=dict)
    mutations: Dict[str, int] = field(default_factory=dict)
    deletes: Dict[str, int] = field(default_factory=dict)

    def line(self, name: str) -> int:
        for table in (self.stores, self.mutations, self.deletes):
            if name in table:
                return table[name]
        return 0


def _parse(source: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source)
    except SyntaxError:
        pass
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            return ast.parse(shell.transform_cell(source))
    except Exception:
        pass
    return None


def mentioned_names(source: str) -> Set[str]:
    """Every name a cell mentions, read or written, at any depth."""

    tree = _parse(source)
    if tree is None:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _root(node: ast.AST) -> Optional[str]:
    """The variable an attribute or subscript chain starts from.

    A chain through a call has no root: ``LinearRegression().fit(...)``
    changes the new object the call returned, not ``LinearRegression``.
    """

    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def static_writes(source: str) -> _Writes:
    """Names a cell's top-level code assigns, mutates or deletes, with lines.

    Function and class bodies are skipped: their code runs when called, not
    when the cell runs. The last write in the cell wins, since that is the one
    that determined the value.
    """

    writes = _Writes()
    tree = _parse(source)
    if tree is None:
        return writes

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            writes.stores[node.name] = node.lineno
            for decorator in node.decorator_list:
                visit(decorator)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    writes.stores[alias.asname or alias.name.split(".")[0]] = node.lineno
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            writes.stores[node.id] = node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
            writes.deletes[node.id] = node.lineno
        elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            name = _root(node)
            if name:
                writes.mutations[name] = node.lineno
        elif isinstance(node, ast.AugAssign):
            name = _root(node.target)
            if name:
                writes.mutations[name] = node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = _root(node.func.value)
            inplace = any(
                k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in node.keywords
            )
            if name and (node.func.attr in _MUTATING_METHODS or inplace):
                writes.mutations[name] = node.lineno
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in tree.body:
        visit(statement)
    return writes


# --- fingerprints ----------------------------------------------------------


def fingerprint(value: Any) -> Any:
    """A cheap summary of the state that crash predictions depend on.

    Two equal fingerprints of the same object mean nothing a crash is likely
    to depend on has changed: shape, columns, dtypes, length, fitted state.
    Values are never read in full, so a DataFrame of any size costs about as
    much as reading its column names. None means the type is not summarised.
    """

    try:
        return _fingerprint(value)
    except Exception:
        return None


def _fingerprint(value: Any) -> Any:
    kind = type(value)
    if kind in (int, float, bool, complex, type(None)):
        return (kind.__name__, value)
    if kind is str:
        return ("str", value if len(value) <= 1000 else (len(value), hash(value)))
    if kind in (list, tuple, set, frozenset):
        return (kind.__name__, len(value))
    if kind is dict:
        keys = list(value)[:200]
        return ("dict", len(value), hash(tuple(k if isinstance(k, (str, int)) else id(k) for k in keys)))

    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str)

    pd = sys.modules.get("pandas")
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            columns = value.columns
            return (
                "DataFrame",
                value.shape,
                hash(tuple(map(str, columns[:5000]))),
                hash(tuple(map(str, value.dtypes.iloc[:5000]))),
            )
        if isinstance(value, pd.Series):
            return ("Series", len(value), str(value.dtype), str(value.name))

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return ("Tensor", tuple(value.shape), str(value.dtype), str(value.device), value.requires_grad)

    base = sys.modules.get("sklearn.base")
    if base is not None and isinstance(value, base.BaseEstimator):
        state = vars(value)
        fitted = tuple(sorted(k for k in state if k.endswith("_") and not k.startswith("__")))
        return ("estimator", fitted, state.get("n_features_in_"))

    return None
