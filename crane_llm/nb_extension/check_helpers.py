"""What the built-in checker's rules are written with.

A rule (see ``check_rules.py``) receives one *site*: an operation the target
cell is about to perform, with the real values it will be performed on. The
kinds of site are:

- ``Subscript``: reading ``obj[key]``
- ``Store``: assigning ``obj[key] = value``
- ``Call``: calling ``func(*args, **kwargs)``, which includes method calls
- ``BinOp``: an arithmetic operation ``left <op> right``
- ``Compare``: a comparison ``left <op> right``, including ``in``
- ``Iterate``: looping over ``obj``, in a ``for``, a comprehension or unpacking
- ``Truth``: testing ``value`` for truth, in an ``if``, ``while``, ``and``,
  ``or`` or ``not``
- ``UnaryOp``: ``-x``, ``+x`` or ``~x``
- ``Delete``: ``del obj[key]``
- ``Unpack``: passing ``*obj`` or ``**obj`` as arguments to a call

A rule that finds the operation will certainly raise calls ``site.crash``.
Otherwise it simply returns.

Values are real objects whenever the checker knows them. A value it did not
compute, such as the result of ``df.head()``, is an ``Opaque`` stand-in, which
no rule should treat as known: test with ``known(value)``.

Libraries are only ever looked up in ``sys.modules``. If the user's kernel has
not imported pandas, no value in it can be a DataFrame, and importing pandas
here would cost seconds for nothing.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

MISSING = object()


# --- values the checker did not compute ------------------------------------


class Opaque:
    """A value the checker produced without computing it.

    ``trusted`` marks a value produced by known side-effect-free library code,
    such as ``df.head()``. Such a value is always a NumPy or pandas object or
    a builtin, so printing it runs no user code.
    """

    __slots__ = ("trusted",)

    def __init__(self, trusted: bool):
        self.trusted = trusted


OPAQUE = Opaque(False)
TRUSTED = Opaque(True)


def known(value: Any) -> bool:
    """True when ``value`` is the real object the cell will see."""

    return not isinstance(value, Opaque)


# --- sites and the rule registry ---------------------------------------------


class RuleCrash(Exception):
    """Raised by ``Site.crash``; the checker turns it into a finding."""

    def __init__(self, exception: str, message: str, variables, detail: str):
        super().__init__(message)
        self.exception = exception
        self.message = message
        self.variables = [v for v in variables if v]
        self.detail = detail


@dataclass
class Site:
    node: ast.AST

    def crash(self, exception: str, message: str, variables=(), detail: str = "") -> None:
        """Report that this operation will certainly raise.

        ``exception`` and ``message`` are what Python would show. ``variables``
        are the notebook variables to blame, which the origin locator traces
        back to the cells that produced them; pass the ``*_root`` attributes.
        ``detail`` is an optional explaining sentence, usually from
        ``ui_texts.json`` under ``checker``.
        """

        raise RuleCrash(exception, message, variables, detail)


@dataclass
class Subscript(Site):
    """Reading ``obj[key]``."""

    obj: Any = None
    key: Any = None
    obj_root: Optional[str] = None
    key_root: Optional[str] = None


@dataclass
class Store(Site):
    """Assigning ``obj[key] = value``."""

    obj: Any = None
    key: Any = None
    value: Any = None
    obj_root: Optional[str] = None
    value_root: Optional[str] = None


@dataclass
class Call(Site):
    """Calling ``func(*args, **kwargs)``.

    For a method call, ``receiver`` is the object it is called on and
    ``method`` its name: ``df.drop(...)`` has receiver ``df`` and method
    ``"drop"``. For a plain function both are None.
    """

    func: Any = None
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    func_root: Optional[str] = None
    arg_roots: List[Optional[str]] = field(default_factory=list)
    kwarg_roots: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def receiver(self) -> Any:
        return bound_receiver(self.func)

    @property
    def method(self) -> Optional[str]:
        return getattr(self.func, "__name__", None) if self.receiver is not None else None

    @property
    def receiver_root(self) -> Optional[str]:
        # The root of ``df.drop`` is ``df``.
        return self.func_root if self.receiver is not None else None

    def arg(self, index: int, keyword: Optional[str] = None, default: Any = MISSING) -> Any:
        """The argument passed at position ``index`` or as ``keyword``."""

        if index < len(self.args):
            return self.args[index]
        if keyword is not None and keyword in self.kwargs:
            return self.kwargs[keyword]
        return default

    def arg_root(self, index: int, keyword: Optional[str] = None) -> Optional[str]:
        if index < len(self.arg_roots):
            return self.arg_roots[index]
        return self.kwarg_roots.get(keyword) if keyword else None

    def is_genuine_method(self) -> bool:
        """True when the method is the one the receiver's class defines, not
        something stored on the instance under the same name."""

        own = getattr(type(self.receiver), self.method or "", None)
        return getattr(self.func, "__func__", None) is own


@dataclass
class BinOp(Site):
    """``left <op> right``. ``op`` is the operator as written: ``"+"``, ``"@"``, ..."""

    op: str = ""
    left: Any = None
    right: Any = None
    left_root: Optional[str] = None
    right_root: Optional[str] = None


@dataclass
class Compare(Site):
    """``left <op> right``. ``op`` as written: ``"<"``, ``"=="``, ``"in"``, ``"not in"``, ...

    Only the first comparison of a chain such as ``a < b < c`` is offered,
    since the rest run only if it is true.
    """

    op: str = ""
    left: Any = None
    right: Any = None
    left_root: Optional[str] = None
    right_root: Optional[str] = None


@dataclass
class Iterate(Site):
    """Iterating over ``obj``. ``context`` is ``"for"``, ``"comprehension"`` or
    ``"unpack"`` (``a, b = obj``), which Python words its errors differently for."""

    obj: Any = None
    obj_root: Optional[str] = None
    context: str = "for"


@dataclass
class Truth(Site):
    """Testing ``value`` for truth: ``if value:``, ``while value:``,
    ``value and ...``, ``not value``."""

    value: Any = None
    root: Optional[str] = None


@dataclass
class UnaryOp(Site):
    """``-operand``, ``+operand`` or ``~operand``. ``op`` is ``"-"``, ``"+"`` or ``"~"``."""

    op: str = ""
    operand: Any = None
    root: Optional[str] = None


@dataclass
class Delete(Site):
    """``del obj[key]``."""

    obj: Any = None
    key: Any = None
    obj_root: Optional[str] = None
    key_root: Optional[str] = None


@dataclass
class Unpack(Site):
    """``f(*obj)`` (``kind="*"``) or ``f(**obj)`` (``kind="**"``)."""

    obj: Any = None
    obj_root: Optional[str] = None
    kind: str = "*"


@dataclass
class _Rule:
    rule_id: str
    site: type
    check: Callable[[Any], None]


RULES: List[_Rule] = []


def rule(site: type, rule_id: str):
    """Register a check for one kind of site.

    ``rule_id`` names the rule in findings and tests, for example
    ``"missing-column"``. Several rules may share an id.
    """

    def register(check: Callable[[Any], None]):
        RULES.append(_Rule(rule_id, site, check))
        return check

    return register


# --- telling values apart -----------------------------------------------------


def numpy():
    return sys.modules.get("numpy")


def pandas():
    return sys.modules.get("pandas")


def is_ndarray(value: Any) -> bool:
    np = numpy()
    return np is not None and type(value) is np.ndarray


def is_frame(value: Any) -> bool:
    pd = pandas()
    return pd is not None and type(value) is pd.DataFrame


def is_series(value: Any) -> bool:
    pd = pandas()
    return pd is not None and type(value) is pd.Series


def is_pandas_index(value: Any) -> bool:
    pd = pandas()
    return pd is not None and isinstance(value, pd.Index) and type(value).__module__.startswith("pandas.")


def is_numpy_scalar(value: Any) -> bool:
    np = numpy()
    return np is not None and isinstance(value, np.generic) and type(value).__module__ == "numpy"


def is_sklearn_estimator(value: Any) -> bool:
    base = sys.modules.get("sklearn.base")
    return (
        base is not None
        and isinstance(value, base.BaseEstimator)
        and type(value).__module__.startswith("sklearn.")
    )


def is_sklearn_estimator_class(kind: Any) -> bool:
    base = sys.modules.get("sklearn.base")
    return (
        base is not None
        and isinstance(kind, type)
        and issubclass(kind, base.BaseEstimator)
        and kind.__module__.startswith("sklearn.")
    )


def is_one_of(value: Any, *candidates: Any) -> bool:
    """Identity, not ``in``: ``==`` on an arbitrary object can run its code,
    and on an array it does not even return a bool."""

    return any(value is candidate for candidate in candidates)


def bound_receiver(func: Any) -> Any:
    import types

    receiver = getattr(func, "__self__", None)
    if isinstance(func, (types.MethodType, types.BuiltinMethodType)) and not isinstance(
        receiver, types.ModuleType
    ):
        return receiver
    return None


SCALARS = (int, float, bool, complex, str, bytes, type(None))
CONTAINERS = (list, tuple, set, frozenset, dict)
# A container longer than this is not inspected element by element.
MAX_INSPECTED = 1000


def is_data(value: Any, depth: int = 0) -> bool:
    """True for values whose printing, hashing or comparison runs no user code."""

    if isinstance(value, Opaque):
        return value.trusted
    kind = type(value)
    if kind in SCALARS or kind is range or kind is slice:
        return True
    if kind in (list, tuple, set, frozenset):
        return depth < 3 and len(value) <= MAX_INSPECTED and all(
            is_data(item, depth + 1) for item in value
        )
    if kind is dict:
        return depth < 3 and len(value) <= MAX_INSPECTED and all(
            is_data(key, depth + 1) and is_data(item, depth + 1) for key, item in value.items()
        )
    if is_ndarray(value):
        # An object array holds arbitrary Python objects, whose own methods
        # would run on printing or arithmetic.
        return value.dtype.kind != "O"
    return is_frame(value) or is_series(value) or is_pandas_index(value) or is_numpy_scalar(value)


def is_hashable_key(key: Any) -> bool:
    """A key whose hashing and comparison run no user code."""

    if type(key) in SCALARS:
        return True
    return type(key) is tuple and all(type(k) in SCALARS for k in key)


def is_label(value: Any) -> bool:
    """A column or index label as written in code: a string or a plain int."""

    return type(value) is str or type(value) is int


def labels(value: Any) -> Optional[List[Any]]:
    """The labels named by a label or a list of labels, or None if neither."""

    if is_label(value):
        return [value]
    if type(value) is list and len(value) <= MAX_INSPECTED and all(is_label(v) for v in value):
        return list(value)
    return None


def plain_axis(index: Any) -> bool:
    """An axis whose label lookup is a plain membership test.

    A MultiIndex matches partial keys, and datetime-like indexes match partial
    date strings, so a label missing from either may still select something.
    """

    pd = pandas()
    if pd is None:
        return False
    return not isinstance(index, (pd.MultiIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex))


def sample_count(value: Any) -> Optional[int]:
    """Number of samples in an array-like a library would see, if knowable."""

    if type(value) in (list, tuple, range):
        return len(value)
    if is_ndarray(value):
        return value.shape[0] if value.ndim >= 1 else None
    if is_frame(value) or is_series(value):
        return len(value.index)
    return None


def special_method(kind: type, name: str) -> Any:
    """``name`` as Python looks up special methods: on the type and its bases,
    never on the instance and never through ``__getattr__``. MISSING if absent."""

    for klass in kind.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return MISSING


def scan_budget() -> int:
    """How many data values a rule may read, from the user's settings.

    Rules that must look at every value of a column or array check their data
    against this first, and skip the check when it is larger: a skipped check
    costs a missed crash, a guess could report a false one.
    """

    from .settings import scan_limit

    return scan_limit()


def shape_text(shape: Tuple[int, ...]) -> str:
    """A shape as NumPy prints it in messages: ``(3,)``, ``(5,3)``."""

    return "(" + ",".join(str(n) for n in shape) + ("," if len(shape) == 1 else "") + ")"
