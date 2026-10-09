"""Rules for Python's own types and builtins: dicts, lists, strings and
numbers, operators, calls, iteration, string formatting and files.
"""

from __future__ import annotations

import builtins
import inspect
import os
import pathlib
import types
from typing import Optional

from ..sites import (
    BinOp,
    Call,
    Compare,
    Delete,
    Iterate,
    MISSING,
    SCALARS,
    Store,
    Subscript,
    UnaryOp,
    Unpack,
    is_hashable_key,
    is_ndarray,
    is_one_of,
    none_detail,
    numpy,
    pandas,
    rule,
    scalar_arithmetic,
    signature_target,
    special_method,
)
from ...texts import text


# --- subscripts and item assignment ------------------------------------------


@rule(Subscript, "missing-key")
def dict_missing_key(site: Subscript) -> None:
    """``d[key]`` on a plain dict that has no such key."""

    if type(site.obj) is dict and is_hashable_key(site.key) and site.key not in site.obj:
        site.crash("KeyError", repr(site.key), [site.obj_root, site.key_root])


@rule(Subscript, "unhashable-key")
def dict_unhashable_key(site: Subscript) -> None:
    """``d[[1, 2]]``: a dict lookup with an unhashable key."""

    if type(site.obj) is dict and type(site.key) in (list, dict, set, bytearray):
        site.crash("TypeError", f"unhashable type: '{type(site.key).__name__}'", [site.key_root])


@rule(Subscript, "index-range")
def sequence_index_out_of_range(site: Subscript) -> None:
    """``lst[i]`` past the end of a list, tuple or string."""

    kind = type(site.obj)
    if kind not in (list, tuple, str) or type(site.key) is not int:
        return
    if not -len(site.obj) <= site.key < len(site.obj):
        noun = {list: "list", tuple: "tuple", str: "string"}[kind]
        detail = (
            text("checker.sequence_length", name=site.obj_root, count=len(site.obj))
            if site.obj_root
            else ""
        )
        site.crash("IndexError", f"{noun} index out of range", [site.obj_root, site.key_root], detail)


@rule(Subscript, "index-type")
def sequence_index_wrong_type(site: Subscript) -> None:
    """``lst["a"]``: lists, tuples and strings take integers or slices only."""

    kind = type(site.obj)
    if kind in (list, tuple, str) and type(site.key) in (str, float, type(None)):
        noun = {list: "list", tuple: "tuple", str: "string"}[kind]
        site.crash(
            "TypeError",
            f"{noun} indices must be integers or slices, not {type(site.key).__name__}",
            [site.obj_root, site.key_root],
        )


@rule(Subscript, "index-type")
def sequence_bad_slice(site: Subscript) -> None:
    """``lst["a":]``, ``lst[1.5:]``, ``lst[::0]``: slice bounds a list, tuple or
    string does not accept."""

    if type(site.obj) not in (list, tuple, str, bytes) or type(site.key) is not slice:
        return
    parts = (site.key.start, site.key.stop, site.key.step)
    if not all(type(p) in (int, bool, type(None)) for p in parts):
        if all(type(p) in SCALARS for p in parts):
            site.crash("TypeError", "slice indices must be integers or None or have an __index__ method",
                       [site.obj_root])
        return
    if site.key.step == 0:
        site.crash("ValueError", "slice step cannot be zero", [site.obj_root])


@rule(Subscript, "not-subscriptable")
def scalar_not_subscriptable(site: Subscript) -> None:
    """``x[...]`` where ``x`` is None or a number."""

    if site.obj is None:
        site.crash("TypeError", "'NoneType' object is not subscriptable", [site.obj_root],
                   none_detail((site.obj, site.obj_root)))
    if type(site.obj) in (int, float, bool):
        site.crash("TypeError", f"'{type(site.obj).__name__}' object is not subscriptable",
                   [site.obj_root])


@rule(Subscript, "missing-key")
def environ_missing_variable(site: Subscript) -> None:
    """``os.environ["API_TOKEN"]`` for a variable that is not set."""

    if site.obj is os.environ and type(site.key) is str and site.key not in os.environ:
        site.crash("KeyError", repr(site.key), [site.key_root],
                   text("checker.env_missing", name=site.key))


@rule(Store, "immutable-assignment")
def assign_into_immutable(site: Store) -> None:
    """``t[0] = 1`` on a tuple, a string, or None."""

    if type(site.obj) in (tuple, str, bytes, frozenset, int, float, bool, type(None)):
        detail = none_detail((site.obj, site.obj_root))
        site.crash("TypeError", f"'{type(site.obj).__name__}' object does not support item assignment",
                   [site.obj_root], detail)


@rule(Delete, "immutable-assignment")
def delete_from_immutable(site: Delete) -> None:
    """``del t[0]`` on a tuple, a string, or None."""

    if type(site.obj) in (tuple, str, bytes, frozenset, int, float, bool, type(None)):
        site.crash("TypeError", f"'{type(site.obj).__name__}' object doesn't support item deletion",
                   [site.obj_root], none_detail((site.obj, site.obj_root)))


@rule(Delete, "missing-key")
def delete_missing_key(site: Delete) -> None:
    """``del d["x"]`` for a key the dict does not have."""

    if type(site.obj) is dict and is_hashable_key(site.key) and site.key not in site.obj:
        site.crash("KeyError", repr(site.key), [site.obj_root, site.key_root])


@rule(Delete, "index-range")
def delete_index_out_of_range(site: Delete) -> None:
    """``del lst[5]`` past the end of a list."""

    if type(site.obj) in (list, bytearray) and type(site.key) is int and not -len(site.obj) <= site.key < len(site.obj):
        site.crash("IndexError", f"{type(site.obj).__name__} assignment index out of range",
                   [site.obj_root, site.key_root])


# --- operators ---------------------------------------------------------------


@rule(BinOp, "operand-types")
def builtin_operand_types(site: BinOp) -> None:
    """Arithmetic on builtin scalars of types Python cannot combine:
    ``None + 1``, ``"a" + 1``."""

    error = _builtin_arithmetic_error(site)
    if isinstance(error, TypeError):
        site.crash("TypeError", str(error), [site.left_root, site.right_root],
                   none_detail((site.left, site.left_root), (site.right, site.right_root)))


@rule(BinOp, "division-by-zero")
def builtin_division_by_zero(site: BinOp) -> None:
    """``x / 0``, ``x // 0`` or ``x % 0`` on builtin numbers."""

    error = _builtin_arithmetic_error(site)
    if isinstance(error, ZeroDivisionError):
        site.crash("ZeroDivisionError", str(error), [site.left_root, site.right_root])


def _builtin_arithmetic_error(site: BinOp) -> Optional[Exception]:
    """The exception arithmetic on two builtin scalars raises, computed for
    real but never on values that could make the result huge."""

    try:
        scalar_arithmetic(site.op, site.left, site.right)
    except (TypeError, ZeroDivisionError) as exc:
        return exc
    return None


@rule(BinOp, "bad-format")
def percent_format_arguments(site: BinOp) -> None:
    """``"%d %d" % (x,)``, ``"%(a)s" % {"b": 1}``: %-formatting with the wrong
    number or names of arguments. Computed for real on builtin values only."""

    template, values = site.left, site.right
    if site.op != "%" or type(template) is not str or len(template) > 10_000:
        return
    if type(values) is tuple:
        if not all(type(v) in SCALARS for v in values):
            return
    elif type(values) is dict:
        if not all(type(k) is str and type(v) in SCALARS for k, v in values.items()):
            return
    else:
        return  # a single builtin value is handled by builtin_operand_types
    try:
        template % values
    except (TypeError, ValueError, KeyError) as exc:
        site.crash(type(exc).__name__, str(exc), [site.left_root, site.right_root])


@rule(UnaryOp, "bad-operand")
def unary_bad_operand(site: UnaryOp) -> None:
    """``-"a"``, ``~1.5``, ``-None``: a unary operator the builtin value does not support."""

    if type(site.operand) not in SCALARS:
        return
    operation = {"-": lambda v: -v, "+": lambda v: +v, "~": lambda v: ~v}[site.op]
    try:
        operation(site.operand)
    except TypeError as exc:
        site.crash("TypeError", str(exc), [site.root],
                   none_detail((site.operand, site.root)))


@rule(Compare, "unsupported-comparison")
def membership_in_non_container(site: Compare) -> None:
    """``x in 5``, ``x in None``: membership in something that is not a container."""

    if site.op not in ("in", "not in"):
        return
    kind = type(site.right)
    if any(special_method(kind, m) is not MISSING for m in ("__contains__", "__iter__", "__getitem__")):
        return
    site.crash("TypeError", f"argument of type '{kind.__name__}' is not iterable", [site.right_root],
               none_detail((site.right, site.right_root)))


@rule(Compare, "unhashable-key")
def membership_with_unhashable_key(site: Compare) -> None:
    """``[1, 2] in some_dict``: looking up an unhashable key in a dict or set."""

    if site.op in ("in", "not in") and type(site.right) in (dict, set, frozenset) \
            and type(site.left) in (list, dict, set, bytearray):
        site.crash("TypeError", f"unhashable type: '{type(site.left).__name__}'", [site.left_root])


@rule(Compare, "unsupported-comparison")
def ordering_incompatible_builtins(site: Compare) -> None:
    """``1 < "a"``, ``None > 0``: ordering builtin values Python cannot compare."""

    if site.op not in _ORDERINGS or type(site.left) not in SCALARS or type(site.right) not in SCALARS:
        return
    try:
        _ORDERINGS[site.op](site.left, site.right)
    except TypeError as exc:
        site.crash("TypeError", str(exc), [site.left_root, site.right_root],
                   none_detail((site.left, site.left_root), (site.right, site.right_root)))


_ORDERINGS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


# --- calls -------------------------------------------------------------------


@rule(Call, "not-callable")
def call_not_callable(site: Call) -> None:
    """``df.shape()``, ``df.columns()``, ``x()`` with ``x`` None: calling
    something whose type has no ``__call__``."""

    if special_method(type(site.func), "__call__") is not MISSING:
        return
    detail = none_detail((site.func, site.func_root))
    site.crash("TypeError", f"'{type(site.func).__name__}' object is not callable", [site.func_root], detail)


@rule(Call, "bad-argument")
def call_arguments_do_not_fit(site: Call) -> None:
    """``f(1)`` for ``def f(a, b)``, ``read_csv(path, error_bad_lines=False)``,
    ``fit_model(X, y, 10)`` where ``epochs`` is keyword-only: arguments the function's
    signature does not accept. Python checks them before running the body.

    Only plain Python functions and their methods, whose signature is the one
    Python enforces. A decorated function may accept other arguments than it
    shows, so it is skipped, except for scikit-learn's parameter validation,
    which checks against the decorated function's own signature.
    """

    target = signature_target(site.func)
    if target is None:
        return
    try:
        signature = inspect.signature(target, follow_wrapped=False)
    except (TypeError, ValueError):
        return
    args = site.args
    if isinstance(site.func, types.MethodType) and target is site.func.__func__:
        args = [site.func.__self__] + list(args)
    try:
        signature.bind(*args, **site.kwargs)
    except TypeError as exc:
        name = getattr(site.func, "__qualname__", getattr(site.func, "__name__", "function"))
        site.crash("TypeError", f"{name}() {exc}", [site.func_root])


@rule(Call, "abstract-class")
def instantiate_abstract_class(site: Call) -> None:
    """Instantiating a class that still has abstract methods."""

    cls = site.func
    if not isinstance(cls, type) or not inspect.isabstract(cls):
        return
    if cls.__new__ is not object.__new__ or type(cls).__call__ is not type.__call__:
        return
    methods = ", ".join(sorted(getattr(cls, "__abstractmethods__", ())))
    site.crash("TypeError", f"Can't instantiate abstract class {cls.__name__} with abstract methods {methods}",
               [site.func_root])


@rule(Call, "len-unsized")
def len_of_unsized(site: Call) -> None:
    """``len(x)`` where ``x`` is None, a number, or a 0-d array."""

    if site.func is not builtins.len or len(site.args) != 1 or site.kwargs:
        return
    value, root = site.args[0], site.arg_roots[0]
    if is_ndarray(value) and value.ndim == 0:
        site.crash("TypeError", "len() of unsized object", [root])
    if type(value) in (int, float, bool, type(None)):
        site.crash("TypeError", f"object of type '{type(value).__name__}' has no len()", [root],
                   none_detail((value, root)))


@rule(Call, "conversion")
def number_conversion(site: Call) -> None:
    """``int("3.5")``, ``float("abc")``: computed for real."""

    if not is_one_of(site.func, builtins.int, builtins.float) or len(site.args) != 1 or site.kwargs:
        return
    value = site.args[0]
    if type(value) in (str, int, float, bool):
        try:
            site.func(value)
        except (ValueError, OverflowError) as exc:
            site.crash(type(exc).__name__, str(exc), [site.arg_roots[0]])


@rule(Call, "range-step")
def range_with_zero_step(site: Call) -> None:
    """``range(0, 10, 0)``."""

    if site.func is not builtins.range or len(site.args) != 3 or site.kwargs:
        return
    # Types first: ``==`` on an arbitrary object would run its own code.
    if all(type(a) is int for a in site.args) and site.args[2] == 0:
        site.crash("ValueError", "range() arg 3 must not be zero", site.arg_roots)


@rule(Call, "bad-format")
def str_format_arguments(site: Call) -> None:
    """``"{} {}".format(x)``, ``"{name}".format(x)``: too few or wrong arguments
    for the template. Computed for real on builtin values only."""

    template = site.receiver
    if type(template) is not str or site.method != "format" or len(template) > 10_000:
        return
    if not all(type(a) in SCALARS for a in site.args) or not all(type(v) in SCALARS for v in site.kwargs.values()):
        return
    try:
        template.format(*site.args, **site.kwargs)
    except (IndexError, KeyError, ValueError) as exc:
        site.crash(type(exc).__name__, str(exc), [site.receiver_root])


@rule(Call, "bad-format")
def format_spec_mismatch(site: Call) -> None:
    """``f"{x:d}"`` or ``format(x, "d")`` with a float, and other format specs
    the value's type rejects. Computed for real on builtin values only."""

    if site.func is not builtins.format or len(site.args) != 2 or site.kwargs:
        return
    value, spec = site.args
    if type(value) not in SCALARS or type(spec) is not str or len(spec) > 100:
        return
    try:
        format(value, spec)
    except (ValueError, TypeError) as exc:
        site.crash(type(exc).__name__, str(exc), [site.arg_roots[0]])


@rule(Call, "bad-argument")
def getenv_with_non_string(site: Call) -> None:
    """``os.getenv(1)``: environment variable names are strings."""

    if site.func is not os.getenv or not site.args:
        return
    key = site.args[0]
    if type(key) in SCALARS and type(key) not in (str, bytes):
        site.crash("TypeError", f"str expected, not {type(key).__name__}", [site.arg_roots[0]])


# --- iteration and unpacking -------------------------------------------------


@rule(Iterate, "not-iterable")
def iterate_not_iterable(site: Iterate) -> None:
    """``for x in None``, ``[f(x) for x in 5]``, ``a, b = 5``: iterating over
    something with neither ``__iter__`` nor ``__getitem__``."""

    kind = type(site.obj)
    if special_method(kind, "__iter__") is not MISSING or special_method(kind, "__getitem__") is not MISSING:
        return
    detail = none_detail((site.obj, site.obj_root))
    if site.context == "unpack":
        message = f"cannot unpack non-iterable {kind.__name__} object"
    else:
        message = f"'{kind.__name__}' object is not iterable"
    site.crash("TypeError", message, [site.obj_root], detail)


@rule(Unpack, "not-iterable")
def star_argument_not_iterable(site: Unpack) -> None:
    """``f(*5)``: unpacking something that cannot be iterated into arguments."""

    kind = type(site.obj)
    if site.kind != "*":
        return
    if special_method(kind, "__iter__") is not MISSING or special_method(kind, "__getitem__") is not MISSING:
        return
    site.crash("TypeError", f"argument after * must be an iterable, not {kind.__name__}", [site.obj_root],
               none_detail((site.obj, site.obj_root)))


@rule(Unpack, "not-a-mapping")
def double_star_argument_not_mapping(site: Unpack) -> None:
    """``f(**[1, 2])``, ``f(**None)``: unpacking something that is not a mapping
    into keyword arguments."""

    if site.kind != "**":
        return
    if type(site.obj) in (list, tuple, set, frozenset, str, bytes, int, float, bool, type(None)) or is_ndarray(site.obj):
        site.crash("TypeError", f"argument after ** must be a mapping, not {type(site.obj).__name__}",
                   [site.obj_root], none_detail((site.obj, site.obj_root)))


# --- files -------------------------------------------------------------------


@rule(Call, "missing-file")
def read_missing_file(site: Call) -> None:
    """``pd.read_csv("data/train.csv")``, ``open("notes.txt")``, ``np.load(...)``
    on a local path that does not exist, relative to the kernel's working
    directory."""

    reader = _file_reader(site)
    if reader is None:
        return
    path, expand_user, root = reader
    if type(path) is not str and not isinstance(path, pathlib.PurePath):
        return
    path = os.fspath(path)
    if "://" in path or not path:
        return
    checked = os.path.expanduser(path) if expand_user else path
    if os.path.exists(checked):
        return
    site.crash("FileNotFoundError", f"[Errno 2] No such file or directory: '{checked}'", [root],
               text("checker.file_missing", path=path, cwd=os.getcwd()))


def _file_reader(site: Call):
    """``(path, expands ~, root)`` when the call opens a file for reading."""

    func = site.func
    if func is builtins.open:
        mode = site.arg(1, "mode", "r")
        if type(mode) is not str or "r" not in mode or any(c in mode for c in "wax"):
            return None
        return site.arg(0, "file"), False, site.arg_root(0, "file")
    pd = pandas()
    if pd is not None:
        for name in _PANDAS_READERS:
            if func is getattr(pd, name, None):
                return site.arg(0, "filepath_or_buffer" if name in ("read_csv", "read_table", "read_fwf") else "path"), True, site.arg_root(0)
    np = numpy()
    if np is not None and func is getattr(np, "load", None):
        return site.arg(0, "file"), False, site.arg_root(0, "file")
    if func is os.listdir and site.args:
        return site.args[0], False, site.arg_roots[0]
    return None


# pandas readers whose first argument is always a path, never data. read_json
# is not among them: it also accepts a JSON string.
_PANDAS_READERS = ("read_csv", "read_table", "read_fwf", "read_excel", "read_parquet", "read_feather",
                   "read_pickle")
