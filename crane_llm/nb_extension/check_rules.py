"""The built-in checker's rules: the crashes it can report as certain.

Each rule is a function that looks at one operation the target cell is about
to perform, with the real values it will be performed on, and calls
``site.crash(...)`` when that operation will certainly raise. The checker
(``checks.py``) walks the cell in evaluation order and offers every operation
it reaches to the rules registered for that kind of site.

Adding a rule
-------------

1. Pick the site: ``Subscript`` (``obj[key]``), ``Store`` (``obj[key] =
   value``), ``Call`` (``func(...)``, including methods), ``BinOp`` (``left +
   right`` and friends), ``Compare`` (``left < right``, ``x in y``),
   ``Iterate`` (``for x in obj``, comprehensions, unpacking), ``Truth``
   (``if value``, ``and``/``or``/``not``), ``UnaryOp`` (``-x``, ``~x``),
   ``Delete`` (``del obj[key]``) or ``Unpack`` (``f(*x)``, ``f(**x)``). Their
   fields are in ``check_helpers.py``.
2. Write a function decorated with ``@rule(<site>, "<rule-id>")``. Return
   when the rule does not apply; call ``site.crash(exception, message,
   variables, detail)`` when the operation will raise.
3. Only report what is certain. A rule that must read every value of a
   column or array checks its size against ``scan_budget()`` first and skips
   larger data, since a skipped check costs a missed crash while a guess
   could report a false one. The budget is the user's ``scan_limit`` setting.
   Beyond that, only report what is certain. Check exact types (``is_frame``, ``type(x) is
   list``), never ``isinstance``, since a subclass can behave differently.
   Treat a value as known only if ``known(value)``. Confirm the behaviour on
   the library itself, on every version you rely on.
4. Add the case to ``check_builtin_checks_are_certain`` in ``smoke_test.py``,
   which runs it for real and checks that it raises what you said it would.

A rule that raises anything other than through ``site.crash`` is treated as
having found nothing, so a mistake in a rule costs a missed crash, never a
false one.

Template::

    @rule(Call, "my-rule")
    def my_rule(site: Call) -> None:
        if site.method != "something" or not is_frame(site.receiver):
            return
        if <the call will certainly raise>:
            site.crash("ValueError", "<what Python would say>", [site.receiver_root])

The checker itself handles the errors that come from Python's own rules
rather than from a library: undefined names, missing attributes (including
on ``None`` and removed library APIs), imports of modules or names that do
not exist, unhashable dict keys, unpacking the wrong number of values, and
syntax errors.

At the bottom of this file are the lists of operations the checker may walk
past without stopping. They decide how far into a cell the rules can see.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import os
import pathlib
import sys
import types
from typing import Any, List, Optional

from .check_helpers import (
    BinOp,
    Call,
    Compare,
    Delete,
    Iterate,
    MISSING,
    SCALARS,
    Store,
    Subscript,
    Truth,
    UnaryOp,
    Unpack,
    is_frame,
    is_hashable_key,
    is_label,
    is_ndarray,
    is_one_of,
    is_series,
    is_sklearn_estimator,
    is_sklearn_estimator_class,
    labels,
    numpy,
    plain_axis,
    rule,
    pandas,
    sample_count,
    scan_budget,
    shape_text,
    special_method,
)
from .texts import text


def _is_none_detail(name: Optional[str]) -> str:
    return text("checker.is_none", name=name) if name else ""


def _columns_detail(root: Optional[str], columns) -> str:
    try:
        names = [str(c) for c in list(columns)[:12]]
    except Exception:
        return ""
    listed = ", ".join(names) + (", ..." if len(columns) > 12 else "")
    if root:
        return text("checker.columns", name=root, columns=listed)
    return text("checker.columns_unnamed", columns=listed)


# --- Python builtins ----------------------------------------------------------


@rule(Subscript, "missing-key")
def dict_missing_key(site: Subscript) -> None:
    """``d[key]`` on a plain dict that has no such key."""

    if type(site.obj) is dict and is_hashable_key(site.key) and site.key not in site.obj:
        site.crash("KeyError", repr(site.key), [site.obj_root, site.key_root])


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


@rule(Subscript, "not-subscriptable")
def scalar_not_subscriptable(site: Subscript) -> None:
    """``x[...]`` where ``x`` is None or a number."""

    if site.obj is None:
        site.crash("TypeError", "'NoneType' object is not subscriptable", [site.obj_root],
                   _is_none_detail(site.obj_root))
    if type(site.obj) in (int, float, bool):
        site.crash("TypeError", f"'{type(site.obj).__name__}' object is not subscriptable",
                   [site.obj_root])


@rule(BinOp, "operand-types")
def builtin_operand_types(site: BinOp) -> None:
    """Arithmetic on builtin scalars of types Python cannot combine:
    ``None + 1``, ``"a" + 1``."""

    error = _builtin_arithmetic_error(site)
    if isinstance(error, TypeError):
        detail = ""
        for value, root in ((site.left, site.left_root), (site.right, site.right_root)):
            if value is None and root:
                detail = _is_none_detail(root)
        site.crash("TypeError", str(error), [site.left_root, site.right_root], detail)


@rule(BinOp, "division-by-zero")
def builtin_division_by_zero(site: BinOp) -> None:
    """``x / 0``, ``x // 0`` or ``x % 0`` on builtin numbers."""

    error = _builtin_arithmetic_error(site)
    if isinstance(error, ZeroDivisionError):
        site.crash("ZeroDivisionError", str(error), [site.left_root, site.right_root])


def _builtin_arithmetic_error(site: BinOp) -> Optional[Exception]:
    """The exception arithmetic on two builtin scalars raises, computed for
    real but never on values that could make the result huge."""

    left, right = site.left, site.right
    if type(left) not in SCALARS or type(right) not in SCALARS or site.op not in _BUILTIN_OPS:
        return None
    for value in (left, right):
        if type(value) in (str, bytes) and len(value) > 10_000:
            return None
        if type(value) is int and abs(value) > 10**12:
            return None
    if site.op == "*" and (type(left) in (str, bytes) or type(right) in (str, bytes)):
        return None
    try:
        _BUILTIN_OPS[site.op](left, right)
    except (TypeError, ZeroDivisionError) as exc:
        return exc
    return None


_BUILTIN_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
}


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
                   _is_none_detail(root) if value is None else "")


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


@rule(Call, "conversion")
def range_with_zero_step(site: Call) -> None:
    if site.func is builtins.range and len(site.args) == 3 and not site.kwargs and site.args[2] == 0:
        if all(type(a) is int for a in site.args):
            site.crash("ValueError", "range() arg 3 must not be zero", site.arg_roots)


# --- pandas ------------------------------------------------------------------


@rule(Subscript, "missing-column")
def dataframe_missing_column(site: Subscript) -> None:
    """``df["col"]`` or ``df[["a", "b"]]`` naming a column that does not exist."""

    frame = site.obj
    if not is_frame(frame) or not plain_axis(frame.columns) or not plain_axis(frame.index):
        return
    columns = frame.columns
    if is_label(site.key):
        if site.key not in columns:
            site.crash("KeyError", repr(site.key), [site.obj_root], _columns_detail(site.obj_root, columns))
        return
    if type(site.key) is list:
        names = labels(site.key)
        missing = [label for label in names or [] if label not in columns]
        if missing:
            site.crash("KeyError", f"{missing} not in index", [site.obj_root],
                       _columns_detail(site.obj_root, columns))


@rule(Store, "column-length")
def dataframe_column_length(site: Store) -> None:
    """``df["col"] = values`` where ``values`` has a different length than ``df``."""

    frame, value = site.obj, site.value
    if not is_frame(frame) or not is_label(site.key) or not plain_axis(frame.columns):
        return
    if type(value) not in (list, range) and not (is_ndarray(value) and value.ndim == 1):
        return
    rows = len(frame.index)
    # An empty DataFrame takes its index from the first column assigned.
    if rows and len(value) != rows:
        site.crash("ValueError", f"Length of values ({len(value)}) does not match length of index ({rows})",
                   [site.obj_root, site.value_root])


def _frame_method(site: Call, *names: str):
    """The DataFrame a genuine ``df.<name>(...)`` call is made on, or None."""

    frame = site.receiver
    if site.method not in names or not is_frame(frame) or not site.is_genuine_method():
        return None
    if not plain_axis(frame.columns) or not plain_axis(frame.index):
        return None
    return frame


@rule(Call, "missing-column")
def dataframe_drop_missing(site: Call) -> None:
    """``df.drop(columns=[...])`` or ``df.drop(labels, axis=...)`` with a label not there."""

    frame = _frame_method(site, "drop")
    if frame is None:
        return
    kwargs = site.kwargs
    if len(site.args) > 1 or "level" in kwargs or kwargs.get("errors", "raise") != "raise":
        return
    columns, index = frame.columns, frame.index
    targets = []
    given = site.arg(0, "labels")
    if given is not MISSING and given is not None:
        if "index" in kwargs or "columns" in kwargs:
            return
        axis = kwargs.get("axis", 0)
        if axis in (1, "columns"):
            targets.append((given, columns))
        elif axis in (0, "index"):
            targets.append((given, index))
        else:
            return
    for keyword, axis_labels in (("columns", columns), ("index", index)):
        if kwargs.get(keyword) is not None:
            targets.append((kwargs[keyword], axis_labels))
    for given, axis_labels in targets:
        names = labels(given)
        if names is None:
            return
        absent = [label for label in names if label not in axis_labels]
        if absent:
            detail = _columns_detail(site.receiver_root, columns) if axis_labels is columns else ""
            site.crash("KeyError", f"{absent} not found in axis", [site.receiver_root], detail)


@rule(Call, "missing-column")
def dataframe_group_or_sort_missing(site: Call) -> None:
    """``df.groupby("col")`` or ``df.sort_values("col")`` by a label that is
    neither a column nor an index level name."""

    frame = _frame_method(site, "groupby", "sort_values")
    if frame is None:
        return
    by = site.arg(0, "by")
    if by is MISSING or site.kwargs.get("axis", 0) not in (0, "index") or "level" in site.kwargs:
        return
    names = labels(by)
    if names is None:
        return
    # groupby treats a list as the group values, not as labels, when it is as
    # long as the frame, and then it does not raise.
    if site.method == "groupby" and type(by) is list and len(by) == len(frame.index):
        return
    level_names = {n for n in frame.index.names if n is not None}
    absent = [label for label in names if label not in frame.columns and label not in level_names]
    if absent:
        site.crash("KeyError", repr(absent[0]), [site.receiver_root],
                   _columns_detail(site.receiver_root, frame.columns))


@rule(Call, "missing-column")
def dataframe_set_index_missing(site: Call) -> None:
    """``df.set_index("col")`` with a column that does not exist. Index level
    names do not count here."""

    frame = _frame_method(site, "set_index")
    if frame is None:
        return
    keys = site.arg(0, "keys")
    names = labels(keys) if keys is not MISSING else None
    absent = [label for label in names or [] if label not in frame.columns]
    if absent:
        site.crash("KeyError", f"None of {absent} are in the columns", [site.receiver_root],
                   _columns_detail(site.receiver_root, frame.columns))


# --- NumPy -------------------------------------------------------------------


@rule(Subscript, "index-range")
def array_index_out_of_range(site: Subscript) -> None:
    """``a[i]`` or ``a[i, j]`` out of bounds, or with more indices than dimensions."""

    array, key = site.obj, site.key
    if not is_ndarray(array) or array.dtype.names:
        return
    indices = (key,) if type(key) is int else key if type(key) is tuple else None
    if indices is None or not all(type(i) in (int, slice) for i in indices):
        return
    if len(indices) > array.ndim:
        site.crash("IndexError", f"too many indices for array: array is {array.ndim}-dimensional, "
                   f"but {len(indices)} were indexed", [site.obj_root])
    for axis, index in enumerate(indices):
        if type(index) is int and not -array.shape[axis] <= index < array.shape[axis]:
            site.crash("IndexError",
                       f"index {index} is out of bounds for axis {axis} with size {array.shape[axis]}",
                       [site.obj_root])


@rule(BinOp, "broadcast")
def array_broadcast(site: BinOp) -> None:
    """``a + b`` (and ``- * / // % ** & | ^``) on arrays whose shapes do not broadcast."""

    if site.op == "@" or not (is_ndarray(site.left) and is_ndarray(site.right)):
        return
    try:
        numpy().broadcast_shapes(site.left.shape, site.right.shape)
    except ValueError:
        site.crash("ValueError", "operands could not be broadcast together with shapes "
                   f"{shape_text(site.left.shape)} {shape_text(site.right.shape)}",
                   [site.left_root, site.right_root])


@rule(BinOp, "matmul-shape")
def array_matmul_operator(site: BinOp) -> None:
    """``a @ b`` whose inner dimensions differ."""

    if site.op == "@" and is_ndarray(site.left) and is_ndarray(site.right):
        message = _matmul_error(site.left.shape, site.right.shape)
        if message:
            site.crash("ValueError", message, [site.left_root, site.right_root])


@rule(Call, "matmul-shape")
def array_dot(site: Call) -> None:
    """``np.dot(a, b)`` or ``np.matmul(a, b)`` whose inner dimensions differ."""

    np = numpy()
    if np is None or not is_one_of(site.func, np.dot, np.matmul) or len(site.args) != 2 or site.kwargs:
        return
    left, right = site.args
    if is_ndarray(left) and is_ndarray(right):
        message = (
            _dot_error(left.shape, right.shape) if site.func is np.dot
            else _matmul_error(left.shape, right.shape)
        )
        if message:
            site.crash("ValueError", message, site.arg_roots)


@rule(Call, "concatenate-shape")
def array_concatenate(site: Call) -> None:
    """``np.concatenate``, ``np.vstack`` or ``np.hstack`` of arrays that do not fit together."""

    np = numpy()
    if np is None or not is_one_of(site.func, np.concatenate, np.vstack, np.hstack) or not site.args:
        return
    arrays = site.args[0]
    if type(arrays) not in (list, tuple) or not all(is_ndarray(a) for a in arrays):
        return
    if site.func is np.concatenate:
        if len(site.args) > 2 or set(site.kwargs) - {"axis"}:
            return
        axis = site.arg(1, "axis", 0)
        shapes = [a.shape for a in arrays]
    else:
        if len(site.args) > 1 or site.kwargs:
            return
        if site.func is np.vstack:
            shapes, axis = [_atleast(a.shape, 2) for a in arrays], 0
        else:
            shapes = [_atleast(a.shape, 1) for a in arrays]
            axis = 0 if shapes and all(len(s) == 1 for s in shapes) else 1
    if axis is not None and type(axis) is not int:
        return
    message = _concatenate_error(shapes, axis)
    if message:
        site.crash("ValueError", message, site.arg_roots)


@rule(Call, "reshape-size")
def array_reshape(site: Call) -> None:
    """``a.reshape(...)`` or ``np.reshape(a, ...)`` to a shape of a different size."""

    np = numpy()
    if is_ndarray(site.receiver) and site.method == "reshape":
        array, root = site.receiver, site.receiver_root
        shape = site.args[0] if len(site.args) == 1 else tuple(site.args)
    elif np is not None and site.func is np.reshape and len(site.args) == 2 and is_ndarray(site.args[0]):
        array, root = site.args[0], site.arg_roots[0]
        shape = site.args[1]
    else:
        return
    if type(shape) is int:
        shape = (shape,)
    if set(site.kwargs) - {"order"} or type(shape) not in (tuple, list):
        return
    message = _reshape_error(array.size, tuple(shape))
    if message:
        detail = text("checker.array_shape", name=root, shape=shape_text(array.shape)) if root else ""
        site.crash("ValueError", message, [root], detail)


# --- scikit-learn -------------------------------------------------------------

# Methods that scikit-learn's own estimator checks require to raise on an
# estimator that has not been fitted. ``transform`` is not among them:
# stateless transformers such as ``Normalizer`` transform without fitting.
_PREDICT_METHODS = frozenset({"predict", "predict_proba", "predict_log_proba", "decision_function"})
# Methods that scikit-learn's estimator checks require to reject input with a
# different number of features than the estimator was fitted on.
_FEATURE_CHECKED_METHODS = _PREDICT_METHODS | {"transform", "score"}


@rule(Call, "not-fitted")
def estimator_not_fitted(site: Call) -> None:
    """``model.predict(X)`` before ``model.fit``."""

    if site.method not in _PREDICT_METHODS or not is_sklearn_estimator(site.receiver):
        return
    validation = sys.modules.get("sklearn.utils.validation")
    exceptions = sys.modules.get("sklearn.exceptions")
    if validation is None or exceptions is None:
        return
    try:
        validation.check_is_fitted(site.receiver)
    except exceptions.NotFittedError as exc:
        root = site.receiver_root
        site.crash("NotFittedError", str(exc).split(".")[0] + ".", [root],
                   text("checker.not_fitted", name=root) if root else "")


@rule(Call, "feature-count")
def estimator_feature_count(site: Call) -> None:
    """``model.predict(X)`` where ``X`` has a different number of columns than
    the data ``model`` was fitted on."""

    estimator = site.receiver
    if site.method not in _FEATURE_CHECKED_METHODS or not is_sklearn_estimator(estimator) or not site.args:
        return
    expected = vars(estimator).get("n_features_in_")
    data, data_root = site.args[0], site.arg_roots[0]
    if type(expected) is not int or not _accepts_2d_arrays(estimator):
        return
    if is_ndarray(data) and data.ndim == 2:
        columns = data.shape[1]
    elif is_frame(data):
        columns = data.shape[1]
    else:
        return
    if columns != expected:
        root = site.receiver_root
        site.crash(
            "ValueError",
            f"X has {columns} features, but {type(estimator).__name__} is expecting "
            f"{expected} features as input.",
            [data_root, root],
            text("checker.feature_count", data=data_root, columns=columns, model=root, expected=expected)
            if data_root and root else "",
        )


@rule(Call, "sample-count")
def estimator_fit_sample_count(site: Call) -> None:
    """``model.fit(X, y)`` for a classifier or regressor, with ``X`` and ``y``
    of different lengths."""

    base = sys.modules.get("sklearn.base")
    estimator = site.receiver
    if site.method != "fit" or not is_sklearn_estimator(estimator) or base is None:
        return
    if not (base.is_classifier(estimator) or base.is_regressor(estimator)):
        return
    x_len, y_len = sample_count(site.arg(0, "X")), sample_count(site.arg(1, "y"))
    if x_len is not None and y_len is not None and x_len != y_len:
        site.crash("ValueError",
                   f"Found input variables with inconsistent numbers of samples: [{x_len}, {y_len}]",
                   [site.arg_root(0, "X"), site.arg_root(1, "y")])


@rule(Call, "sample-count")
def train_test_split_sample_count(site: Call) -> None:
    """``train_test_split(X, y)`` with arrays of different lengths."""

    selection = sys.modules.get("sklearn.model_selection")
    if selection is None or site.func is not getattr(selection, "train_test_split", None):
        return
    lengths = [sample_count(a) for a in site.args]
    if len(lengths) >= 2 and all(n is not None for n in lengths) and len(set(lengths)) > 1:
        site.crash("ValueError", f"Found input variables with inconsistent numbers of samples: {lengths}",
                   site.arg_roots)


@rule(Call, "bad-argument")
def estimator_bad_argument(site: Call) -> None:
    """``LogisticRegression(n_estimators=3)``: an argument the estimator does
    not take. scikit-learn estimators declare every parameter explicitly."""

    if not is_sklearn_estimator_class(site.func):
        return
    try:
        inspect.signature(site.func).bind(*site.args, **site.kwargs)
    except TypeError as exc:
        site.crash("TypeError", f"{site.func.__name__}.__init__() {exc}", [])


def _accepts_2d_arrays(estimator) -> bool:
    """True when scikit-learn's own estimator checks hold this estimator to
    rejecting inputs with the wrong number of features."""

    utils = sys.modules.get("sklearn.utils")
    get_tags = getattr(utils, "get_tags", None) if utils is not None else None
    try:
        if get_tags is not None:
            tags = get_tags(estimator)
            return (
                bool(tags.input_tags.two_d_array)
                and not tags.input_tags.one_d_array
                and not tags.no_validation
            )
        tags = estimator._get_tags()
        return tags.get("X_types") == ["2darray"] and not tags.get("no_validation", False)
    except Exception:
        return False


# --- any call -------------------------------------------------------------------


@rule(Call, "not-callable")
def call_not_callable(site: Call) -> None:
    """``df.shape()``, ``df.columns()``, ``x()`` with ``x`` None: calling
    something whose type has no ``__call__``."""

    if special_method(type(site.func), "__call__") is not MISSING:
        return
    detail = _is_none_detail(site.func_root) if site.func is None else ""
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

    target = _signature_target(site.func)
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


def _signature_target(func: Any) -> Any:
    """The function whose signature Python checks a call against, or None."""

    plain = func.__func__ if isinstance(func, types.MethodType) else func
    if not isinstance(plain, types.FunctionType) or "__signature__" in vars(plain):
        return None
    wrapped = getattr(plain, "__wrapped__", None)
    if wrapped is None:
        return plain
    code = getattr(plain, "__code__", None)
    if (
        isinstance(wrapped, types.FunctionType)
        and code is not None
        and code.co_filename.replace("\\", "/").endswith("sklearn/utils/_param_validation.py")
    ):
        return wrapped
    return None


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


# --- iteration, truth and comparison ----------------------------------------------


@rule(Iterate, "not-iterable")
def iterate_not_iterable(site: Iterate) -> None:
    """``for x in None``, ``[f(x) for x in 5]``, ``a, b = 5``: iterating over
    something with neither ``__iter__`` nor ``__getitem__``."""

    kind = type(site.obj)
    if special_method(kind, "__iter__") is not MISSING or special_method(kind, "__getitem__") is not MISSING:
        return
    detail = _is_none_detail(site.obj_root) if site.obj is None else ""
    if site.context == "unpack":
        message = f"cannot unpack non-iterable {kind.__name__} object"
    else:
        message = f"'{kind.__name__}' object is not iterable"
    site.crash("TypeError", message, [site.obj_root], detail)


@rule(Truth, "ambiguous-truth")
def ambiguous_truth_value(site: Truth) -> None:
    """``if df:``, ``if arr:`` with more than one element, ``not series``."""

    value = site.value
    if is_frame(value) or is_series(value):
        kind = "DataFrame" if is_frame(value) else "Series"
        site.crash("ValueError", f"The truth value of a {kind} is ambiguous. Use a.empty, a.bool(), "
                   "a.item(), a.any() or a.all().", [site.root])
    if is_ndarray(value) and value.size > 1:
        site.crash("ValueError", "The truth value of an array with more than one element is ambiguous. "
                   "Use a.any() or a.all()", [site.root])


@rule(Compare, "unsupported-comparison")
def membership_in_non_container(site: Compare) -> None:
    """``x in 5``, ``x in None``: membership in something that is not a container."""

    if site.op not in ("in", "not in"):
        return
    kind = type(site.right)
    if any(special_method(kind, m) is not MISSING for m in ("__contains__", "__iter__", "__getitem__")):
        return
    site.crash("TypeError", f"argument of type '{kind.__name__}' is not iterable", [site.right_root],
               _is_none_detail(site.right_root) if site.right is None else "")


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
        detail = ""
        for value, root in ((site.left, site.left_root), (site.right, site.right_root)):
            if value is None and root:
                detail = _is_none_detail(root)
        site.crash("TypeError", str(exc), [site.left_root, site.right_root], detail)


_ORDERINGS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@rule(Subscript, "unhashable-key")
def dict_unhashable_key(site: Subscript) -> None:
    """``d[[1, 2]]``: a dict lookup with an unhashable key."""

    if type(site.obj) is dict and type(site.key) in (list, dict, set, bytearray):
        site.crash("TypeError", f"unhashable type: '{type(site.key).__name__}'", [site.key_root])


@rule(Store, "immutable-assignment")
def assign_into_immutable(site: Store) -> None:
    """``t[0] = 1`` on a tuple, a string, or None."""

    if type(site.obj) in (tuple, str, bytes, frozenset, int, float, bool, type(None)):
        detail = _is_none_detail(site.obj_root) if site.obj is None else ""
        site.crash("TypeError", f"'{type(site.obj).__name__}' object does not support item assignment",
                   [site.obj_root], detail)


# --- string formatting ----------------------------------------------------------------


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


# --- files and modules ------------------------------------------------------------------


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


# --- pandas: converting, positions and labels ------------------------------------------


@rule(Call, "astype-int")
def astype_to_integer(site: Call) -> None:
    """``df["id"].astype(int)`` where some value cannot become an integer:
    text such as ``"n/a"``, ``None``, or NaN or infinity in a float column.

    Reads every value, so it is skipped for data larger than the scan limit.
    """

    obj = site.receiver
    if site.method != "astype" or not (is_series(obj) or is_frame(obj)) or not site.is_genuine_method():
        return
    if site.kwargs.get("errors", "raise") != "raise":
        return
    dtype = site.arg(0, "dtype")
    if is_series(obj):
        targets = [(obj, dtype)]
    elif type(dtype) is dict:
        targets = [(obj[col], dt) for col, dt in dtype.items() if is_label(col) and col in obj.columns]
    else:
        targets = [(obj.iloc[:, i], dtype) for i in range(obj.shape[1])]
    budget = scan_budget()
    for series, dt in targets:
        if not is_series(series) or not _is_numpy_integer(dt):
            continue
        if len(series) > budget:
            return
        budget -= len(series)
        problem = _integer_cast_problem(series, dt)
        if problem is not None:
            exception, message, value = problem
            site.crash(exception, message, [site.receiver_root],
                       text("checker.bad_integer_value", value=repr(value)))


def _is_numpy_integer(dtype: Any) -> bool:
    np = numpy()
    if np is None or dtype is None:
        return False
    if dtype is int:
        return True
    if not (isinstance(dtype, (str, np.dtype)) or (isinstance(dtype, type) and issubclass(dtype, np.integer))):
        return False
    try:
        # pandas' nullable "Int64" is not a NumPy dtype and accepts missing values.
        return np.dtype(dtype).kind in "iu"
    except TypeError:
        return False


def _integer_cast_problem(series, dtype):
    """``(exception, message, value)`` for a value pandas cannot cast, or None."""

    np = numpy()
    kind = series.dtype.kind
    if kind == "f":
        values = series.to_numpy()
        finite = np.isfinite(values)
        if finite.all():
            return None
        bad = values[~finite][0]
        errors = getattr(pandas(), "errors", None)
        exception = "IntCastingNaNError" if hasattr(errors, "IntCastingNaNError") else "ValueError"
        return exception, "Cannot convert non-finite values (NA or inf) to integer", bad
    if kind != "O":
        return None
    for value in series.array:
        if type(value) in (int, bool):
            continue
        if type(value) not in SCALARS:
            # int() of an arbitrary object runs its own code.
            return None
        try:
            int(value)
            continue
        except (TypeError, ValueError, OverflowError):
            pass
        # Confirm with pandas itself, on this one value in the column's own
        # dtype: one value that cannot be cast makes the whole cast fail.
        try:
            pandas().Series([value], dtype=series.dtype).astype(dtype)
        except Exception as exc:
            return type(exc).__name__, str(exc), value
    return None


@rule(Subscript, "index-range")
def pandas_iloc_out_of_range(site: Subscript) -> None:
    """``df.iloc[n]`` or ``s.iloc[n]`` past the end."""

    target = _indexer_target(site.obj, "_iLocIndexer")
    if target is None:
        return
    keys = site.key if type(site.key) is tuple else (site.key,)
    if len(keys) > target.ndim or not all(type(k) in (int, slice) for k in keys):
        return
    for axis, key in enumerate(keys):
        if type(key) is not int:
            continue
        size = target.shape[axis]
        if not -size <= key < size:
            message = (
                "single positional indexer is out-of-bounds"
                if type(site.key) is not tuple
                else f"index {key} is out of bounds for axis {axis} with size {size}"
            )
            # The root names the target only for df.iloc, not for df["col"].iloc.
            detail = (
                text("checker.positions", name=site.obj_root, shape=shape_text(target.shape))
                if site.obj_root and is_frame(target) else ""
            )
            site.crash("IndexError", message, [site.obj_root], detail)


@rule(Subscript, "missing-label")
def pandas_loc_missing_label(site: Subscript) -> None:
    """``df.loc["row"]``, ``df.loc[["a", "b"]]``, ``df.loc["row", "col"]`` with a
    label that is not in the index or columns."""

    target = _indexer_target(site.obj, "_LocIndexer")
    if target is None:
        return
    keys = site.key if type(site.key) is tuple else (site.key,)
    axes = [target.index] + ([target.columns] if is_frame(target) else [])
    if len(keys) > len(axes):
        return
    for key, axis_labels in zip(keys, axes):
        if not plain_axis(axis_labels):
            return
        if type(key) is slice:
            continue
        names = labels(key)
        if names is None:
            return
        missing = [label for label in names if label not in axis_labels]
        if missing:
            message = repr(missing[0]) if is_label(key) else f"{missing} not in index"
            site.crash("KeyError", message, [site.obj_root])


def _indexer_target(indexer: Any, kind: str):
    """The DataFrame or Series behind ``df.loc`` / ``df.iloc``, or None."""

    if type(indexer).__module__ != "pandas.core.indexing" or type(indexer).__name__ != kind:
        return None
    target = getattr(indexer, "obj", None)
    return target if is_frame(target) or is_series(target) else None


# --- scikit-learn: feature names and targets --------------------------------------------


@rule(Call, "feature-names")
def estimator_feature_names(site: Call) -> None:
    """``model.predict(new_df)`` where ``new_df``'s columns are not the ones
    the estimator was fitted on, in the same order. An error from
    scikit-learn 1.2 on; before that only a warning."""

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _FEATURE_CHECKED_METHODS or not is_sklearn_estimator(estimator) or not is_frame(data):
        return
    fitted = vars(estimator).get("feature_names_in_")
    if fitted is None or not _sklearn_at_least(1, 2) or not _accepts_2d_arrays(estimator):
        return
    columns = list(data.columns)
    if not all(type(c) is str for c in columns):
        return
    expected = [str(c) for c in fitted]
    if columns != expected:
        data_root, root = site.arg_root(0, "X"), site.receiver_root
        site.crash("ValueError", "The feature names should match those that were passed during fit.",
                   [data_root, root],
                   text("checker.feature_names", data=data_root, columns=_listed(columns),
                        model=root, expected=_listed(expected)) if data_root and root else "")


@rule(Call, "continuous-target")
def classification_metric_on_continuous(site: Call) -> None:
    """``f1_score(y_true, scores)`` where ``scores`` are continuous, such as
    a regressor's predictions. Reads every value, within the scan limit."""

    metrics = sys.modules.get("sklearn.metrics")
    if metrics is None or not any(site.func is getattr(metrics, n, None) for n in _CLASSIFICATION_METRICS):
        return
    y_true, y_pred = site.arg(0, "y_true"), site.arg(1, "y_pred")
    if not _small_targets(y_true, y_pred):
        return
    checker = getattr(sys.modules.get("sklearn.metrics._classification"), "_check_targets", None)
    if checker is None:
        return
    try:
        checker(y_true, y_pred)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [site.arg_root(0, "y_true"), site.arg_root(1, "y_pred")])


@rule(Call, "continuous-target")
def classifier_fit_on_continuous(site: Call) -> None:
    """``LogisticRegression().fit(X, y)`` where ``y`` is continuous. Reads
    every value of ``y``, within the scan limit."""

    base = sys.modules.get("sklearn.base")
    estimator = site.receiver
    if site.method != "fit" or not is_sklearn_estimator(estimator) or base is None or not base.is_classifier(estimator):
        return
    y = site.arg(1, "y")
    if not _small_targets(y):
        return
    check = getattr(sys.modules.get("sklearn.utils.multiclass"), "check_classification_targets", None)
    if check is None:
        return
    try:
        check(y)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [site.arg_root(1, "y"), site.receiver_root])


# Metrics that check their targets with ``_check_targets`` before anything else.
_CLASSIFICATION_METRICS = (
    "accuracy_score", "balanced_accuracy_score", "f1_score", "fbeta_score", "precision_score",
    "recall_score", "precision_recall_fscore_support", "confusion_matrix", "classification_report",
    "jaccard_score", "matthews_corrcoef", "hamming_loss", "zero_one_loss", "cohen_kappa_score",
)


def _small_targets(*values) -> bool:
    """Plain arrays of targets, within the scan limit together."""

    total = 0
    for value in values:
        if is_ndarray(value):
            if value.dtype.kind == "O":
                return False
        elif not (is_series(value) or (type(value) is list and all(type(v) in SCALARS for v in value))):
            return False
        total += len(value)
    return total <= scan_budget()


def _sklearn_at_least(major: int, minor: int) -> bool:
    module = sys.modules.get("sklearn")
    try:
        parts = str(module.__version__).split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def _listed(names: List[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (", ..." if len(names) > limit else "")


# --- more of Python's own errors ---------------------------------------------------------


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
                   _is_none_detail(site.root) if site.operand is None else "")


@rule(Delete, "immutable-assignment")
def delete_from_immutable(site: Delete) -> None:
    """``del t[0]`` on a tuple, a string, or None."""

    if type(site.obj) in (tuple, str, bytes, frozenset, int, float, bool, type(None)):
        site.crash("TypeError", f"'{type(site.obj).__name__}' object doesn't support item deletion",
                   [site.obj_root], _is_none_detail(site.obj_root) if site.obj is None else "")


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


@rule(Unpack, "not-iterable")
def star_argument_not_iterable(site: Unpack) -> None:
    """``f(*5)``: unpacking something that cannot be iterated into arguments."""

    kind = type(site.obj)
    if site.kind != "*":
        return
    if special_method(kind, "__iter__") is not MISSING or special_method(kind, "__getitem__") is not MISSING:
        return
    site.crash("TypeError", f"argument after * must be an iterable, not {kind.__name__}", [site.obj_root],
               _is_none_detail(site.obj_root) if site.obj is None else "")


@rule(Unpack, "not-a-mapping")
def double_star_argument_not_mapping(site: Unpack) -> None:
    """``f(**[1, 2])``, ``f(**None)``: unpacking something that is not a mapping
    into keyword arguments."""

    if site.kind != "**":
        return
    if type(site.obj) in (list, tuple, set, frozenset, str, bytes, int, float, bool, type(None)) or is_ndarray(site.obj):
        site.crash("TypeError", f"argument after ** must be a mapping, not {type(site.obj).__name__}",
                   [site.obj_root], _is_none_detail(site.obj_root) if site.obj is None else "")


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


@rule(Call, "bad-argument")
def getenv_with_non_string(site: Call) -> None:
    """``os.getenv(1)``: environment variable names are strings."""

    if site.func is not os.getenv or not site.args:
        return
    key = site.args[0]
    if type(key) in SCALARS and type(key) not in (str, bytes):
        site.crash("TypeError", f"str expected, not {type(key).__name__}", [site.arg_roots[0]])


@rule(Subscript, "missing-key")
def environ_missing_variable(site: Subscript) -> None:
    """``os.environ["API_TOKEN"]`` for a variable that is not set."""

    if site.obj is os.environ and type(site.key) is str and site.key not in os.environ:
        site.crash("KeyError", repr(site.key), [site.key_root],
                   text("checker.env_missing", name=site.key))


# --- NumPy: axes, files, histograms, indexing ----------------------------------------------

# Reductions that take ``axis`` and raise AxisError when it is out of range.
_AXIS_REDUCTIONS = frozenset(
    {"sum", "mean", "min", "max", "std", "var", "prod", "argmax", "argmin", "any", "all",
     "cumsum", "cumprod", "median", "ptp", "amin", "amax", "nanmean", "nansum"}
)


@rule(Call, "axis-range")
def array_axis_out_of_range(site: Call) -> None:
    """``a.sum(axis=3)`` or ``np.mean(a, axis=2)`` with an axis the array does not have."""

    np = numpy()
    if np is None:
        return
    if is_ndarray(site.receiver) and site.method in _AXIS_REDUCTIONS:
        array, root, axis = site.receiver, site.receiver_root, site.arg(0, "axis", None)
    elif any(site.func is getattr(np, name, None) for name in _AXIS_REDUCTIONS) and site.args and is_ndarray(site.args[0]):
        array, root, axis = site.args[0], site.arg_roots[0], site.arg(1, "axis", None)
    else:
        return
    axes = axis if type(axis) is tuple else (axis,)
    for each in axes:
        if type(each) is int and not -array.ndim <= each < array.ndim:
            site.crash("AxisError", f"axis {each} is out of bounds for array of dimension {array.ndim}",
                       [root], text("checker.array_shape", name=root, shape=shape_text(array.shape)) if root else "")


@rule(Call, "array-dimensions")
def savetxt_dimensions(site: Call) -> None:
    """``np.savetxt(path, a)`` with ``a`` of more than two dimensions (or none)."""

    np = numpy()
    data = site.arg(1, "X")
    if np is None or site.func is not getattr(np, "savetxt", None) or not is_ndarray(data):
        return
    if data.ndim == 0 or data.ndim > 2:
        site.crash("ValueError", f"Expected 1D or 2D array, got {data.ndim}D array instead",
                   [site.arg_root(1, "X")])


@rule(Call, "non-finite-data")
def histogram_of_non_finite(site: Call) -> None:
    """``np.histogram(a)`` where ``a`` holds NaN or infinity and no ``range`` is
    given, so the bin range cannot be worked out. Reads every value, within
    the scan limit."""

    np = numpy()
    if np is None or site.func is not getattr(np, "histogram", None) or not site.args:
        return
    data, bins = site.args[0], site.arg(1, "bins", 10)
    if not is_ndarray(data) or data.dtype.kind != "f" or site.arg(2, "range", None) is not None:
        return
    if type(bins) not in (int, str) or data.size == 0 or data.size > scan_budget():
        return
    if np.isfinite(data).all():
        return
    site.crash("ValueError", f"autodetected range of [{np.min(data)}, {np.max(data)}] is not finite",
               [site.arg_roots[0]])


@rule(Subscript, "index-type")
def array_string_index(site: Subscript) -> None:
    """``arr["price"]`` on a plain NumPy array, which only takes positions."""

    if is_ndarray(site.obj) and not site.obj.dtype.names and type(site.key) is str:
        site.crash("IndexError", "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and "
                   "integer or boolean arrays are valid indices", [site.obj_root])


@rule(Subscript, "index-type")
def list_indexed_by_array(site: Subscript) -> None:
    """``items[idx]`` where ``idx`` is an array of several values: a list takes
    one integer."""

    if type(site.obj) in (list, tuple, str) and is_ndarray(site.key) and site.key.size != 1:
        site.crash("TypeError", "only integer scalar arrays can be converted to a scalar index",
                   [site.obj_root, site.key_root])


@rule(Call, "astype-int")
def array_astype_integer(site: Call) -> None:
    """``a.astype(int)`` on an array of text with a value that is not an integer.
    Reads every value, within the scan limit."""

    np = numpy()
    array = site.receiver
    if np is None or not is_ndarray(array) or site.method != "astype" or array.dtype.kind not in "USO":
        return
    dtype = site.arg(0, "dtype")
    if not _is_numpy_integer(dtype) or array.size > scan_budget():
        return
    for value in array.ravel():
        value = value.item() if hasattr(value, "item") and array.dtype.kind != "O" else value
        if type(value) not in SCALARS:
            return
        try:
            int(value)
            continue
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            np.array([value], dtype=array.dtype).astype(dtype)
        except Exception as exc:
            site.crash(type(exc).__name__, str(exc), [site.receiver_root],
                       text("checker.bad_integer_value", value=repr(value)))


# --- pandas: accessors, lengths, concatenating and merging -----------------------------------


@rule(BinOp, "length-mismatch")
def series_arithmetic_length(site: BinOp) -> None:
    """``s + [1, 2]`` or ``s * arr`` where the list or array has a different
    length than the Series. Two Series are aligned by index instead, so they
    never qualify."""

    if site.op not in ("+", "-", "*", "/", "//", "%", "**"):
        return
    pair = _series_and_sequence(site.left, site.right)
    if pair is not None:
        series, other = pair
        site.crash("ValueError", f"operands could not be broadcast together with shapes "
                   f"({len(series)},) ({len(other)},) ", [site.left_root, site.right_root])


@rule(Compare, "length-mismatch")
def series_comparison_length(site: Compare) -> None:
    """``s == [1, 2]`` with a list or array of a different length than the Series."""

    if site.op not in ("==", "!=", "<", "<=", ">", ">="):
        return
    pair = _series_and_sequence(site.left, site.right)
    if pair is not None:
        series, other = pair
        site.crash("ValueError", f"('Lengths must match to compare', ({len(series)},), ({len(other)},))",
                   [site.left_root, site.right_root])


def _series_and_sequence(left, right):
    """``(series, other)`` when one side is a Series and the other a list or
    1-D array of a different length, else None."""

    for series, other in ((left, right), (right, left)):
        if not is_series(series):
            continue
        plain_list = type(other) is list and all(type(v) in SCALARS for v in other)
        flat_array = is_ndarray(other) and other.ndim == 1 and other.dtype.kind != "O"
        if (plain_list or flat_array) and len(other) != len(series):
            return series, other
    return None


@rule(Call, "empty-concat")
def concat_nothing(site: Call) -> None:
    """``pd.concat([])``: nothing to concatenate."""

    pd = pandas()
    if pd is None or site.func is not getattr(pd, "concat", None):
        return
    objs = site.arg(0, "objs")
    if type(objs) in (list, tuple):
        if not objs:
            site.crash("ValueError", "No objects to concatenate", [site.arg_root(0, "objs")])
        if all(o is None for o in objs):
            site.crash("ValueError", "All objects passed were None", [site.arg_root(0, "objs")])


@rule(Call, "missing-column")
def merge_missing_key(site: Call) -> None:
    """``pd.merge(a, b, on="id")``, ``a.merge(b, left_on=...)``,
    ``a.join(b, on="id")`` with a key column one of the frames does not have."""

    pd = pandas()
    if pd is None:
        return
    if site.func is getattr(pd, "merge", None) and len(site.args) >= 2:
        left, right, given = site.args[0], site.args[1], site.kwargs
        left_root = site.arg_roots[0]
    elif site.method == "merge" and is_frame(site.receiver) and site.is_genuine_method() and site.args:
        left, right, given = site.receiver, site.args[0], site.kwargs
        left_root = site.receiver_root
    elif site.method == "join" and is_frame(site.receiver) and site.is_genuine_method() and "on" in site.kwargs:
        left, right, given = site.receiver, site.arg(0, "other"), {"left_on": site.kwargs["on"]}
        left_root = site.receiver_root
    else:
        return
    if not is_frame(left) or not (is_frame(right) or site.method == "join"):
        return
    if any(k in given for k in ("left_index", "right_index")) and site.method != "join":
        return
    sides = []
    if "on" in given:
        if "left_on" in given or "right_on" in given:
            return
        sides = [(left, given["on"], left_root), (right, given["on"], None)]
    else:
        if "left_on" in given:
            sides.append((left, given["left_on"], left_root))
        if "right_on" in given and is_frame(right):
            sides.append((right, given["right_on"], None))
    for frame, keys, root in sides:
        if not is_frame(frame) or not plain_axis(frame.columns):
            return
        names = labels(keys)
        if names is None:
            return
        levels = {n for n in frame.index.names if n is not None}
        missing = [k for k in names if k not in frame.columns and k not in levels]
        if missing:
            site.crash("KeyError", repr(missing[0]), [root], _columns_detail(root, frame.columns))


# --- scikit-learn: values the estimator cannot take ---------------------------------------------


@rule(Call, "non-finite-data")
def estimator_rejects_nan(site: Call) -> None:
    """``model.fit(X, y)`` or ``model.predict(X)`` where ``X`` holds NaN or
    infinity, for an estimator that does not accept missing values. Reads
    every value, within the scan limit."""

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _DATA_METHODS or not _validates_plain_input(estimator):
        return
    values = _numeric_values(data)
    if values is None or _allows_nan(estimator):
        return
    np = numpy()
    if np.isfinite(values).all():
        return
    message = (
        "Input X contains NaN." if np.isnan(values).any()
        else "Input X contains infinity or a value too large for dtype('float64')."
    )
    site.crash("ValueError", message, [site.arg_root(0, "X")], text("checker.contains_nan", name=site.arg_root(0, "X"))
               if site.arg_root(0, "X") else "")


@rule(Call, "bad-value")
def estimator_rejects_text(site: Call) -> None:
    """``model.fit(df, y)`` where a column holds text that is not a number,
    such as ``"male"``, for an estimator that only takes numbers. Reads every
    value of the text columns, within the scan limit.

    Each suspect value is confirmed by scikit-learn's own input validation,
    on a one-row copy of the data in the same dtype, so the check follows
    however the installed versions convert text.
    """

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _DATA_METHODS or not _validates_plain_input(estimator):
        return
    if not _takes_only_numbers(estimator):
        return
    validation = sys.modules.get("sklearn.utils.validation")
    check_array = getattr(validation, "check_array", None)
    if check_array is None:
        return
    np = numpy()
    if is_frame(data):
        if not plain_axis(data.columns):
            return
        text_columns = [i for i in range(data.shape[1]) if _is_text_dtype(data.dtypes.iloc[i])]
        if not text_columns or sum(len(data) for _ in text_columns) > scan_budget():
            return
        candidates = (
            (value, i) for i in text_columns for value in data.iloc[:, i].array
        )
    elif is_ndarray(data) and data.dtype.kind in "US":
        if data.size > scan_budget():
            return
        candidates = ((v.item(), None) for v in data.ravel())
    else:
        return
    for value, column in candidates:
        if type(value) is not str:
            if type(value) not in SCALARS:
                return
            continue
        try:
            float(value)
            continue
        except ValueError:
            pass
        if column is None:
            sample = np.array([[value]], dtype=data.dtype)
        else:
            sample = pandas().DataFrame(
                {data.columns[column]: pandas().Series([value], dtype=data.dtypes.iloc[column])}
            )
        try:
            check_array(sample)
        except ValueError as exc:
            site.crash("ValueError", str(exc), [site.arg_root(0, "X")],
                       text("checker.text_in_numbers", value=repr(value)))
        return


def _is_text_dtype(dtype) -> bool:
    """object, or one of pandas' string dtypes (the default for text from
    pandas 3). Not category, which scikit-learn converts differently."""

    pd = pandas()
    if str(dtype) == "object":
        return True
    string_dtype = getattr(pd, "StringDtype", None) if pd is not None else None
    return string_dtype is not None and isinstance(dtype, string_dtype)


def _takes_only_numbers(estimator) -> bool:
    """True when the estimator does not accept text or categories as input.

    Encoders such as OneHotEncoder take text; scikit-learn records that in
    the estimator's tags, which changed form in 1.6.
    """

    utils = sys.modules.get("sklearn.utils")
    get_tags = getattr(utils, "get_tags", None) if utils is not None else None
    try:
        if get_tags is not None:
            inputs = get_tags(estimator).input_tags
            return not inputs.string and not inputs.categorical and not inputs.dict
        return estimator._get_tags().get("X_types") == ["2darray"]
    except Exception:
        return False


@rule(Call, "too-few-samples")
def kneighbors_more_than_fitted(site: Call) -> None:
    """``knn.predict(X)`` where ``n_neighbors`` is larger than the number of
    samples ``knn`` was fitted on."""

    estimator = site.receiver
    neighbors = sys.modules.get("sklearn.neighbors")
    if neighbors is None or type(estimator) not in (neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor):
        return
    if site.method not in ("predict", "predict_proba", "score") or not site.args:
        return
    fitted, wanted = vars(estimator).get("n_samples_fit_"), estimator.n_neighbors
    if type(fitted) is int and type(wanted) is int and wanted > fitted:
        site.crash("ValueError", f"Expected n_neighbors <= n_samples,  but n_samples = {fitted}, "
                   f"n_neighbors = {wanted}", [site.receiver_root])


@rule(Call, "bad-argument")
def sklearn_parameter_constraints(site: Call) -> None:
    """``train_test_split(X, test_size=1.5)`` and any other scikit-learn
    function that validates its parameters (from 1.2 on): a plain value
    outside the function's declared constraints. Checked with scikit-learn's
    own validator, which the function runs before anything else."""

    func = site.func
    constraints = getattr(func, "_skl_parameter_constraints", None)
    validator = getattr(sys.modules.get("sklearn.utils._param_validation"), "validate_parameter_constraints", None)
    sklearn = sys.modules.get("sklearn")
    if not isinstance(constraints, dict) or validator is None or sklearn is None:
        return
    try:
        if sklearn.get_config().get("skip_parameter_validation"):
            return
        target = _signature_target(func)
        bound = inspect.signature(target or func, follow_wrapped=False).bind_partial(*site.args, **site.kwargs)
    except Exception:
        return
    # Only plain values: checking an array or estimator against a constraint
    # would look at objects whose meaning the checker does not model.
    plain = {k: v for k, v in bound.arguments.items() if type(v) in SCALARS and k in constraints}
    if not plain:
        return
    errors = sys.modules.get("sklearn.utils._param_validation")
    invalid = getattr(errors, "InvalidParameterError", ValueError)
    try:
        validator({k: constraints[k] for k in plain}, plain, caller_name=getattr(func, "__qualname__", "function"))
    except invalid as exc:
        site.crash(type(exc).__name__, str(exc), [])


@rule(Call, "bad-argument")
def train_test_split_sizes(site: Call) -> None:
    """``train_test_split(X, test_size=1.5)``: a test or train size that is
    neither a fraction in (0, 1) nor a count below the number of samples."""

    split = sys.modules.get("sklearn.model_selection._split")
    selection = sys.modules.get("sklearn.model_selection")
    if split is None or selection is None or site.func is not getattr(selection, "train_test_split", None):
        return
    validate = getattr(split, "_validate_shuffle_split", None)
    lengths = [sample_count(a) for a in site.args]
    if validate is None or not lengths or any(n is None for n in lengths) or len(set(lengths)) != 1:
        return
    test_size, train_size = site.kwargs.get("test_size"), site.kwargs.get("train_size")
    if not all(v is None or type(v) in (int, float) for v in (test_size, train_size)):
        return
    try:
        validate(lengths[0], test_size, train_size, default_test_size=0.25)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [])


_DATA_METHODS = frozenset({"fit", "predict", "predict_proba", "predict_log_proba", "decision_function",
                           "transform", "fit_transform", "fit_predict", "score"})


def _validates_plain_input(estimator) -> bool:
    """A scikit-learn estimator, not a pipeline or meta-estimator, that its own
    estimator checks hold to validating numeric 2-D input."""

    base = sys.modules.get("sklearn.base")
    metaestimators = sys.modules.get("sklearn.utils.metaestimators")
    if base is None or not is_sklearn_estimator(estimator):
        return False
    if isinstance(estimator, base.MetaEstimatorMixin):
        return False
    composition = getattr(metaestimators, "_BaseComposition", None)
    if composition is not None and isinstance(estimator, composition):
        return False
    return _accepts_2d_arrays(estimator)


def _allows_nan(estimator) -> bool:
    utils = sys.modules.get("sklearn.utils")
    get_tags = getattr(utils, "get_tags", None) if utils is not None else None
    try:
        if get_tags is not None:
            return bool(get_tags(estimator).input_tags.allow_nan)
        return bool(estimator._get_tags().get("allow_nan", False))
    except Exception:
        return True  # unknown: assume allowed, and claim nothing


def _numeric_values(data):
    """All values of a numeric array or DataFrame, within the scan limit, else None."""

    np = numpy()
    if is_ndarray(data):
        if data.dtype.kind not in "fiub" or data.size > scan_budget():
            return None
        return data
    if is_frame(data):
        if data.size > scan_budget() or not all(dt.kind in "fiub" for dt in data.dtypes):
            return None
        return data.to_numpy(dtype=np.float64)
    return None


# --- matplotlib --------------------------------------------------------------------------------


@rule(Call, "array-dimensions")
def imshow_shape(site: Call) -> None:
    """``plt.imshow(a)`` with an array that cannot be an image: it must be
    (M, N), or (M, N, k) with k of 1, 3 or 4."""

    pyplot = sys.modules.get("matplotlib.pyplot")
    axes = sys.modules.get("matplotlib.axes")
    is_pyplot = pyplot is not None and site.func is getattr(pyplot, "imshow", None)
    is_method = (
        axes is not None and site.method == "imshow" and isinstance(site.receiver, axes.Axes)
        and getattr(site.func, "__func__", None) is axes.Axes.imshow
    )
    data = site.arg(0, "X")
    if not (is_pyplot or is_method) or not is_ndarray(data):
        return
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] not in (1, 3, 4)):
        site.crash("TypeError", f"Invalid shape {data.shape} for image data", [site.arg_root(0, "X")])


# --- PyTorch ---------------------------------------------------------------------------------


def _is_tensor(value) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and type(value) is torch.Tensor


@rule(Call, "matmul-shape")
def linear_layer_input_size(site: Call) -> None:
    """``layer(x)`` for an ``nn.Linear`` whose ``in_features`` differs from the
    size of ``x``'s last dimension. Only a plain ``nn.Linear`` with no hooks,
    whose forward is exactly the matrix product."""

    torch = sys.modules.get("torch")
    layer = site.func
    if torch is None or type(layer) is not torch.nn.Linear or len(site.args) != 1 or site.kwargs:
        return
    module = sys.modules.get("torch.nn.modules.module")
    hook_tables = [layer._forward_hooks, layer._forward_pre_hooks] + [
        getattr(module, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks")
    ]
    if any(hook_tables):
        return
    x = site.args[0]
    if not _is_tensor(x) or x.dim() == 0 or x.shape[-1] == layer.in_features:
        return
    rows = 1
    for n in x.shape[:-1]:
        rows *= n
    site.crash("RuntimeError", f"mat1 and mat2 shapes cannot be multiplied ({rows}x{x.shape[-1]} and "
               f"{layer.in_features}x{layer.out_features})", [site.arg_roots[0], site.func_root],
               text("checker.linear_size", data=site.arg_roots[0], size=x.shape[-1], layer=site.func_root,
                    expected=layer.in_features) if site.arg_roots[0] and site.func_root else "")


@rule(Call, "tensor-conversion")
def tensor_numpy_requires_grad(site: Call) -> None:
    """``t.numpy()`` on a tensor that requires grad."""

    tensor = site.receiver
    if _is_tensor(tensor) and site.method == "numpy" and tensor.requires_grad and not site.kwargs.get("force"):
        site.crash("RuntimeError", "Can't call numpy() on Tensor that requires grad. "
                   "Use tensor.detach().numpy() instead.", [site.receiver_root])


@rule(Call, "tensor-conversion")
def tensor_item_of_many(site: Call) -> None:
    """``t.item()`` on a tensor that does not hold exactly one value."""

    tensor = site.receiver
    if _is_tensor(tensor) and site.method == "item" and tensor.numel() != 1:
        site.crash("RuntimeError", f"a Tensor with {tensor.numel()} elements cannot be converted to Scalar",
                   [site.receiver_root])


@rule(BinOp, "matmul-shape")
def tensor_matmul_operator(site: BinOp) -> None:
    """``a @ b`` on tensors whose inner dimensions differ."""

    if site.op == "@" and _is_tensor(site.left) and _is_tensor(site.right):
        message = _matmul_error(tuple(site.left.shape), tuple(site.right.shape))
        if message:
            site.crash("RuntimeError", message, [site.left_root, site.right_root])


@rule(Call, "matmul-shape")
def tensor_matmul_function(site: Call) -> None:
    """``torch.matmul(a, b)`` whose inner dimensions differ."""

    torch = sys.modules.get("torch")
    if torch is None or site.func is not torch.matmul or len(site.args) != 2:
        return
    left, right = site.args
    if _is_tensor(left) and _is_tensor(right):
        message = _matmul_error(tuple(left.shape), tuple(right.shape))
        if message:
            site.crash("RuntimeError", message, site.arg_roots)


@rule(BinOp, "broadcast")
def tensor_broadcast(site: BinOp) -> None:
    """``a + b`` (and ``- * / // % **``) on tensors whose shapes do not broadcast."""

    if site.op not in ("+", "-", "*", "/", "//", "%", "**") or not (_is_tensor(site.left) and _is_tensor(site.right)):
        return
    try:
        numpy().broadcast_shapes(tuple(site.left.shape), tuple(site.right.shape))
    except ValueError:
        site.crash("RuntimeError", f"The size of tensor a ({tuple(site.left.shape)}) must match the size of "
                   f"tensor b ({tuple(site.right.shape)})", [site.left_root, site.right_root])


@rule(Call, "concatenate-shape")
def tensor_cat(site: Call) -> None:
    """``torch.cat([a, b])`` of tensors whose other dimensions differ."""

    torch = sys.modules.get("torch")
    if torch is None or site.func is not torch.cat or not site.args:
        return
    tensors = site.args[0]
    if type(tensors) not in (list, tuple) or not all(_is_tensor(t) for t in tensors):
        return
    # torch.cat still accepts the legacy empty 1-D tensor alongside anything.
    if any(tuple(t.shape) == (0,) for t in tensors):
        return
    dim = site.arg(1, "dim", 0)
    if type(dim) is not int:
        return
    message = _concatenate_error([tuple(t.shape) for t in tensors], dim)
    if message:
        site.crash("RuntimeError", message, site.arg_roots)


@rule(Call, "reshape-size")
def tensor_view_size(site: Call) -> None:
    """``t.view(3, 4)`` or ``t.reshape(3, 4)`` to a shape of a different size."""

    tensor = site.receiver
    if not _is_tensor(tensor) or site.method not in ("view", "reshape") or site.kwargs or not site.args:
        return
    shape = site.args[0] if len(site.args) == 1 else tuple(site.args)
    if type(shape) is int:
        shape = (shape,)
    if type(shape) not in (tuple, list) and type(shape).__name__ != "Size":
        return
    shape = tuple(shape)
    if not all(type(n) is int for n in shape):
        return
    message = _reshape_error(tensor.numel(), shape)
    if message:
        site.crash("RuntimeError", f"shape '{list(shape)}' is invalid for input of size {tensor.numel()}",
                   [site.receiver_root])


# --- shape arithmetic ---------------------------------------------------------


def _atleast(shape, ndim: int) -> tuple:
    """The shape ``np.atleast_1d`` (ndim=1) or ``np.atleast_2d`` (ndim=2) gives."""

    shape = tuple(shape)
    if len(shape) >= ndim:
        return shape
    if not shape:
        return (1,) * ndim
    return (1,) + shape


def _matmul_error(a, b) -> Optional[str]:
    if len(a) == 0 or len(b) == 0:
        return "matmul: Input operand does not have enough dimensions"
    left = a if len(a) >= 2 else (1,) + tuple(a)
    right = b if len(b) >= 2 else tuple(b) + (1,)
    if left[-1] != right[-2]:
        return (
            f"matmul: Input operand 1 has a mismatch in its core dimension 0 "
            f"(size {right[-2]} is different from {left[-1]})"
        )
    try:
        numpy().broadcast_shapes(tuple(left[:-2]), tuple(right[:-2]))
    except ValueError:
        return f"operands could not be broadcast together with shapes {shape_text(a)} {shape_text(b)}"
    return None


def _dot_error(a, b) -> Optional[str]:
    if len(a) == 0 or len(b) == 0:
        return None
    inner = b[0] if len(b) == 1 else b[-2]
    if a[-1] != inner:
        axis = 0 if len(b) == 1 else len(b) - 2
        return (
            f"shapes {shape_text(a)} and {shape_text(b)} not aligned: "
            f"{a[-1]} (dim {len(a) - 1}) != {inner} (dim {axis})"
        )
    return None


def _reshape_error(size: int, shape: tuple) -> Optional[str]:
    if not all(type(n) is int for n in shape):
        return None
    unknown = [n for n in shape if n == -1]
    if len(unknown) > 1:
        return "can only specify one unknown dimension"
    if any(n < -1 for n in shape):
        return "negative dimensions not allowed"
    known_size = 1
    for n in shape:
        if n != -1:
            known_size *= n
    shown = "(" + ",".join(str(n) for n in shape) + ")"
    if unknown:
        if known_size == 0 or size % known_size != 0:
            return f"cannot reshape array of size {size} into shape {shown}"
        return None
    if known_size != size:
        return f"cannot reshape array of size {size} into shape {shown}"
    return None


def _concatenate_error(shapes: List[tuple], axis: Optional[int]) -> Optional[str]:
    if not shapes:
        return "need at least one array to concatenate"
    if axis is None:
        return None
    ndim = len(shapes[0])
    if ndim == 0:
        return "zero-dimensional arrays cannot be concatenated"
    for index, shape in enumerate(shapes[1:], start=1):
        if len(shape) != ndim:
            return (
                "all the input arrays must have same number of dimensions, but the array at index 0 "
                f"has {ndim} dimension(s) and the array at index {index} has {len(shape)} dimension(s)"
            )
    if not -ndim <= axis < ndim:
        return f"axis {axis} is out of bounds for array of dimension {ndim}"
    axis %= ndim
    for index, shape in enumerate(shapes[1:], start=1):
        for dim in range(ndim):
            if dim != axis and shape[dim] != shapes[0][dim]:
                return (
                    "all the input array dimensions except for the concatenation axis must match "
                    f"exactly, but along dimension {dim}, the array at index 0 has size "
                    f"{shapes[0][dim]} and the array at index {index} has size {shape[dim]}"
                )
    return None


# =============================================================================
# Operations the checker may walk past
# =============================================================================
#
# After an operation, the checker either continues to the next one or stops
# and leaves the cell to the LLM. It may continue only past operations that
# change nothing, so that every later value is still exactly what the cell
# will see. If one of these raised instead, the cell would still crash, only
# earlier, so the verdict stays right.
#
# Adding a name here lets the rules see further into cells that use it.
# Only add operations that never modify their arguments, the receiver, or
# anything else, for any input.

# pandas methods that return a new object and change nothing, as long as
# ``inplace`` is not set. Their result is not computed.
PANDAS_PURE_METHODS = frozenset(
    {
        "head", "tail", "describe", "info", "nunique", "unique", "value_counts",
        "isnull", "isna", "notnull", "notna", "copy", "sum", "mean", "median",
        "min", "max", "std", "var", "count", "corr", "astype", "to_numpy",
        "drop", "dropna", "fillna", "rename", "reset_index", "sort_values",
        "sort_index", "set_index", "groupby", "select_dtypes", "duplicated",
        "drop_duplicates", "memory_usage", "abs", "round", "nlargest", "nsmallest",
        "idxmax", "idxmin", "any", "all", "get_dummies", "transpose", "reindex",
    }
)

# Attributes of pandas objects that are cheap to read and whose real value is
# kept. Other pandas attributes are side-effect free too, but may copy data,
# so they are left uncomputed.
PANDAS_CHEAP_ATTRIBUTES = frozenset(
    {"shape", "columns", "index", "dtypes", "dtype", "ndim", "size", "empty", "name", "names",
     "loc", "iloc",
     # Reading an accessor checks the dtype and raises AttributeError when it
     # does not fit, as .str does on a numeric column.
     "str", "dt", "cat"}
)

# NumPy array methods that return a new array and change nothing.
NUMPY_PURE_METHODS = frozenset(
    {
        "reshape", "astype", "copy", "sum", "mean", "min", "max", "std", "var",
        "flatten", "ravel", "transpose", "squeeze", "any", "all", "argmax",
        "argmin", "round", "tolist", "cumsum", "clip",
    }
)

# NumPy functions (``np.<name>``) that return a new array and change nothing.
NUMPY_PURE_FUNCTIONS = frozenset(
    {
        "array", "asarray", "zeros", "ones", "full", "arange", "linspace", "mean",
        "sum", "sqrt", "log", "exp", "abs", "max", "min", "unique", "argmax",
        "argmin", "std", "var", "median", "round", "isnan", "transpose", "squeeze",
        "expand_dims", "stack", "where", "cumsum", "clip", "eye", "dot", "matmul",
        "concatenate", "vstack", "hstack", "reshape",
    }
)

# NumPy constructors that are run for real when they build at most a million
# items, so that later rules see the actual shape, as in
# ``model.fit(X, np.ones(4))``.
NUMPY_BUILT_FOR_REAL = frozenset({"zeros", "ones", "full", "eye", "array", "asarray"})

# Libraries whose module-level ``__getattr__`` is a plain lookup, so that
# ``np.float`` can be checked by calling it.
TRUSTED_MODULE_GETATTR = frozenset({"numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn"})

# How the checker spells each operator in ``BinOp.op``.
OPERATORS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|",
    ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}


# How the checker spells each comparison in ``Compare.op``.
COMPARISONS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}
