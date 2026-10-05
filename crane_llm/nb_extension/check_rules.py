"""The built-in checker's rules: the crashes it can report as certain.

Each rule is a function that looks at one operation the target cell is about
to perform, with the real values it will be performed on, and calls
``site.crash(...)`` when that operation will certainly raise. The checker
(``checks.py``) walks the cell in evaluation order and offers every operation
it reaches to the rules registered for that kind of site.

Adding a rule
-------------

1. Pick the site: ``Subscript`` (``obj[key]``), ``Store`` (``obj[key] =
   value``), ``Call`` (``func(...)``, including methods) or ``BinOp``
   (``left + right`` and friends). Their fields are in ``check_helpers.py``.
2. Write a function decorated with ``@rule(<site>, "<rule-id>")``. Return
   when the rule does not apply; call ``site.crash(exception, message,
   variables, detail)`` when the operation will raise.
3. Only report what is certain. Check exact types (``is_frame``, ``type(x) is
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
on ``None`` and removed library APIs), unpacking the wrong number of values,
and syntax errors.

At the bottom of this file are the lists of operations the checker may walk
past without stopping. They decide how far into a cell the rules can see.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
from typing import List, Optional

from .check_helpers import (
    BinOp,
    Call,
    MISSING,
    SCALARS,
    Store,
    Subscript,
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
    sample_count,
    shape_text,
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
    {"shape", "columns", "index", "dtypes", "dtype", "ndim", "size", "empty", "name", "names"}
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
