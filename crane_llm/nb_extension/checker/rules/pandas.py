"""Rules for pandas: missing columns and labels, positions out of range,
conversions that fail, and operands of the wrong length.
"""

from __future__ import annotations

from typing import Optional

from ..sites import (
    BinOp,
    Call,
    Compare,
    MISSING,
    SCALARS,
    Store,
    Subscript,
    Truth,
    indexer_target,
    is_frame,
    is_label,
    is_ndarray,
    is_numpy_integer,
    is_series,
    labels,
    numpy,
    pandas,
    plain_axis,
    rule,
    scan_budget,
    shape_text,
)
from ...texts import text


# --- columns -----------------------------------------------------------------


def _columns_detail(root: Optional[str], columns) -> str:
    try:
        names = [str(c) for c in list(columns)[:12]]
    except Exception:
        return ""
    listed = ", ".join(names) + (", ..." if len(columns) > 12 else "")
    if root:
        return text("checker.columns", name=root, columns=listed)
    return text("checker.columns_unnamed", columns=listed)


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


# --- positions and labels ----------------------------------------------------


@rule(Subscript, "index-range")
def pandas_iloc_out_of_range(site: Subscript) -> None:
    """``df.iloc[n]`` or ``s.iloc[n]`` past the end."""

    target = indexer_target(site.obj, "_iLocIndexer")
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

    target = indexer_target(site.obj, "_LocIndexer")
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


# --- converting --------------------------------------------------------------


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
        if not is_series(series) or not is_numpy_integer(dt):
            continue
        if len(series) > budget:
            return
        budget -= len(series)
        problem = _integer_cast_problem(series, dt)
        if problem is not None:
            exception, message, value = problem
            site.crash(exception, message, [site.receiver_root],
                       text("checker.bad_integer_value", value=repr(value)))


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


# --- operators and truth -----------------------------------------------------


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


@rule(Truth, "ambiguous-truth")
def dataframe_ambiguous_truth(site: Truth) -> None:
    """``if df:``, ``not series``: a DataFrame or Series has no single truth value."""

    value = site.value
    if is_frame(value) or is_series(value):
        kind = "DataFrame" if is_frame(value) else "Series"
        site.crash("ValueError", f"The truth value of a {kind} is ambiguous. Use a.empty, a.bool(), "
                   "a.item(), a.any() or a.all().", [site.root])


# --- concatenating -----------------------------------------------------------


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
