"""Rules for NumPy arrays: indexing, shapes that do not fit together, axes
out of range, and values a function cannot take.
"""

from __future__ import annotations

from ..sites import (
    BinOp,
    Call,
    SCALARS,
    Subscript,
    Truth,
    is_ndarray,
    is_numpy_integer,
    is_one_of,
    numpy,
    rule,
    scan_budget,
    shape_text,
)
from .shapes import atleast, matmul_error, dot_error, reshape_error, concatenate_error
from ...texts import text


# --- indexing ----------------------------------------------------------------


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


# --- shapes ------------------------------------------------------------------


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
        message = matmul_error(site.left.shape, site.right.shape)
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
            dot_error(left.shape, right.shape) if site.func is np.dot
            else matmul_error(left.shape, right.shape)
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
            shapes, axis = [atleast(a.shape, 2) for a in arrays], 0
        else:
            shapes = [atleast(a.shape, 1) for a in arrays]
            axis = 0 if shapes and all(len(s) == 1 for s in shapes) else 1
    if axis is not None and type(axis) is not int:
        return
    message = concatenate_error(shapes, axis)
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
    message = reshape_error(array.size, tuple(shape))
    if message:
        detail = text("checker.array_shape", name=root, shape=shape_text(array.shape)) if root else ""
        site.crash("ValueError", message, [root], detail)


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


# --- values ------------------------------------------------------------------


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


@rule(Call, "astype-int")
def array_astype_integer(site: Call) -> None:
    """``a.astype(int)`` on an array of text with a value that is not an integer.
    Reads every value, within the scan limit."""

    np = numpy()
    array = site.receiver
    if np is None or not is_ndarray(array) or site.method != "astype" or array.dtype.kind not in "USO":
        return
    dtype = site.arg(0, "dtype")
    if not is_numpy_integer(dtype) or array.size > scan_budget():
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


@rule(Truth, "ambiguous-truth")
def array_ambiguous_truth(site: Truth) -> None:
    """``if arr:`` for an array of more than one element."""

    if is_ndarray(site.value) and site.value.size > 1:
        site.crash("ValueError", "The truth value of an array with more than one element is ambiguous. "
                   "Use a.any() or a.all()", [site.root])
