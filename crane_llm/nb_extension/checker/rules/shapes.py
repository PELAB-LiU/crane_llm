"""Shape arithmetic shared by the NumPy and PyTorch rules: the error each
operation gives for shapes that do not fit together, or None.
"""

from __future__ import annotations

from typing import List, Optional

from ..sites import (
    numpy,
    shape_text,
)


def atleast(shape, ndim: int) -> tuple:
    """The shape ``np.atleast_1d`` (ndim=1) or ``np.atleast_2d`` (ndim=2) gives."""

    shape = tuple(shape)
    if len(shape) >= ndim:
        return shape
    if not shape:
        return (1,) * ndim
    return (1,) + shape


def matmul_error(a, b) -> Optional[str]:
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


def dot_error(a, b) -> Optional[str]:
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


def reshape_error(size: int, shape: tuple) -> Optional[str]:
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


def concatenate_error(shapes: List[tuple], axis: Optional[int]) -> Optional[str]:
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
