"""Rules for PyTorch tensors and layers: shapes that do not fit together, and
conversions a tensor does not allow.
"""

from __future__ import annotations

import sys

from ..sites import (
    BinOp,
    Call,
    numpy,
    rule,
)
from .shapes import matmul_error, reshape_error, concatenate_error
from ...texts import text


# --- shapes ------------------------------------------------------------------


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


@rule(BinOp, "matmul-shape")
def tensor_matmul_operator(site: BinOp) -> None:
    """``a @ b`` on tensors whose inner dimensions differ."""

    if site.op == "@" and _is_tensor(site.left) and _is_tensor(site.right):
        message = matmul_error(tuple(site.left.shape), tuple(site.right.shape))
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
        message = matmul_error(tuple(left.shape), tuple(right.shape))
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
    message = concatenate_error([tuple(t.shape) for t in tensors], dim)
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
    message = reshape_error(tensor.numel(), shape)
    if message:
        site.crash("RuntimeError", f"shape '{list(shape)}' is invalid for input of size {tensor.numel()}",
                   [site.receiver_root])


# --- conversions -------------------------------------------------------------


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
