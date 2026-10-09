"""Rules for matplotlib.
"""

from __future__ import annotations

import sys

from ..sites import (
    Call,
    is_ndarray,
    rule,
)


# --- images ------------------------------------------------------------------


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
