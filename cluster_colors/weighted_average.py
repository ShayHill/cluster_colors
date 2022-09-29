#!/usr/bin/env python3
# last modified: 220929 09:48:12
"""Weighted average of cluster colors.

:author: Shay Hill
:created: 2022-09-21
"""

import numpy as np
import numpy.typing as npt
from typing import Any
import wquantiles as wq

_FPArray = npt.NDArray[np.floating[Any]]


def _apply_weight(colors: _FPArray) -> _FPArray:
    """Get a copy of the input array with rgb values multiplied by alpha.

    :param colors: array of rgbw colors, with shape (..., 4)
    :return: array of rgb colors, with shape (..., 3)

    Multiply each color by its weight (alpha channel). Convert to float so uint8
    values do not roll over when multiplied or summed.
    """
    return colors[..., :3].astype(float) * colors[..., 3:]


def get_weighted_average(colors: _FPArray) -> _FPArray:
    """Get the weighted average of the input colors. Keep full weight.

    :param colors: array of rgbw colors, with shape (..., 4)
    :return: array of average color, with shape (1, 4)
        [average_r, average_g, average_b, total_weight]
    :return: (0, 0, 0, 0) if alpha channed sums to 0
    :raises ValueError: if the input array is empty
    :raises ValueError: if the input array has no alpha channel

    Multiply each color by its weight (alpha channel). Sum the results. Divide by
    the sum of the weights.
    """
    if not colors.size:
        raise ValueError("Empty array")
    if colors.shape[-1] != 4:
        raise ValueError("No alpha channel to weigh by")
    full_weight = np.sum(colors[..., 3])  # type: ignore
    if not full_weight:
        raise ValueError("Cannot return median when every value has 0 weight")
        return np.zeros((4,))
    weighted = _apply_weight(colors)
    average_color = np.sum(weighted, axis=-2) / full_weight  # type: ignore
    return np.concatenate([average_color, [full_weight]])  # type: ignore
