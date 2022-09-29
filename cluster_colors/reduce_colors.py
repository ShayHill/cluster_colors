#!/usr/bin/env python3
# last modified: 220921 21:48:01
"""Reduce colors by grouping into n-bit representions.

A reasonable way to reduce 24-bit colors (8 bits per channel, 16_777_216 possible
colors) to 1, 8, 64, 512, 4096, 32_768, 262_144, or 2_097_152 possible colors without
Scipy.

:author: Shay Hill
:created: 2022-09-19
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from cluster_colors.weighted_average import get_weighted_average

_T = TypeVar("_T")
_FPArray = npt.NDArray[np.floating[Any]]
_B8Array = npt.NDArray[np.uint8]


def _get_nbit(color: _FPArray, nbits: int) -> tuple[int, int, int]:
    """Get n-bit representation of color."""
    den = 2 ** (8 - nbits)
    r, g, b = color[:3]
    return r // den, g // den, b // den


def _add_weight(colors: _FPArray, alpha_is_transparency: bool = False) -> _FPArray:
    """Add or infer an alpha channel (to be used as weight).

    :param colors: array of rgb colors with shape (..., 3) or rgba with shape (..., 4)
    :param alpha_is_transparency: if True, infer alpha channel from 255 - transparency
    :return: array of rgba colors, with shape (..., 4)

    If alpha channel is missing, add it with value 255.
    If alpha channed is present, and alpha_is_transparency is True, invert it.
    If alpha channel is present, and alpha_is_transparency is False, do nothing.
    """
    if colors.shape[-1] == 4:
        if alpha_is_transparency:
            colors[..., 3] = 255 - colors[..., 3]
        return colors
    alpha = np.full(colors.shape[:-1], 255, dtype=float)
    return np.dstack((colors, alpha))  # type: ignore


def _rgb_to_nbit(colors: _FPArray, nbits: int) -> npt.NDArray[np.uint8]:
    """Convert 8-bit rgb portion of [r, g, b, a] colors to n-bit integers.

    :param colors: array of rgba colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of n-bit rgb colors, with shape (..., 3)
    """
    den = 2 ** (8 - nbits)
    return (colors[..., :3] // den).astype(np.uint8)


def _groupby_array(
        vvals: npt.NDArray[Any], vkeys: npt.NDArray[Any]
) -> dict[tuple[Any, ...], npt.NDArray[Any]]:
    """Group vvals by vkeys.

    :param vvals: array of values, with shape (..., n)
    :param vkeys: array of values, with shape (..., m)
        where vkeys.shape[:-1] == vvals.shape[:-1]
    :return: the last dimension of vvals grouped by the last dimension of vkeys
        tuple(vkey) : array of vvals with shape (-1, vvals.shape[-1])

    NOTE: vstack was slower
    """
    nbit2colors: dict[tuple[Any, ...], list[npt.NDArray[Any]]] = {}

    vvals_ = np.reshape(vvals, (-1, vvals.shape[-1]))  # type: ignore
    vkeys_ = np.reshape(vkeys, (-1, vkeys.shape[-1]))  # type: ignore
    for key, val in zip(vkeys_, vvals_):
        key_ = tuple(key)
        try:
            nbit2colors[key_].append(val)
        except KeyError:
            nbit2colors[key_] = [val]
    return {k: np.array(v) for k, v in nbit2colors.items()}


def _groupby_nbit(
        colors: _FPArray, nbits: Literal[1, 2, 3, 4, 5, 6, 7, 8]
) -> dict[tuple[int, int, int], _FPArray]:
    """Group colors by n-bit representation.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: dict of {n-bit representation: array of colors}
    :raises ValueError: if nbits outside [1..8]
    """
    if nbits not in range(1, 9):
        raise ValueError(f"nbits must be in [1..8], not {nbits}")
    nbit_representations = _rgb_to_nbit(colors, nbits)
    return _groupby_array(colors, nbit_representations)


def _map_averages_to_nbit(
        colors: _FPArray, nbits: Literal[1, 2, 3, 4, 5, 6, 7, 8]
) -> dict[tuple[int, int, int], _FPArray]:
    """Get reduced colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)
    """
    nbit2colors = _groupby_nbit(colors, nbits)
    return {k: get_weighted_average(v) for k, v in nbit2colors.items()}


from PIL import Image

img = Image.open("sugar-shack-barnes.jpg")
colors = np.array(img)
import time

start = time.time()
aaa = _groupby_nbit(colors, 3)
print(time.time() - start)
