#!/usr/bin/env python3
# last modified: 221001 20:12:17
"""Reduce colors by grouping into n-bit representions.

A reasonable and deterministic way to reduce 24-bit colors (8 bits per channel,
16_777_216 possible colors) to 1, 8, 64, 512, 4096, 32_768, 262_144, or 2_097_152
possible colors without Scipy.

:author: Shay Hill
:created: 2022-09-19
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from cluster_colors.stacked_quantile import get_stacked_medians

_FPArray = npt.NDArray[np.floating[Any]]
_NBits = Literal[1, 2, 3, 4, 5, 6, 7, 8]


def _rgb_to_nbit(colors: _FPArray, nbits: _NBits) -> npt.NDArray[np.uint8]:
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

    vvals_ = np.reshape(vvals, (-1, vvals.shape[-1]))
    vkeys_ = np.reshape(vkeys, (-1, vkeys.shape[-1]))
    for key, val in zip(vkeys_, vvals_):
        key_ = tuple(key)
        try:
            nbit2colors[key_].append(val)
        except KeyError:
            nbit2colors[key_] = [val]
    return {k: np.array(v) for k, v in nbit2colors.items()}


def _groupby_nbit(
    colors: _FPArray, nbits: _NBits
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


def _combine_weighted_colors(rgbws: _FPArray) -> _FPArray:
    """Get median color and sum of weights.

    :param rgbws: array of colors and weights, with shape (..., 4)
    :return: array of median color and sum of weights, with shape (..., 4)
    """
    rgbs, ws = rgbws[..., :3], rgbws[..., 3:]
    median_color = get_stacked_medians(rgbs, ws)
    return np.concatenate([median_color, np.sum(ws, axis=0)])  # type: ignore


def _map_averages_to_nbit(
    colors: _FPArray, nbits: _NBits
) -> dict[tuple[int, int, int], _FPArray]:
    """Get reduced colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)
    """
    nbit2colors = _groupby_nbit(colors, nbits)
    return {k: _combine_weighted_colors(v) for k, v in nbit2colors.items()}


def reduce_colors(colors: npt.NDArray[Any], nbits: _NBits) -> _FPArray:
    """Reduce 8-bit colors (each with a weight) to a maximum of (2**nbits)**3 colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)
    """
    nbit2colors = _map_averages_to_nbit(colors, nbits)
    return np.array(list(nbit2colors.values()))
