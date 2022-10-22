#!/usr/bin/env python3
# last modified: 221022 11:28:37
"""Reduce colors by averaging colors with the same n-bit representation.

A reasonable and deterministic way to reduce 24-bit colors (8 bits per channel,
16_777_216 possible colors) to 1, 8, 64, 512, 4096, 32_768, 262_144, or 2_097_152
possible colors without Scipy.

On my laptop, this takes about 2 seconds to pool 500**2 colors to a maximum of
262_144. About 6 seconds to pool 1000**2.

:author: Shay Hill
:created: 2022-09-19
"""

from typing import Annotated, Literal

import numpy as np

from cluster_colors.stack_colors import stack_colors
from cluster_colors.type_hints import FPArray, Pixels

_8BitCube = Annotated[FPArray, (256, 256, 256, ...)]
_NBits = Literal[1, 2, 3, 4, 5, 6, 7, 8]


def _pool_8bit_cube(colors: _8BitCube, nbits: _NBits) -> FPArray:
    """Sum values by n-bit representation of their indices.

    :param colors: array of colors, with shape (256, 256, 256, 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (n, n, n, 4) where n is 2**nbits
    """
    block = 2 ** (8 - nbits)
    da, db, dc = (x // block for x in colors.shape[:3])
    return colors.reshape(da, block, db, block, dc, block, 4).sum(axis=(1, 3, 5))  # type: ignore


def pool_colors(colors: Pixels, nbits: _NBits = 6) -> FPArray:
    """Reduce 8-bit colors (each with a weight) to a maximum of (2**nbits)**3 colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)

    Create a 256 x 256 x 256 matrix. Place each color at it's rgb coordinate in that
    matrix. Scale these colors by their weight so that later on, sum / combined
    weight will give an average color.

    Sum colors at adjacent coordinates then divine rgb by total weight.
    
    # TODO: refactor this to only accept stacked colors
    """
    colors = stack_colors(colors)
    if nbits == 8 or len(colors) <= 2**nbits:
        return colors

    colorspace = np.zeros((256, 256, 256, 4), dtype=np.float64)
    for r, g, b, w in colors:
        colorspace[int(r), int(g), int(b)] = np.array([r, g, b, 1]) * w

    colorspace = _pool_8bit_cube(colorspace, nbits)
    colors = colorspace.reshape(-1, 4)
    colors = colors[colors[:, 3] > 0]
    colors[:, :3] /= colors[:, 3:]
    return colors
