#!/usr/bin/env python3
# last modified: 221026 13:54:06
"""Reduce colors by averaging colors with the same n-bit representation.

A reasonable and deterministic way to reduce 24-bit colors (8 bits per channel,
16_777_216 possible colors) to 1, 8, 64, 512, 4096, 32_768, 262_144, or 2_097_152
possible colors without Scipy.

On my laptop, this takes about 2 seconds to pool 500**2 colors to a maximum of
262_144. About 6 seconds to pool 1000**2.

Two-axis vectors (presumably <gray, weight>) should pass through unaffected because
they shouldn't have more than 256 unique values.

:author: Shay Hill
:created: 2022-09-19
"""

from typing import Annotated, Literal, Sequence, Callable
from itertools import chain, repeat

import numpy as np

from cluster_colors.stack_colors import stack_colors
from cluster_colors.type_hints import FPArray, Pixels

_8BitCube = Annotated[FPArray, (256, 256, 256, ...)]
_NBits = Literal[1, 2, 3, 4, 5, 6, 7, 8]
_FReduce = Callable[[FPArray, tuple[int]], FPArray]


def _pool(matrix: FPArray, kernel_shape: tuple[int], func: _FReduce) -> FPArray:
    """Pool a multi-dimensional array of numbers or arrays of numbers.

    :param matrix: array of numbers or arrays of numbers
    :param kernel_shape: shape of the kernel to pool with
    :param func: function to reduce the kernel with
    :return: pooled array

    Given an array (*dims) and a kernel shape (*kernel_dims), pool the array by func
    to each kernel_shape subarray.
    """
    assert all(v % k == 0 for v, k in zip(matrix.shape, kernel_shape))

    matrix_shape = matrix.shape[:len(kernel_shape)]
    vector_shape = matrix.shape[len(kernel_shape):]
    folded_dims = [(v // k, k) for v, k in zip(matrix_shape, kernel_shape)]
    pools_shape = tuple(chain(*folded_dims))
    reshaped = matrix.reshape(pools_shape + vector_shape)

    return func(reshaped, axis=tuple(range(len(pools_shape)))[1::-1])


def _pool_8bit_cube(colors: _8BitCube, nbits: _NBits) -> FPArray:
    """Sum values by n-bit representation of their indices.

    :param colors: array of colors, with shape (256, 256, 256, 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (n, n, n, 4) where n is 2**nbits
    """
    block = 2 ** (8 - nbits)
    kernel_shape = tuple(block for _ in colors.shape[:-1])
    return _pool(colors, kernel_shape, np.sum)


def pool_colors(colors: Pixels, nbits: _NBits = 6) -> FPArray:
    """Reduce 8-bit colors (each with a weight) to a maximum of (2**nbits)**3 colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)

    Create a (256,) * color_channels matrix. Place each color at its rgb coordinate
    in that matrix. Scale these colors by their weight so that later on, sum /
    combined weight will give an average color.

    Sum colors at adjacent coordinates then divine rgb by total weight.
    """
    num_axes = colors.shape[-1]
    max_colors = (2**nbits) ** (num_axes - 1)
    if nbits == 8 or len(colors) <= max_colors:
        return colors

    colorspace = np.zeros((256,) * (num_axes - 1) + (num_axes,), dtype=np.float64)
    for *vs, w in colors:
        colorspace[tuple(int(x) for x in vs)] = np.array(vs + [1]) * w

    colorspace = _pool_8bit_cube(colorspace, nbits)
    colors = colorspace.reshape(-1, num_axes)
    colors = colors[colors[:, -1] > 0]
    colors[:, :-1] /= colors[:, -1:]
    return colors
