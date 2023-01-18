#!/usr/bin/env python3
# last modified: 230118 10:54:03
"""Reduce colors by averaging colors with the same n-bit representation.

A reasonable and deterministic way to reduce 24-bit colors (8 bits per channel,
16_777_216 possible colors) to 1, 8, 64, 512, 4096, 32_768, 262_144, or 2_097_152
possible colors without Scipy.

Two-axis vectors (presumably <gray, weight>) should pass through unaffected because
they shouldn't have more than 256 unique values.

:author: Shay Hill
:created: 2022-09-19
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from itertools import chain
from typing import Annotated, Callable

import numpy as np

from cluster_colors.type_hints import FPArray, NBits

_8BitCube = Annotated[FPArray, (256, 256, 256, ...)]
_FReduce = Callable[[FPArray, tuple[int, ...]], FPArray]


def _pool(matrix: FPArray, kernel_shape: tuple[int, ...], func: _FReduce) -> FPArray:
    """Pool a multi-dimensional array of numbers or arrays of numbers.

    :param matrix: array of numbers or arrays of numbers
    :param kernel_shape: shape of the kernel to pool with
    :param func: function to reduce the kernel. Must accept an array and axes as a
        tuple of floats.
    :return: pooled array

    Given an array (*dims) and a kernel shape (*kernel_dims), pool the array by func
    to each kernel_shape subarray.

    This expects the matrix to be a multiple of the kernel_shape in each dimension.
    For example, start with a 12x12 matrix. Pool this with a matix of 3x3. The result
    will be a 4x4 matrix. If you pool this with a 2x2 kernel, you'll get a 6x6
    matrix. If you pool this with a 4x4 kernel, you'll get a 3x3 matrix.

    The value of func will determine how the kernel is reduced. For example, if you
    use sum, the [0,0] value of a 4x4 matrix pooled to a 2x2 matrix will be the sum
    of the 16 values in the 4x4 matrix.
    """
    assert all(v % k == 0 for v, k in zip(matrix.shape, kernel_shape))
    matrix_shape = matrix.shape[: len(kernel_shape)]
    vector_shape = matrix.shape[len(kernel_shape) :]
    folded_dims = [(v // k, k) for v, k in zip(matrix_shape, kernel_shape)]
    pools_shape = tuple(chain(*folded_dims))
    reshaped = matrix.reshape(pools_shape + vector_shape)
    result = func(reshaped, tuple(range(len(matrix_shape))))
    return result


def _pool_8bit_cube(colors: FPArray, nbits: NBits) -> FPArray:
    """Sum values by n-bit representation of their indices.

    :param colors: array of colors, with shape (256, 256, 256, 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (n, n, n, 4) where n is 2**nbits
    """
    block = 2 ** (8 - nbits)
    kernel_shape: tuple[int, ...] = tuple(block for _ in colors.shape[:-1])
    return _pool(colors, kernel_shape, np.sum)


def _fill_colorspace(colors: FPArray) -> FPArray:
    """Fill values by n-bit representation of their indices.

    :param colors: array of colors, with shape (-1, 4)
    :return: a 256x256x256x4 array, where for every (..., w) in colors,
        result[...] = (...*w, w)
        e.g.,
        for every (r, g, b, w) in colors,
        result[r, g, b] = (r*w, g*w, b*w, w)

    Will fill any number of color channels, assuming all are 8-bit.
    """
    num_axes = colors.shape[-1]
    colorspace_shape = (256,) * (num_axes - 1) + (num_axes,)
    colorspace = np.zeros(colorspace_shape, dtype="float")
    ixs = colors[..., :-1].astype(int)
    wss = colors[..., -1:]
    vss = ixs.astype(float) * wss
    colorspace[*ixs.T] = np.concatenate((vss, wss), axis=-1)
    return colorspace


def pool_colors(colors: FPArray, nbits: NBits = 6) -> FPArray:
    """Reduce 8-bit colors (each with a weight) to a maximum of (2**nbits)**3 colors.

    :param colors: array of colors, with shape (..., 4)
    :param nbits: number of bits per channel
    :return: array of reduced colors, with shape (..., 4)

    Create a (256,) * color_channels matrix. Place each color at its rgb coordinate
    in that matrix. Scale these colors by their weight so that later on, sum /
    combined weight will give an average color.

    Sum colors at adjacent coordinates then divine rgb by total weight.

    Will skip this entirely if you have fewer than 2**nbits-per-channel colors.
    There's no reason to discard information if you already have a workable amount of
    colors.
    """
    num_axes = colors.shape[-1]
    max_colors = (2**nbits) ** (num_axes - 1)
    if nbits == 8 or len(colors) <= max_colors:
        return colors

    colorspace = _fill_colorspace(colors)
    colorspace = _pool_8bit_cube(colorspace, nbits)
    colors = colorspace.reshape(-1, num_axes)
    colors = colors[colors[:, -1] > 0]
    colors[:, :-1] /= colors[:, -1:]
    return colors
