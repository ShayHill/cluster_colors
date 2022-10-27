#!/usr/bin/env python3
# last modified: 221026 12:22:13
"""Add and manipulate a vector weight axis.

This project is built around combining identical vectors (presumably colors) into
single instances with a weight axis (reflecting the combined weight of the combined
weight of the combined vectors) then treating those single combined vectors as
multiples. For instance:

(1, 2, 3), (1, 2, 3), (4, 5, 6) would be stored as

(1, 2, 3, 2), (4, 5, 6, 1) but still treated as if it were

(1, 2, 3), (1, 2, 3), (4, 5, 6)

When working with pngs, there may be no need to add a weight channel, as the alpha
channel will serve the same function. Each pixel's alpha value will be interpreted as
the weight of that pixel.

The functions in this module return float arrays, not uint8 arrays. The reason's
being that float arrays will go out of range instead of wrapping around, which is
what we want (so we can identify and address it outside the module).

:author: Shay Hill
:created: 2022-10-18
"""

from typing import Annotated, Any, Optional

import numpy as np
cimport numpy as np
from numpy import typing as npt

import cython
cimport cython
from libc.stdlib cimport malloc, free
import pandas as pd

from cluster_colors.type_hints import FPArray, Pixels, StackedColors

cdef int weight
cdef int vector_length
cdef int w
cdef int i


def add_weight_axis(vectors, weight=0) -> FPArray:
    """Add a weight axis to a vector of vectors.

    :param vectors: A vector of vectors with shape (..., n).
    :param weight: The weight to add to each vector in the vector of vectors.
    :return: A vector of vectors with a weight axis. (..., n + 1)

    The default weight is 255, which is the maximum value of a uint8. This will
    reflect full opacity, which makes sense when working with color vectors.

    If these vectors will only ever be used to represent multiple, full instances,
    then the weight could be any value, as long as it is consistent. 1 might be a
    more intuitive value in that case, as a vector with v[-1] == n would be a vector
    with n instances.
    """
    # cdef np.ndarray[np.int_t] vectors
    assert weight > 0, "Weight must be greater than 0."
    ws = np.full(vectors.shape[:-1] + (1,), weight)
    return np.append(vectors, ws, axis=-1).astype(int)  # type: ignore


def stack_vectors(
    np.ndarray[np.int_t, ndim=2] flat_vectors
) -> Annotated[FPArray, (-1, -1)]:
    """Find and count unique vectors.

    :param vectors: array of numbers, with shape (..., n)
    :param weight: optionally provide a weight axis value.
        If not supplied, will assume last axis of each vector is a weight.
    :return: unique (by v[:-1]) with
        v[-1] equal to the sum of all v[-1] where v[:-1] == v[:-1]
    """
    # cdef np.ndarray[np.int_t] weighted_vectors
    # cdef np.ndarray[np.int_t, ndim=2] flat_vectors
    cdef np.ndarray[np.int_t, ndim=2] unique_vectors
    cdef np.ndarray[np.int_t, ndim=1] where_seen
    cdef np.ndarray[np.int_t, ndim=1] sum_weights
    cdef np.ndarray[np.int_t, ndim=2] weights
    cdef np.ndarray[np.int_t, ndim=2] vs
    cdef np.ndarray[np.int_t, ndim=2] ws
    cdef int i
    cdef int weight
    
    vs, ws = np.split(flat_vectors, [-1], axis=-1)
    unique_vectors, where_seen = np.unique(vs, return_inverse=True, axis=0)
    sum_weights = np.zeros(unique_vectors.shape[0], dtype=int)
    for i, w in zip(where_seen, ws):
        sum_weights[i] += w

    weights = np.array(sum_weights).reshape(-1, 1)
    return np.append(unique_vectors, sum_weights.reshape(-1, 1), axis=-1)  # type: ignore


def stack_colors(colors: Pixels) -> StackedColors:
    """Call stack_vectors with some inferences.

    :param colors: array of colors, with shape (..., 1), (..., 3) or (..., 4)
    :return: array of stacked colors, with shape (n, 4)
    :raises ValueError: if colors do not have shape (..., 1), (..., 3) or (..., 4)

    Assumes four-channel colors already have a weight in the rourth channel.
    Assumes three-channel vectors are opaque, 8-bit colors and adds a weight of 255.
    Assumes one-channel vectors are opaque, 8-bit greyscal colors
        and adds a weight of 255.
    """
    aaa = np.asarray(colors, dtype=int).reshape(-1, colors.shape[-1])
    return stack_vectors(aaa)

    cdef np.ndarray[np.int_t, ndim=2] flat_colors

    vector_length = colors.shape[-1]
    flat_colors = colors.astype(int).reshape(-1, colors.shape[-1])
    if vector_length == 4:
        return stack_vectors(flat_colors)
    elif vector_length == 3:
        flat_colors = add_weight_axis(flat_colors, 255)
        return stack_vectors(flat_colors)
    elif vector_length == 1:
        flat_colors = add_weight_axis(flat_colors, 255)
        return stack_vectors(flat_colors)
    else:
        raise ValueError(
            f"Expected colors to have shape (..., 1), (..., 3) or (..., 4), "
            f"but got {vector_length}."
        )
