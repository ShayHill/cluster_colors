"""Add and manipulate a vector weight axis.

This project is built around combining identical vectors (presumably colors) into
single instances with a weight axis (reflecting the combined weight of the combined
vectors) then treating those single combined vectors as multiples. For instance:

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

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from typing import Any

import numpy as np
from numpy import typing as npt

from cluster_colors.type_hints import FPArray, Vectors


def add_weight_axis(
    vectors: npt.NDArray[np.number[Any]], weight: float = 255.0
) -> FPArray:
    """Add a weight axis to a vector of vectors.

    :param vectors: A vector of vectors with shape (..., n).
    :param weight: The weight to add to each vector in the vector of vectors.
    :return: A vector of vectors with a weight axis. (..., n + 1)
    :raise ValueError: If the weight is not a positive number.

    The default weight is 255, which is the maximum value of a uint8. This will
    reflect full opacity, which makes sense when working with color vectors.

    If these vectors will only ever be used to represent multiple, full instances,
    then the weight could be any value, as long as it is consistent. 1 might be a
    more intuitive value in that case, as a vector with v[-1] == n would be a vector
    with n instances.
    """
    if weight <= 0:
        msg = f"Weight must be greater than 0. Got {weight}."
        raise ValueError(msg)
    ws = np.ones(vectors.shape[0]).reshape(-1, 1) * weight
    return np.hstack([vectors, ws]).astype(float)


def stack_vectors(
    vectors: npt.NDArray[np.number[Any]], weight: float | None = None
) -> Vectors:
    """Find and count unique vectors.

    :param vectors: array of numbers, with shape (..., n)
    :param weight: optionally provide a weight axis value.
        If not supplied, will assume last axis of each vector is a weight.
    :return: unique (by v[:-1]) with
        v[-1] equal to the sum of all v[-1] where v[:-1] == v[:-1]
    """
    if weight is not None:
        vectors = add_weight_axis(vectors, weight)

    flat_vectors = vectors.reshape(-1, vectors.shape[-1]).astype(float)

    unique_vectors, where_seen = np.unique(
        flat_vectors[:, :-1], return_inverse=True, axis=0
    )
    idx2seen = [0.0] * len(unique_vectors)
    for i, idx in enumerate(where_seen):
        idx2seen[idx] += flat_vectors[i, -1]
    weights = np.array(idx2seen).reshape(-1, 1)
    return np.append(unique_vectors, weights, axis=-1)
