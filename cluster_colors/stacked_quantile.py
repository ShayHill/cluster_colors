#!/usr/bin/env python3
# last modified: 220929 20:49:54
"""'Stacked' quantile functions. Close to weighted quantile functions.

These functions are used to calculate quantiles of a set of values, where each value
has a weight. The typical process for calculating a weighted quantile is to create a
CDF from the weights, then interpolate the values to find the quantile.

These functions, however, treat weighted values (given integer weights) exactly as
multiple values. So values (1, 2, 3) with weights (4, 5, 6) will be treated as
(1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3). If the quantile falls exactly between
two values, the non-weighted average of the two values is returned. This is
consistent with the "weights as occurrences" interpretation. Strips all zero-weight
values, so these will never be included in such averages.

If using non-integer weights, the results will be as if some scalar were applied to
make all weights into integers.

This "weights as occurrences" interpretation has two pitfalls:

    1.  Identical values will be returned for different quantiles (e.g., the results
        for quantiles == 0.5, 0.6, and 0.7 might be identical). The effect of this is
        that some some common data practices like "robust scalar" will *not* be
        robust because of the potential for a 0 interquartile range. Again this is
        consistent, because the same thing could happen with repeated, non-weighted
        values.

    2.  With any number of values, the stacked_median could still be the first or
        last value (if it has enough weight), so separating by the median is not
        robust. This could also happen with repeaded, non-weighted values. The
        workaround is to divide the values into group_a = values strictly < median,
        group_b = values strictly > median, then add == median to the smaller group.

:author: Shay Hill
:created: 2022-09-23
"""
from typing import Any, TypeAlias, cast, Iterable

import numpy as np
import numpy.typing as npt

_FPArray: TypeAlias = npt.NDArray[np.floating[Any]]


def get_stacked_quantile(
    values: Iterable[float] | _FPArray,
    weights: Iterable[float] | _FPArray,
    quantile: float,
) -> float:
    """Get a weighted quantile for a value.

    :param values: array of values with shape (n,)
    :param weights: array of weights where weights.shape == values.shape
    :param quantile: quantile to calculate, in [0, 1]
    :return: weighted quantile of values
    :raises ValueError: if values and weights do not have the same length
    :raises ValueError: if quantile is not in interval [0, 1]
    :raises ValueError: if values array is empty (after removing zero-weight values)
    :raises ValueError: if weights are not all positive
    """
    avalues: _FPArray = np.asarray(values)
    aweights: _FPArray = np.asarray(weights)
    if avalues.shape[-1] != aweights.shape[-1]:
        raise ValueError("values and weights must be the same length")
    if quantile < 0 or quantile > 1:
        raise ValueError("quantile must be in interval [0, 1]")

    avalues, aweights = avalues[aweights != 0], aweights[aweights != 0]

    if not len(avalues):
        raise ValueError("avalues empty (after removing zero-weight avalues)")
    if any(weight < 0 for weight in aweights):
        raise ValueError("aweights must be non-negative")

    sorter = np.argsort(avalues)
    sorted_values = avalues[sorter]
    sorted_weights = aweights[sorter]
    cum_aweights = cast(_FPArray, np.cumsum(sorted_weights))
    target = cum_aweights[-1] * quantile
    index = np.searchsorted(cum_aweights, target, side="right")
    if index == 0:
        return sorted_values[0]
    if np.isclose(cum_aweights[index - 1], target):
        lower = sorted_values[index - 1]
        upper = sorted_values[index]
        return (lower + upper) / 2
    return sorted_values[index]


def get_stacked_quantiles(
    values: _FPArray, weights: _FPArray, quantile: float
) -> _FPArray:
    """Get a weighted quantile for an array of vectors.

    :param values: array of vectors with shape (..., m)
        will return one m-length vector
    :param weights: array of weights with shape (..., 1)
        where shape[:-1] == values.shape[:-1]
    :param quantile: quantile to calculate, in [0, 1]
    :return: axiswise weighted quantile of an m-length vector
    :raises ValueError: if values and weights do not have the same shape[:-1]

    The "gotcha" here is that the weights must be passed as 1D vectors, not scalars.
    """
    if values.shape[:-1] != weights.shape[:-1]:
        raise ValueError(
            "values and weights must have the same shape up to the last axis"
        )
    flat_vectors = values.reshape(-1, values.shape[-1]).T
    flat_weights = weights.flatten()
    by_axis: list[float] = []
    for axis in flat_vectors:
        by_axis.append(get_stacked_quantile(axis, flat_weights, quantile))
    return np.array(by_axis)


def get_stacked_median(
    values: Iterable[float] | _FPArray, weights: Iterable[float] | _FPArray
) -> float:
    """Get a weighted median for a value.

    :param values: array of values with shape (n,)
    :param weights: array of weights where weights.shape == values.shape
    :return: weighted median of values
    :raises ValueError: if values and weights do not have the same length
    :raises ValueError: if values array is empty (after removing zero-weight values)
    :raises ValueError: if weights are not all positive
    """
    return get_stacked_quantile(values, weights, 0.5)


def get_stacked_medians(values: _FPArray, weights: _FPArray) -> _FPArray:
    """Get a weighted median for an array of vectors.

    :param values: array of vectors with shape (..., m)
        will return one m-length vector
    :param weights: array of weights with shape (..., 1)
        where shape[:-1] == values.shape[:-1]
    :return: axiswise weighted median of an m-length vector
    :raises ValueError: if values and weights do not have the same shape[:-1]

    The "gotcha" here is that the weights must be passed as 1D vectors, not scalars.
    """
    return get_stacked_quantiles(values, weights, 0.5)
