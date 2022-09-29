#!/usr/bin/env python3
# last modified: 220926 11:09:20
"""Weighted median of floats.

:author: Shay Hill
:created: 2022-09-23
"""
import numpy as np
import numpy.typing as npt
from typing import TypeAlias, Any, cast

_FPArray: TypeAlias = npt.NDArray[np.floating[Any]]


def stacked_quantile(values: _FPArray, weights: _FPArray, quantile: float) -> float:
    """Get a weighted quantile for a value.

    :param values: array of values with shape (n,)
    :param weights: array of weights where weights.shape == values.shape
    :param quantile: quantile to calculate
    :return: itemwise weighted quantile of values
    :raises ValueError: if values and weights do not have the same length
    :raises ValueError: if quantile is not in the range [0, 1]
    :raises ValueError: if values array is empty
    :raises ValueError: if weights are not all positive
    :raises ValueError: if weights sum to zero
    """
    if values.shape[-1] != weights.shape[-1]:
        raise ValueError("values and weights must be the same length")
    if quantile < 0 or quantile > 1:
        raise ValueError("quantile must be in interval [0, 1]")
    if not len(values):
        raise ValueError("values must not be empty")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) == 0:
        raise ValueError("weights must not sum to zero")

    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cum_weights = cast(_FPArray, np.cumsum(sorted_weights))
    target = cum_weights[-1] * quantile
    index = np.searchsorted(cum_weights, target, side="right")
    # if index == len(sorted_values):
        # return sorted_values[-1]
    if index == 0:
        return sorted_values[0]
    if np.isclose(cum_weights[index - 1], target):
        lower = sorted_values[index - 1]
        upper = sorted_values[index]
        return (lower + upper) / 2
    return sorted_values[index]


def stacked_quantiles(
    values: _FPArray, weights: _FPArray, quantile: float
) -> float | _FPArray:
    """Get a weighted quantile for an array of values or vectors.

    :param values: array of values or vectors
        if len(values.shape) == 1, then values are treates as scalars
        if len(values.shape) > 1, then values are treated as vectors
    :param weights: array of weights where weights.shape[:-1] == values.shape[:-1]
    :return: axiswise weighted quantile of values or vectors
    :raises ValueError: if values and weights do not have the same shape[:-1]
    """
    if len(values.shape) == 1:
        return stacked_quantile(values, weights, quantile)

    if values.shape[:-1] != weights.shape[:-1]:
        raise ValueError(
            "values and weights must have the same shape up to the last axis"
        )
    flat_vectors = values.reshape(-1, values.shape[-1])
    flat_weights = weights.reshape(-1, weights.shape[-1])
    by_axis: list[float] = []
    for axis in flat_vectors.T:
        by_axis.append(stacked_quantile(axis, flat_weights, quantile))
    return np.array(by_axis)


if __name__ == "__main__":
    print(stacked_quantile(np.array([1, 2]), np.array([1, 1]), 0.5))
