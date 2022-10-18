#!/usr/bin/env python3
# last modified: 221018 10:27:43
"""Test reduce_colors() function.

:author: Shay Hill
:created: 2022-09-19
"""
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false

from typing import Literal, TypeVar

import numpy as np
import pytest

from cluster_colors import reduce_colors
from PIL import Image


_T = TypeVar("_T")

DIAG = np.array([(i, i, i, i + 1) for i in range(256)])


def _chunk(list_: list[_T], nbits: Literal[1, 2, 3, 4, 5, 6, 7, 8]) -> list[list[_T]]:
    """Split list into 2**nbits parts."""
    chunk_size = len(list_) // 2**nbits
    return [list_[i : i + chunk_size] for i in range(0, len(list_), chunk_size)]


@pytest.fixture(scope="module", params=range(1, 9))
def nbits(request) -> Literal[1, 2, 3, 4, 5, 6, 7, 8]:
    """Number of bits to use for color quantization"""
    return request.param


@pytest.fixture(scope="module")
def grouped_by_nbits(nbits) -> dict[tuple[int, int, int], reduce_colors._FPArray]:
    """Result of _group_by_nbit_representation for the diagonal."""
    return reduce_colors._groupby_nbit(DIAG, nbits)


@pytest.fixture(scope="module")
def averaged_by_nbits(nbits) -> dict[tuple[int, int, int], reduce_colors._FPArray]:
    """Result of _map_averages_to_nbit_representations for the diagonal."""
    return reduce_colors._map_averages_to_nbit(DIAG, nbits)


@pytest.fixture(scope="module")
def diag_chunks(nbits) -> list[list[float]]:
    """Expected weight boxes of diagonals."""
    return _chunk([x[3] for x in DIAG], nbits)


@pytest.fixture(scope="module")
def diag_sums(diag_chunks) -> list[float]:
    """Expected sums of diagonal sections."""
    return [sum(chunk) for chunk in diag_chunks]


@pytest.fixture(scope="module")
def diag_means(diag_chunks) -> list[float]:
    """Expected average of diagonal sections."""
    expected_means: list[float] = []
    for chunk in diag_chunks:
        expected_means.append(sum((x - 1) * x for x in chunk) / sum(chunk))
    return expected_means


class TestGroupByNbitRepresentation:
    def test_diagonal_into_2_to_the_nbits_groups(self, nbits, grouped_by_nbits):
        assert len(grouped_by_nbits.items()) == 2**nbits

    def test_diagonal_into_equal_sized_groups(self, nbits, grouped_by_nbits):
        assert all(
            len(group) == 256 // 2**nbits for group in grouped_by_nbits.values()
        )


class TestMapAveragesToNbitRepresentations:
    def test_weight_sums(self, diag_sums, averaged_by_nbits):
        for average, sum_weight in zip(averaged_by_nbits.values(), diag_sums):
            assert sum_weight == average[3]


class TestReduceColors:
    def test_sum_weight(self):
        """Total weight of all colors is number of pixels."""
        img = Image.open("test/sugar-shack-barnes.jpg")
        colors = np.array(img)
        weights = np.full(colors.shape[:-1], 1, dtype=float)
        weighted_colors = np.dstack((colors, weights))
        reduced = reduce_colors.reduce_colors(weighted_colors, 4)
        assert np.sum(reduced[..., 3]) == colors.shape[0] * colors.shape[1]

    def test_robust_to_order(self):
        """Order of colors should not matter."""
        img = Image.open("test/sugar-shack-barnes.jpg")
        colors = np.array(img)
        weights = np.full(colors.shape[:-1], 1, dtype=float)
        weighted_colors = np.dstack((colors, weights))
        reduced = {tuple(x) for x in reduce_colors.reduce_colors(weighted_colors, 4)}
        reduced2 = {
            tuple(x) for x in reduce_colors.reduce_colors(weighted_colors[::-1], 4)
        }
        assert reduced == reduced2
