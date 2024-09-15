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

import random
from typing import TypeVar

import numpy as np
from PIL import Image

from cluster_colors import pool_colors
from cluster_colors.cluster_supercluster import DivisiveSupercluster
from cluster_colors.paths import TEST_DIR
from cluster_colors.vector_stacker import stack_vectors

_T = TypeVar("_T")

DIAG = np.array([(i, i, i, i + 1) for i in range(256)])

img = Image.open(TEST_DIR / "sugar-shack-barnes.jpg")
colors = np.array(img).reshape(-1, 1)[::16]
stacked_colors = stack_vectors(colors)


class TestPoolColors:
    def test_sum_weight(self):
        """Total weight of all colors is number of pixels."""
        img = Image.open(TEST_DIR / "sugar-shack-barnes.jpg")
        colors = np.array(img)[:100, :100]
        weights = np.full(colors.shape[:-1], 1, dtype=float)
        weighted_colors = stack_vectors(np.dstack((colors, weights)))
        reduced = pool_colors.pool_colors(weighted_colors, 4)
        assert np.sum(reduced[..., 3]) == colors.shape[0] * colors.shape[1]

    def test_robust_to_order(self):
        """Order of colors should not matter."""
        reduced = {tuple(x) for x in pool_colors.pool_colors(stacked_colors, 4)}
        reduced2 = {tuple(x) for x in pool_colors.pool_colors(stacked_colors[::-1], 4)}
        assert reduced == reduced2

    def test_robust_to_order_with_ties(self):
        """Order does not matter, even with ties."""

        def split_ten_times(vecs):
            clusters = DivisiveSupercluster.from_vectors(np.array(vecs))
            for _ in range(10):
                clusters.split()
            return {tuple(sorted(x.ixs)) for x in clusters.clusters}

        vecs_ = [(x, y, z) for x in range(5) for y in range(4) for z in range(3)]
        expect = split_ten_times(vecs_)
        for _ in range(10):
            random.shuffle(vecs_)
            result = split_ten_times(vecs_)
            assert result == expect

    def test_singles(self):
        """When color has a weight of 1 and does not stack, return same."""
        reduced = {tuple(x) for x in pool_colors.pool_colors(colors, 4)}
        reduced2 = {tuple(x) for x in pool_colors.pool_colors(colors[::-1], 4)}
        assert reduced == reduced2

    def test_small_number_of_identical_colors(self):
        """Do not return identical colors.

        Previousy, `pool_colors` would not try to pool when there were fewer input
        colors than the number of boxes in an nbits cube. This is to avoid needlessly
        discarding colors when few colors are present. This behavior caused problems:

        * identical colors were stacked by `stack_vectors`
        * *nearly* identical colors were not stacked or pooled.

        This caused problems later on, because the nearly identical colors *were*
        identical at 8-bits. Eventually, these colors created identical clusters,
        which broke the clustering algorithm.
        """
        colors = np.array(
            [
                [254.8, 0, 0, 1],
                [0, 254.9, 0, 1],
                [0, 254.8, 0, 1],
                [0, 254.7, 0, 1],
                [0, 0, 255, 1],
                [0, 0, 254.9, 1],
                [0, 0, 254.8, 1],
                [0, 0, 254.7, 1],
                [0, 0, 254.6, 1],
            ]
        )
        stacked_colors = stack_vectors(colors)
        reduced = sorted(map(tuple, pool_colors.pool_colors(stacked_colors, 4)))
        expect = [[0, 0, 254.8, 5], [0, 254.8, 0, 3], [254.8, 0, 0, 1]]
        np.testing.assert_array_almost_equal(reduced, expect)
