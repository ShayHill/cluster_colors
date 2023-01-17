#!/usr/bin/env python3
# last modified: 221106 16:41:03
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

from typing import TypeVar

import numpy as np
from PIL import Image

from cluster_colors import pool_colors
from cluster_colors.paths import TEST_DIR
from cluster_colors.stack_vectors import stack_vectors

_T = TypeVar("_T")

DIAG = np.array([(i, i, i, i + 1) for i in range(256)])


class TestPoolColors:
    def test_sum_weight(self):
        """Total weight of all colors is number of pixels."""
        img = Image.open(TEST_DIR / "sugar-shack-barnes.jpg")
        colors = np.array(img)
        weights = np.full(colors.shape[:-1], 1, dtype=float)
        weighted_colors = np.dstack((colors, weights))
        reduced = pool_colors.pool_colors(weighted_colors, 4)
        assert np.sum(reduced[..., 3]) == colors.shape[0] * colors.shape[1]

    def test_robust_to_order(self):
        """Order of colors should not matter."""
        img = Image.open(TEST_DIR / "sugar-shack-barnes.jpg")
        colors = np.array(img)
        stacked_colors = stack_vectors(colors)
        reduced = {tuple(x) for x in pool_colors.pool_colors(stacked_colors, 4)}
        reduced2 = {tuple(x) for x in pool_colors.pool_colors(stacked_colors[::-1], 4)}
        assert reduced == reduced2

    def test_singles(self):
        """Single color should be returned."""
        img = Image.open(TEST_DIR / "sugar-shack-barnes.jpg")
        colors = np.array(img).reshape(-1, 1)
        stacked_colors = stack_vectors(colors)
        reduced = {tuple(x) for x in pool_colors.pool_colors(stacked_colors, 4)}
        reduced2 = {tuple(x) for x in pool_colors.pool_colors(stacked_colors[::-1], 4)}
        assert reduced == reduced2
