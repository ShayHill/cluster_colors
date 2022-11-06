#!/usr/bin/env python3
# last modified: 221106 16:42:15
"""Test cut_colors.py

:author: Shay Hill
:created: 2022-10-22
"""
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

from cluster_colors import cut_colors
from cluster_colors import stack_vectors
import numpy as np

class TestCutColors:
    # TODO: test cut_colors
    def test_cut_colors(self):
        """Call cut_colors with 100_000 random colors and pass result to stack_vectors."""
        colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
        colors = stack_colors.stack_vectors(colors)
        aaa = cut_colors.cut_colors(colors, 512)
