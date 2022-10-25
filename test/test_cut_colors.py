#!/usr/bin/env python3
# last modified: 221022 14:12:28
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
from cluster_colors import stack_colors
import numpy as np

class TestCutColors:
    def test_cut_colors(self):
        """Call cut_colors with 100_000 random colors and pass result to stack_colors."""
        colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
        colors = stack_colors.stack_colors(colors)
        aaa = cut_colors.cut_colors(colors, 512)
        breakpoint()
    




