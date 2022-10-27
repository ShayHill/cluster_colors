#!/usr/bin/env python3
# last modified: 221026 14:19:47
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
    # TODO: test cut_colors
    def test_cut_colors(self):
        """Call cut_colors with 100_000 random colors and pass result to stack_colors."""
        colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
        colors = stack_colors.stack_colors(colors)
        aaa = cut_colors.cut_colors(colors, 512)
    
# def time_cut_colors():
    # """Time the cut_colors function."""
    # import time
    # from cluster_colors import pool_colors
    # from cluster_colors import stack_colors
    # start = time.time()
    # colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
    # colors = stack_colors.add_weight_axis(colors, 255)
    # colors = stack_colors.stack_colors(colors)
    # colors = pool_colors.pool_colors(colors)
    # aaa = cut_colors(colors, 512)
    # print (time.time() - start)

# if __name__ == "__main__":
    # time_cut_colors()




