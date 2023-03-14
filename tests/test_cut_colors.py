"""Test cut_colors.py

:author: Shay Hill
:created: 2022-10-22
"""
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

import numpy as np
import numpy.typing as npt
from typing import Annotated

from cluster_colors import cut_colors, stack_vectors as sv

import pytest

ColorsArray = Annotated[npt.NDArray[np.float_], (-1, 3)]


@pytest.fixture(
    scope="function",
    params=[np.random.randint(0, 255, (1000, 3), dtype=np.uint8) for _ in range(10)],
)
def colors(request) -> ColorsArray:
    return request.param


class TestCutColors:
    # TODO: test cut_colors
    def test_cut_colors(self, colors: ColorsArray):
        """Call cut_colors with 100_000 random colors and pass result to stack_vectors."""
        # colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
        colors = sv.stack_vectors(colors)
        aaa = cut_colors.cut_colors(colors, 512)
