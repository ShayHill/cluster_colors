"""Test cut_colors.py

:author: Shay Hill
:created: 2022-10-22
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from cluster_colors import cut_colors
from cluster_colors import vector_stacker as sv

ColorsArray = Annotated[npt.NDArray[np.floating], (-1, 3)]

rng = np.random.default_rng(42)


@pytest.fixture(
    params=[rng.integers(0, 255, size=(1000, 3), dtype=np.uint8) for _ in range(10)]
)
def colors(request: pytest.FixtureRequest) -> ColorsArray:
    return request.param


class TestCutColors:
    def test_cut_colors(self, colors: ColorsArray) -> None:
        """Call cut_colors with 100_000 random colors and pass to stack_vectors."""
        colors = sv.stack_vectors(colors)
        _ = cut_colors.cut_colors(colors, 512)
