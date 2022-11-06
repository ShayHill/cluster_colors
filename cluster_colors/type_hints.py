#!/usr/bin/env python3
# last modified: 221028 15:44:36
"""Type hints for cluster_colors.

:author: Shay Hill
:created: 2022-10-22
"""

from typing import Annotated, Any, TypeAlias

import numpy as np
from numpy import typing as npt

# input pixel array, or something that looks like an input pixel array.
Pixels: TypeAlias = Annotated[npt.NDArray[np.number[Any]], "(..., -1)"]

# colors in the form of a 2D array of RGBW values.
StackedColors: TypeAlias = Annotated[npt.NDArray[np.floating[Any]], (-1, 4)]

# array that has been cast to float, but it not expected to have a weight axis or
# particular shape.
FPArray: TypeAlias = npt.NDArray[np.floating[Any]]
