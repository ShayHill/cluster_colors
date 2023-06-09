"""Type hints for cluster_colors.

:author: Shay Hill
:created: 2022-10-22
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from numpy import typing as npt

# input pixel array, or something that looks like an input pixel array.
Pixels: TypeAlias = Annotated[npt.NDArray[np.number[Any]], "(..., -1)"]


# array that has been cast to float, but it not expected to have a weight axis or
# particular shape.
FPArray: TypeAlias = npt.NDArray[np.float_]

# a 1D array of floats
Vector: TypeAlias = Annotated[FPArray, (-1,)]
# something that can be cast to a vector
VectorLike: TypeAlias = Sequence[float] | Vector
# an array of vectors, expected to have a weight axis
StackedVectors: TypeAlias = Annotated[npt.NDArray[np.float_], (-1, -1)]

# number of bits in a color channel
NBits = Literal[1, 2, 3, 4, 5, 6, 7, 8]
