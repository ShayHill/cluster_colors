"""One member of a cluster.

I left some flexibility in this library to work with vectors of any length, but there
are some specialized methods for working with RGB colors. These will fail if the
vector is not 3 values long.

:author: Shay Hill
:created: 2024-09-01
"""

from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING

from basic_colormath import float_tuple_to_8bit_int_tuple, rgb_to_lab

if TYPE_CHECKING:
    from cluster_colors.type_hints import FPArray, StackedVectors, Vector


class Member:
    """A member of a cluster.

    This will work with any weighted vector (any vector with one extra value on the
    last axis for weight).

    When clustering initial image arrays returned from `stack_image_colors`, the
    weight axis will only represent the number of times the color appears in the
    image. After removing some color or adding an alpha channel, the weight will also
    reflect the alpha channel, with transparent colors weighing less.
    """

    def __init__(self, weighted_vector: Vector) -> None:
        """Create a new Member instance.

        :param weighted_vector: a vector with a weight in the last axis
            (r, g, b, w)
        :param ancestors: sets of ancestors to merge
        """
        self.as_array = weighted_vector

    @property
    def vs(self) -> FPArray:
        """All value axes of the Member as a tuple.

        :return: tuple of values that are not the weight
            the (r, g, b) in (r, g, b, w)
        """
        return self.as_array[:-1]

    @property
    def w(self) -> float:
        """Weight of the Member.

        :return: weight of the Member
            the w in (r, g, b, w)
        """
        return self.as_array[-1]

    @property
    def rgb_floats(self) -> tuple[float, float, float]:
        """The color of the Member.

        :return: (r, g, b) of the Member

        This will only work with vectors that have a 3 value axis.
        """
        r, g, b = self.vs
        return (r, g, b)

    @ft.cached_property
    def rgb(self) -> tuple[int, int, int]:
        """The color of the Member as 8-bit integers.

        :return: (r, g, b) of the Member

        This will only work with vectors that have a 3 value axis.
        """
        return float_tuple_to_8bit_int_tuple(self.rgb_floats)

    @ft.cached_property
    def lab(self) -> tuple[float, float, float]:
        """The color of the Member in CIELAB space.

        :return: (L, a, b) of the Member

        This will only work with vectors that have a 3 value axis.
        """
        return rgb_to_lab(self.rgb)

    @classmethod
    def new_members(cls, stacked_vectors: StackedVectors) -> set[Member]:
        """Transform an array of vectors into a set of Member instances.

        :param stacked_vectors: (-1, n + 1) a list of vectors with weight channels in
            the last axis
        :return: set of Member instances
        """
        return {Member(v) for v in stacked_vectors if v[-1]}
