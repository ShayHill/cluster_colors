"""One member of a cluster.

I left some flexibility in this library to work with vectors of any length, but there
are specialized methods for working with RGB colors. These will fail if the vector is
not 3 values long.

:author: Shay Hill
:created: 2024-09-01
"""

from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING, Literal
import numpy as np
from stacked_quantile import get_stacked_median, get_stacked_medians

from basic_colormath import float_tuple_to_8bit_int_tuple, rgb_to_lab

if TYPE_CHECKING:
    from cluster_colors.type_hints import FPArray, StackedVectors, Vector, VectorLike

def _cmp(a: float, b: float) -> Literal[-1, 0, 1]:
    """Compare two floats.

    :param a: float
    :param b: float
    :return: -1 if a < b, 0 if a == b, 1 if a > b
    """
    cmp = int(a > b) - int(a < b)
    if cmp == -1:
        return -1
    if cmp == 1:
        return 1
    return 0

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


def split_members_by_plane(members: set[Member], abc: FPArray) -> tuple[set[Member], set[Member]]:
    """Split members into two sets based on their relative distance from a plane.

    :param members: set of Member instances
    :param abc: (a, b, c) of the plane equation ax + by + cz = 0
    :return: two sets of Member instances one on each side of the plane. Member
        instances exactly on the plane will be included in the smaller set.
    :raises ValueError: if all members are on one side of the plane

    The splitting is a bit funny due to innate characteristice of the stacked
    median. It is possible to get a split with members
        a) on one side of the splitting plane; and
        b) exactly on the splitting plane.
    See stacked_quantile module for details, but that case is covered here.
    """
    scored: list[tuple[float, Member]] = [(np.dot(abc, m.vs), m) for m in members]
    scores = np.array([s for s, _ in scored])
    weights = np.array([m.w for _, m in scored])
    median_score = get_stacked_median(scores, weights)

    lteqgt: tuple[set[Member], set[Member], set[Member]] = (set(), set(), set())
    for score, member in scored:
        lteqgt[_cmp(score, median_score) + 1].add(member)
    lt, eq, gt = lteqgt
    lt_wt, eq_wt, gt_wt = (sum(y.w for y in x) for x in lteqgt)

    if sum(1 for x in (lt_wt, eq_wt, gt_wt) if x) < 2:
        msg = "All members on one side of the plane."
        raise ValueError(msg)

    if not gt:
        return lt, eq
    if not lt:
        return eq, gt
    if sum(m.w for m in lt) < sum(m.w for m in gt):
        return lt | eq, gt
    return lt, eq | gt
