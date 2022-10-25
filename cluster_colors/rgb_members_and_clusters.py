from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, DefaultDict, Iterable, TypeAlias, cast, Sequence

import numpy as np
from numpy import cov as np_cov  # type: ignore
from numpy import typing as npt

from cluster_colors.stacked_quantile import get_stacked_median, get_stacked_medians

np_cov = cast(Any, np_cov)

_RgbF = tuple[float, float, float]
_RgbI = tuple[int, int, int]

_ThreeInts: TypeAlias = tuple[int, int, int]
_FourInts: TypeAlias = tuple[int, int, int, int]
_FPArray: TypeAlias = npt.NDArray[np.floating[Any]]

_ColorArray = (
    Annotated[npt.NDArray[np.uint8], "(..., 3) | (..., 4)"] | Sequence[Sequence[int]]
)


def get_squared_error(color_a: _RgbF, color_b: _RgbF) -> float:
    """Get squared distance between two colors.

    :param color_a: rgb tuple
    :param color_b: rgb tuple
    :return: squared distance from color_a to color_b
    """
    squared_error: np.floating[Any] = np.sum(np.subtract(color_a, color_b) ** 2)
    return float(squared_error)


@dataclass(frozen=True)
class Member:
    """A member of a cluster.

    :param rgb: rgb tuple
    :param w: combined weight of all occurrences of this rgb value in an image

    When clustering initial images arrays, the weight will only represent the number
    of times the color appears in the image. After removing some color or adding an
    alpha channel, the weight will also reflect the alpha channel, with transparent
    colors weighing less.
    """

    rgbw: Annotated[_FPArray, 4]

    @property
    def rgb(self) -> _RgbF:
        r, g, b = self.rgbw[:3]
        return r, g, b

    @property
    def w(self) -> float:
        return self.rgbw[3]
    
    def __hash__(self) -> int:
        return id(self)

    # @staticmethod
    # def _color_to_rgbw(color: Iterable[float]) -> _FourInts:
        # """Add a weight of 255 to a 3-value vector. Infer weight for 4-value vector.

        # :param color: rgb or rgba tuple
        # :return: rwbw tuple
        # :raises TypeError: if color cannot be cast to int without loss of precision
        # :raises ValueError: if color is not in [0..255]
        # :raises ValueError: if color is not a 3- or 4-tuple

        # For an rgb tuple, the weight is 255.
        # For an rgba tuple, the alpha value float subtracted from 255 to get the weight.

            # >>> Member._color_to_rgbw((1, 2, 3))
            # (1, 2, 3, 255)

            # >>> Member._color_to_rgbw((1, 2, 3, 4))
            # (1, 2, 3, 4)
        # """
        # color = tuple(color)
        # if not all(0 <= x <= 255 for x in color[:3]):
            # raise ValueError("color values must be in [0..255]")
        # if any(x % 1 for x in color):
            # raise TypeError("color values must castable to int without loss")
        # if len(color) == 3:
            # r, g, b = (int(x) for x in color)
            # return (r, g, b, 255)
        # elif len(color) != 4:
            # raise ValueError(f"color must be 3 or 4 values, not {len(color)}")

    @classmethod
    def new_members(cls, colors: _ColorArray) -> set[Member]:
        # TODO: check on definition of _ColorArray
        """Transform an array of rgb or rgbw colors into a set of _Member instances.

        :param colors: list of colors
        :return: set of Member instances

        Silently drop colors without weight. It is possible to return an empty set if
        no colors have weight > 0.
        """
        # TODO: only accept stacked colors
        return {Member(color) for color in colors if color[3] > 0}
        # TODO: delete commented-out code in Member.new_members
        # if isinstance(colors, np.ndarray):
            # colors = colors.reshape(-1, colors.shape[-1])
        # rgbws = [cls._color_to_rgbw(color) for color in colors]

        # if not all(x[3] >= 0 for x in rgbws):
            # raise ValueError("color weights must be non-negative")

        # rgbws = [x for x in rgbws if x[3] > 0]

        # color2weight: DefaultDict[_RgbF, float] = defaultdict(float)
        # for r, g, b, w in rgbws:
            # color2weight[(r, g, b)] += w
        # return {cls(np.array(k + (v,))) for k, v in color2weight.items()}


class Cluster:
    """A cluster of _Member instances.

    :param members: _Member instances

    Hold members in a set. It is important for convergence that the exemplar is not
    updated each time a member is added or removed. Add members from other clusters to
    queue_add and self members to queue_sub. Do not update the members or
    process_queue until each clusters' members have be offered to all other clusters.

    When all clusters that should be moved have been inserted into queues, call
    process_queue and update the exemplar for the next round.

    This is almost a frozen class, but the queue_add and queue_sub attributes are
    mutable.
    """

    def __init__(
        self,
        members: Iterable[Member],
        exemplar_age: int = 0,
    ) -> None:
        assert members
        self.members = set(members)
        self.exemplar_age = exemplar_age

        self.queue_add: set[Member] = set()
        self.queue_sub: set[Member] = set()

    @functools.cached_property
    def weight(self) -> float:
        """Get total weight of members.

        :return: total weight of members
        """
        return sum(member.w for member in self.members)
    
    @functools.cached_property
    def quick_error(self) -> float:
        """Product of max dimension and weight as a rough cost metric.

        :return: product of max dimension and weight
        """
        if len(self.members) == 1:
            return 0.0
        rgbs = [member.rgb for member in self.members]
        max_dim = max(np.ptp(rgbs, axis=0))
        return max_dim * self.weight

    @functools.cached_property
    def exemplar(self) -> _RgbF:
        """Get cluster exemplar.

        :return: the weighted average of all members.

        The exemplar property is lazy, so reset it by setting self._exemplar to None.
        """
        members_array = np.array([member.rgbw for member in self.members])
        colors, weights = members_array[:, :3], members_array[:, 3:]
        return tuple(get_stacked_medians(colors, weights))

    @functools.cached_property
    def as_member(self) -> Member:
        """Get cluster as a Member instance.

        :return: Member instance with rgb and weight of exemplar
        """
        return Member(np.array(self.exemplar + (self.weight,)))

    @functools.cached_property
    def _axis_of_highest_variance(self) -> npt.NDArray[np.floating[Any]]:
        """Get the first Eigenvector of the covariance matrix.

        Under a lot of condions, this will be the axis of highest variance. There are
        things that will break this, like all colors lying on the (1, 1, 1) line, but
        you could use ALMOST any axis to split that cluster anyway, so it isn't worth
        worrying about. There may be other cases where this breaks (silently makes a
        suboptimal split) but I haven't found them yet.
        """
        members_array = np.array([member.rgbw for member in self.members])
        colors, weights = members_array[:, :3], members_array[:, 3]
        covariance_matrix: _FPArray = np_cov(colors.T, aweights=weights)
        return np.linalg.eig(covariance_matrix)[1][0]

    def split(self) -> set[Cluster]:
        """Split cluster into two clusters.

        :return: two new clusters

        Split the cluster into two clusters by the plane perpendicular to the axis of
        highest variance.

        The splitting is a bit funny due to particulars of the stacked median. See
        stacked_quantile module for details.
        """
        if len(self.members) == 1:
            raise ValueError("Cannot split a cluster with only one member")
        if len(self.members) == 2:
            a, b = self.members
            return {Cluster([a]), Cluster([b])}
        abc = self._axis_of_highest_variance

        def get_rel_dist(rgb: _RgbF) -> float:
            """Get relative distance of rgb from plane Ax + By + Cz + 0."""
            return float(np.dot(abc, rgb))  # type: ignore

        scored = [(get_rel_dist(member.rgb), member) for member in self.members]
        median_score = get_stacked_median(
            [s for s, _ in scored], [m.w for _, m in scored]
        )
        left = {m for s, m in scored if s < median_score}
        right = {m for s, m in scored if s > median_score}
        left_weight = sum(m.w for m in left)
        right_weight = sum(m.w for m in right)
        if right_weight < left_weight:
            right |= {m for s, m in scored if s == median_score}
        else:
            left |= {m for s, m in scored if s == median_score}
        return {Cluster(left), Cluster(right)}

    @functools.cache
    def get_squared_error(self, color: _RgbF) -> float:
        """Get the cost of adding a member to this cluster.

        :param member: _Member instance
        :return: cost of adding member to this cluster
        """
        # TODO: try sorting arguments to borrow from other's cache
        return get_squared_error(color, self.exemplar)

    def get_half_squared_error(self, color: _RgbF) -> float:
        """Get the cost of adding a member to this cluster.

        :param member: _Member instance
        :return: cost of adding member to this cluster
        # TODO: factor out get_half_squared_error
        """
        return self.get_squared_error(color) / 4

    @functools.cached_property
    def sum_squared_error(self) -> float:
        """Get the sum of squared errors of all members.

        :return: sum of squared errors of all members
        """
        return sum(
            self.get_squared_error(member.rgb) * member.w for member in self.members
        )

    def process_queue(self) -> Cluster:
        """Process the add and sub queues and update exemplars."""
        if self.queue_add or self.queue_sub:
            return Cluster(self.members - self.queue_sub | self.queue_add)
        else:
            self.exemplar_age += 1
            return self
