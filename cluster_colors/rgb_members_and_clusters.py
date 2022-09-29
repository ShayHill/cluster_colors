from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, DefaultDict, Iterable, Optional, Annotated, TypeAlias


import numpy as np
from numpy import typing as npt

from cluster_colors.stacked_quantile import get_stacked_medians

_RgbF = tuple[float, float, float]
_RgbI = tuple[int, int, int]

_ThreeInts: TypeAlias = tuple[int, int, int]
_FourInts: TypeAlias = tuple[int, int, int, int]

_ColorArray = (
    Annotated[npt.NDArray[np.uint8], "(..., 3) | (..., 4)"] | Iterable[Iterable[int]]
)

def _get_squared_error(color_a: _RgbF, color_b: _RgbF) -> float:
    """Get squared distance between two colors.

    :param color_a: rgb tuple
    :param color_b: rgb tuple
    :return: squared distance from color_a to color_b
    """
    return np.sum((np.array(color_a) - np.array(color_b)) ** 2)

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

    rgb: _RgbI
    w: float

    @property
    def rgbt(self) -> tuple[float, float, float, float]:
        # TODO: check for usages
        """Get rgba tuple."""
        return self.rgb + (255 - self.w,)

    @property
    def rgbw(self) -> tuple[float, float, float, float]:
        """Get rgbw tuple."""
        return self.rgb + (self.w,)

    @property
    def as_2bit(self) -> _RgbI:
        """Get 2-bit rgb tuple.

        :return: 2-bit rgb tuple

        This is used for the initial clustering.
        """
        r, g, b = self.rgb
        return (r // 64, g // 64, b // 64)

    @staticmethod
    def _color_to_rgbw(color: Iterable[float]) -> _FourInts:
        """Add a weight of 255 to a 3-value vector. Infer weight for 4-value vector.

        :param color: rgb or rgba tuple
        :return: rwbw tuple
        :raises TypeError: if color cannot be cast to int without loss of precision
        :raises ValueError: if color is not in [0..255]
        :raises ValueError: if color is not a 3- or 4-tuple

        For an rgb tuple, the weight is 255.
        For an rgba tuple, the alpha value float subtracted from 255 to get the weight.

            >>> Member._color_to_rgbw((1, 2, 3))
            (1, 2, 3, 255)

            >>> Member._color_to_rgbw((1, 2, 3, 4))
            (1, 2, 3, 251)
        """
        # TODO: this looks all wrong now. Do not assume alpha channel is transparency
        color = tuple(color)
        if not all(0 <= x <= 255 for x in color[:3]):
            raise ValueError("color values must be in [0..255]")
        if any(x % 1 for x in color):
            raise TypeError("color values must castable to int without loss")
        if len(color) == 3:
            r, g, b = (int(x) for x in color)
            return (r, g, b, 255)
        elif len(color) == 4:
            r, g, b, a = (int(x) for x in color)
            return (r, g, b, 255 - a)
        raise ValueError(f"color must be 3 or 4 values, not {len(color)}")

    @classmethod
    def new_members(cls, colors: _ColorArray) -> set["Member"]:
        """Transform an array of rgb or rgba colors into a set of _Member instances.
        #TODO: fix docstring for new_members

        :param colors: list of colors
        :return: list of weighted colors
        """
        if isinstance(colors, np.ndarray):
            colors = colors.reshape(-1, colors.shape[-1])
        rgbws = [cls._color_to_rgbw(color) for color in colors]

        color2weight: DefaultDict[_RgbI, int] = defaultdict(int)
        for r, g, b, w in rgbws:
            color2weight[(r, g, b)] += w
        return {cls(k, v) for k, v in color2weight.items()}

    @property
    def weighted(self) -> _RgbF:
        """Get weighted rgb tuple.

        :return: weighted rgb tuple

        For creating weighted averages or comparing cluster weights.
        """
        r, g, b = self.rgb
        return (r * self.w, g * self.w, b * self.w)


class Cluster:
    """A cluster of _Member instances.

    :param members: _Member instances

    Hold members in a set. It is important for convergence that the exemplar is not
    updated each time a member is added or removed. Add members from other clusters to
    queue_add and self members to queue_sub. Do not update the members or
    process_queue until each clusters' members have be offered to all other clusters.

    When all clusters that should be moved have been inserted into queues, call
    process_queue and update the exemplar for the next round.
    """

    def __init__(self, members: Iterable[Member]) -> None:
        self.members = set(members)
        self._exemplar: Optional[_RgbF] = None
        self.exemplar_age = 0
        self.exemplar_cost_func = lru_cache(maxsize=None)(_get_squared_error)
        self.queue_add: set[Member] = set()
        self.queue_sub: set[Member] = set()

    @property
    def weight(self) -> float:
        """Get total weight of members.

        :return: total weight of members
        """
        return sum(member.w for member in self.members)

    @property
    def exemplar(self) -> _RgbF:
        """Get cluster exemplar.

        :return: the weighted average of all members.

        The exemplar property is lazy, so reset it by setting self._exemplar to None.
        """
        if self._exemplar is None:
            members_array = np.array([member.rgbw for member in self.members])
            colors, weights = members_array[:, :3], members_array[:, 3:]
            self._exemplar = tuple(get_stacked_medians(colors, weights))
            self.exemplar_age = 0
        return self._exemplar

    @property
    def _axis_of_highest_variance(self) -> npt.NDArray[np.floating[Any]]:
        """Get the first Eigenvector of the covariance matrix.

        Under a lot of condions, this will be the axis of highest variance. There are
        things that will break this, like all colors lying on the (1, 1, 1) line, but
        you could use almost ANY axis to split that cluster anyway, so it isn't worth
        worring about. There may be other cases where this breaks (silenly makes a
        sub-optimal split) but I haven't found them yet.
        """
        members_array = np.array([member.rgbw for member in self.members])
        covariance_matrix = np.cov(members_array[:, :3].T, aweights=members_array[:, 3])
        return np.linalg.eig(_covariance_matrix)[1][0]

    @staticmethod
    def _filter_out_weightless_clusters(self, *cluster: Cluster) -> set[Cluster]:
        """Filter out clusters with no members.

        :param cluster: clusters to filter
        :return: clusters with members
        """
        return {c for c in cluster if c.weight > 0}

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
            return self._filter_out_weightless_clusters(Cluster([a]), Cluster([b]))
        abc = self._axis_of_highest_variance

        def get_rel_dist(rgb) -> float:
            """Get relative distance of rgbw from plane abc."""
            return np.dot(abc, rgb)

        scored = [(get_rel_dist(member.rgb), member) for member in self.members]
        median_score = stacked_median([s for s, _ in scored], [m.w for _, m in scored])
        left = {m for s, m in scored if s < median_score}
        right = {m for s, m in scored if s > median_score}
        left_weight = sum(m.w for m in left)
        right_weight = sum(m.w for m in right)
        if right_weight < left_weight:
            right |= {m for s, m in scored if s == median_score}
        else:
            left |= {m for s, m in scored if s == median_score}
        return self._filter_out_weightless_clusters(Cluster([left]), Cluster([right]))

    @functools.cache
    def get_squared_error(self, color: _RgbF) -> float:
        """Get the cost of adding a member to this cluster.

        :param member: _Member instance
        :return: cost of adding member to this cluster
        """
        # TODO: try sorting arguments to borrow from other's cache
        return _get_squared_error(color, self.exemplar)

    @functools.cache
    def get_half_squared_error(self, color: _RgbF) -> float:
        """Get the cost of adding a member to this cluster.

        :param member: _Member instance
        :return: cost of adding member to this cluster
        """
        error = pow(self.get_squared_error(color), 0.5)
        return pow(error / 2, 2)

    @property
    def sum_squared_error(self) -> float:
        """Get the sum of squared errors of all members.

        :return: sum of squared errors of all members
        """
        return sum(self.get_squared_error(member.rgb) * member.w for member in self.members)

    def process_queue(self) -> None:
        """Process the add and sub queues and update exemplars."""
        if self.queue_add or self.queue_sub:
            self.members |= self.queue_add
            self.members -= self.queue_sub
            self._exemplar = None
            self.get_squared_error.cache_clear()
            self.get_half_squared_error.cache_clear()
            self.queue_add.clear()
            self.queue_sub.clear()
        else:
            self.exemplar_age += 1
