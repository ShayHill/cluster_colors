from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Annotated, Any, Iterable, Iterator, TypeAlias, cast, Sequence, \
    Optional

import numpy as np
from numpy import cov as np_cov  # type: ignore
from numpy import typing as npt

from cluster_colors.distance_matrix import DistanceMatrix
from cluster_colors.stacked_quantile import get_stacked_median, get_stacked_medians
from cluster_colors.type_hints import FPArray

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

    return: squared distance from color_a to color_b
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

    as_array: Annotated[_FPArray, 4]

    @property
    def vs(self) -> _RgbF:
        return tuple(self.as_array[:-1])

    @property
    def w(self) -> float:
        return self.as_array[-1]

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    def new_members(cls, stacked_vectors: FPArray) -> set[Member]:
        # TODO: check on definition of _ColorArray
        """Transform an array of rgb or rgbw colors into a set of _Member instances.

        :param colors: list of colors
        :return: set of Member instances

        Silently drop colors without weight. It is possible to return an empty set if
        no colors have weight > 0.
        """
        return {Member(v) for v in stacked_vectors if v[-1] > 0}


class Cluster:
    """A cluster of _Member instances.

    :param members: _Member instances

    Hold members in a set. It is important for convergence that the exemplar is not
    updated each time a member is added or removed. Add members from other clusters to
    queue_add and self members to queue_sub. Do not update the members or
    process_queue until each clusters' members have be offered to all other clusters.

    When all clusters that should be moved have been inserted into queues, call
    process_queue and, if changes have occurred, create a new exemplar for the next
    round.

    This is almost a frozen class, but the queue_add and queue_sub attributes are
    mutable.
    """

    def __init__(self, members: Iterable[Member]) -> None:
        assert members, "cannot create an empty cluster"
        self.members = set(members)
        self.exemplar_age = 0
        self.queue_add: set[Member] = set()
        self.queue_sub: set[Member] = set()

    @functools.cached_property
    def as_array(self) -> _FPArray:
        """Cluster as an array of member arrays."""
        return np.array([member.as_array for member in self.members])

    @functools.cached_property
    def as_member(self) -> Member:
        """Get cluster as a Member instance.

        :return: Member instance with rgb and weight of exemplar
        """
        vss, ws = np.split(self.as_array, [-1], axis=1)
        vs = get_stacked_medians(vss, ws)
        return Member(np.append(vs, sum(ws)))

    @property
    def w(self) -> float:
        """Total weight of members.
        """
        return self.as_member.w

    @property
    def vs(self) -> tuple[float, ...]:
        """Values for cluster as a member instance"""
        return self.as_member.vs

    @property
    def exemplar(self) -> tuple[float, ...]:
        """Get cluster exemplar.

        :return: the weighted average of all members.

        If I strictly followed my own conventions, I'd just call this property `vs`,
        but this value acts as the exemplar when clustering, so I prefer to use this
        alias in my clustering code.
        """
        return self.vs

    @functools.cached_property
    def _np_linalg_eig(self) -> tuple[FPArray, FPArray]:
        """Cache the value of np.linalf.eig on the covariance matrix of the cluster."""
        vss, ws = np.split(self.as_array, [-1], axis=1)
        covariance_matrix: FPArray = np_cov(vss.T, aweights=ws.flatten())
        return np.linalg.eig(covariance_matrix)

    @functools.cached_property
    def _variance(self) -> float:
        """Get the variance of the cluster.

        :return: variance of the cluster
        """
        return max(self._np_linalg_eig[0])

    @functools.cached_property
    def _axis_of_highest_variance(self) -> FPArray:
        """Get the first Eigenvector of the covariance matrix.

        Under a lot of conditions, this will be the axis of highest variance. There are
        things that will break this, like all colors lying on the (1, 1, 1) line, but
        you could use ALMOST any axis to split that cluster anyway, so it isn't worth
        worrying about. There may be other cases where this breaks (silently makes a
        suboptimal split) but I haven't found them yet.
        """
        return self._np_linalg_eig[1][np.argmax(self._np_linalg_eig[0])]

    @functools.cached_property
    def quick_error(self) -> float:
        """Product of variance and weight as a rough cost metric.

        :return: product of max dimension and weight

        This is the errir used to determine if a cluster should be split in the
        cutting pre-clustering step. For that purpose, it is superior to sum squared
        error, because you *want* to isolate outliers in the cutting step.
        """
        if len(self.members) == 1:
            return 0.0
        return self._variance * self.w

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

        scored = [(get_rel_dist(member.vs), member) for member in self.members]
        median_score = get_stacked_median(
            [s for s, _ in scored], [m.w for _, m in scored]
        )
        left = {m for s, m in scored if s < median_score}
        right = {m for s, m in scored if s > median_score}
        center = {m for s, m in scored if s == median_score}
        if center and sum(m.w for m in left) < sum(m.w for m in right):
            left |= center
        else:
            right |= center
        return {Cluster(left), Cluster(right)}

    @functools.cache
    def se(self, member: Member) -> float:
        """Get the cost of adding a member to this cluster.

        :param member: _Member instance
        :return: cost of adding member to this cluster
        """
        return get_squared_error(member.vs, self.exemplar)

    @functools.cached_property
    def sse(self) -> float:
        """Get the sum of squared errors of all members.

        :return: sum of squared errors of all members
        """
        return sum(self.se(member) * member.w for member in self.members)

    def process_queue(self) -> Cluster:
        """Process the add and sub queues and update exemplars."""
        if self.queue_add or self.queue_sub:
            self.exemplar_age = 0
            return Cluster(self.members - self.queue_sub | self.queue_add)
        else:
            self.exemplar_age += 1
            return self


def _get_cluster_squared_error(cluster_a: Cluster, cluster_b: Cluster) -> float:
    """Get squared distance between two clusters.

    :param cluster_a: Cluster
    :param cluster_b: Cluster
    :return: squared distance from cluster_a.exemplar to cluster_b.exemplar
    """
    return get_squared_error(cluster_a.exemplar, cluster_b.exemplar)


class Clusters:
    """A set of Cluster instances with cached distances and queued updates.

    Maintains a cached matrix of squared distances between all Cluster exemplars.
    Created for K-medians which passes members around *before* updating exemplars, so
    any changes identified must be staged in each Cluster's queue_add and queue_sub
    sets then applied with _Clusters.process_queues.
    """

    def __init__(self, clusters: Iterable[Cluster]):
        self._clusters: set[Cluster] = set()
        self.spans: DistanceMatrix[Cluster]
        self.spans = DistanceMatrix(_get_cluster_squared_error)
        self.add(*clusters)



    def __iter__(self) -> Iterator[Cluster]:
        return iter(self._clusters)

    def __len__(self) -> int:
        return len(self._clusters)

    def add(self, *cluster_args: Cluster) -> None:
        for cluster in cluster_args:
            self._clusters.add(cluster)
            self.spans.add(cluster)

    def remove(self, *cluster_args: Cluster) -> None:
        for cluster in cluster_args:
            self._clusters.remove(cluster)
            self.spans.remove(cluster)

    def process_queues(self) -> None:
        processed = {c.process_queue() for c in self._clusters}
        self.remove(*(self._clusters - processed))
        self.add(*(processed - self._clusters))
