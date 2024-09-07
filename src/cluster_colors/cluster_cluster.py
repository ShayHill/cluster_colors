"""Type for a single cluster.

:author: Shay Hill
:created: 2024-09-03
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, TypeVar

import numpy as np
from basic_colormath import get_delta_e_lab, get_sqeuclidean, rgb_to_lab
from paragraphs import par
from stacked_quantile import get_stacked_median, get_stacked_medians

from cluster_colors.cluster_member import Member, Members, split_members_by_plane
from cluster_colors.distance_matrix import DistanceMatrix
import itertools

_RGB = tuple[float, float, float]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from cluster_colors.type_hints import FPArray, StackedVectors, Vector, VectorLike


_sn_gen = itertools.count()

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


def _get_squared_error(vector_a: Vector, vector_b: Vector) -> np.float64:
    """Get squared distance between two vectors.

    :param vector_a: (-1,) vector
    :param vector_b: (-1,) vector
    :return: squared Euclidian distance from vector_a to vector_b
    """
    return np.sum((vector_a - vector_b) ** 2)


class Cluster:
    """A cluster of Member instances.

    :param members: Member instances

    Hold Members in a set. It is important for convergence that the exemplar is not
    updated each time a member is added or removed. Add members from other clusters
    to queue_add and self members to queue_sub. Do not update the members or
    process_queue until each cluster's members have be offered to all other clusters.

    When all clusters that should be moved have been inserted into queues, call
    process_queue and, if changes have occurred, create a new Cluster instance for
    the next round.

    This is almost a frozen class, but the queue_add, queue_sub, and exemplar_age
    attributes are intended to be mutable.
    """

    # def __init__(self, members: Iterable[Member]) -> None:
    #     """Initialize a Cluster instance.

    #     :param members: Member instances
    #     :raise ValueError: if members is empty
    #     """
    #     if not members:
    #         msg = "Cannot create an empty cluster"
    #         raise ValueError(msg)
    #     if not any(m.w for m in members):
    #         msg = "Cannot create a cluster with only 0-weight members"
    #         raise ValueError(msg)
    #     self.members = set(members)

    #     # State members to cache changes for convergence and avoid redundant
    #     # comparisons.
    #     self.is_new: bool = True
    #     self.queue_add: set[Member] = set()
    #     self.queue_sub: set[Member] = set()

    #     as_array = np.array([m.as_array for m in self.members])
    #     self._vss, self._ws = np.split(as_array, [-1], axis=1)
    #     self._children: Annotated[set[Cluster], "doubleton"] | None = None

    def __init__(self, members: Members, ixs: Iterable[int] | None = None) -> None:
        """Identify a cluster by the indices of its members.

        :param members: Members instance
        :param ixs: optional indices of members. If None, use all members.
        """
        self.members = members
        if ixs is None:
            self.ixs = np.arange(len(self.members), dtype=np.int32)
        else:
            self.ixs = np.array(list(ixs), dtype=np.int32)
        self._sn = next(_sn_gen)

    def __len__(self) -> int:
        """Get the number of members in the cluster.

        :return: number of members in the cluster
        """
        return len(self.ixs)

    @classmethod
    def from_stacked_vectors(cls, stacked_vectors: StackedVectors) -> Cluster:
        """Create a Cluster instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :return: A Cluster instance with members
            {Member([r0, g0, b0, w0]), Member([r1, g1, b1, w1]), ...}
        """
        members = Members.from_stacked_vectors(stacked_vectors)
        return cls(members)

    def __iter__(self) -> Iterator[Member]:
        """Iterate over members.

        :return: None
        :yield: Members
        """
        return iter(self.members)

    @functools.cached_property
    def as_member(self) -> Member:
        """Get cluster as a Member instance.

        :return: Member instance with median rgb and sum weight of cluster members
        """
        medians = get_stacked_medians(self._vss, self._ws)
        return Member(np.array([*medians, np.sum(self._ws)]))

    # ===========================================================================
    #   Vector-like properties
    # ===========================================================================

    @property
    def vs(self) -> FPArray:
        """Values for cluster as a member instance.

        :return: tuple of values (r, g, b) from self.as_member((r, g, b, w))
        """
        return self.as_member.vs

    @property
    def w(self) -> float:
        """Total weight of members.

        :return: total weight of members
        """
        return self.as_member.w

    def _get_medoid_respecting_weights(self, ixs: Iterable[int] | None = None) -> int:
        """Get the index of the mediod, respecting weights.

        :param ixs: optional subset of members indices. I can't see a use case for
            manually passing this, but it's here to break ties in property medoid.
        :return: index of the mediod, respecting weights
        """
        ixs = self.ixs if ixs is None else np.array(list(ixs), dtype=np.int32)
        if len(ixs) == 1:
            return int(ixs[0])
        return int(ixs[np.argmin(self.members.weighted_pmatrix[ixs].sum(axis=1))])

    @functools.cached_property
    def exemplar(self) -> int:
        """Get cluster exemplar.

        :return: the index of the exemplar with the least cost

        If I strictly followed my own conventions, I'd just call this property `vs`,
        but this value acts as the exemplar when clustering, so I prefer to use this
        alias in my clustering code.
        """
        return self._get_medoid_respecting_weights()

    @property
    def as_vector(self) -> Vector:
        """Get the exemplar as a vector.

        :return: exemplar as a vector
        """
        return self.members.vectors[self.exemplar]

    @property
    def as_stacked_vector(self) -> Vector:
        """Get the exemplar as a stacked vector.

        :return: exemplar as a stacked vector
        """
        weight = self.members.weights[self.exemplar]
        return np.append(self.as_vector, weight)

    @functools.cached_property
    def medoid(self) -> int:
        """Get the index of the mediod, mostly ignoring weights.

        :return: index of the mediod

        If multiple members are tied for cost, use weights to break the tie. This
        will always be the case with two members, but it is theoretically possible
        with more members. That won't happen, but it's cheap to cover the case.
        """
        if len(self.ixs) < 3:
            return self._get_medoid_respecting_weights()

        row_sums = self.members.pmatrix[self.ixs].sum(axis=1)
        min_cost = np.min(row_sums)
        arg_where_min = np.argwhere(row_sums == min_cost).flatten()

        if len(arg_where_min) == 1:
            return int(self.ixs[arg_where_min[0]])
        else:
            return self._get_medoid_respecting_weights(arg_where_min)

    @property
    def rgb_floats(self) -> tuple[float, float, float]:
        """Get the exemplar as a tuple of floats.

        :return: exemplar as a tuple of floats
        """
        return self.as_member.rgb_floats

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Get the exemplar as a tuple of 8-bit integers.

        :return: exemplar as a tuple of 8-bit integers
        """
        return self.as_member.rgb

    @property
    def lab(self) -> tuple[float, float, float]:
        """Get the exemplar as a tuple of CIELAB floats.

        :return: exemplar as a tuple of CIELAB floats
        """
        return self.as_member.lab

    def _covariance_matrix(self) -> FPArray:
        """Get the covariance matrix of the cluster.

        :return: covariance matrix of the cluster
        """
        # vss, ws = self._vss, self._ws
        vss = self.members.vectors[:, :-1]
        ws = self.members.vectors[:, -1:]
        frequencies = np.clip(ws.flatten(), 1, None).astype(int)
        return np.cov(vss.T, fweights=frequencies)

    @property
    def covariance_matrix(self) -> FPArray:
        """Get the covariance matrix of the cluster.

        :return: covariance matrix of the cluster
        """
        vs = self.members.vectors
        ws = np.ceil(self.members.weights)
        return np.cov(vs.T, fweights=ws)

    @functools.cached_property
    def _np_linalg_eig(self) -> tuple[FPArray, FPArray]:
        """Cache the value of np.linalg.eig on the covariance matrix of the cluster.

        :return: tuple of eigenvalues and eigenvectors
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)
            return np.real(eigenvalues), np.real(eigenvectors)
        except:
            breakpoint()

    @property
    def _variance(self) -> float:
        """Get the variance of the cluster.

        :return: variance of the cluster
        """
        return max(self._np_linalg_eig[0])

    @property
    def _direction_of_highest_variance(self) -> FPArray:
        """Get the first Eigenvector of the covariance matrix.

        :return: first Eigenvector of the covariance matrix

        Return the normalized eigenvector with the largest eigenvalue.
        """
        eigenvalues, eigenvectors = self._np_linalg_eig
        return eigenvectors[:, np.argmax(eigenvalues)]

    @functools.cached_property
    def error(self) -> float:
        """Get the sum of proximity errors of all members.

        :return: sum of squared errors of all members
        """
        if len(self) == 1:
            return 0
        return float(np.sum(self.members.weighted_pmatrix[self.exemplar]))

    @property
    def error_metric(self) -> tuple[float, int]:
        """Break ties in the error property.

        :return: the error and negative _sn so older cluster will split in case of a
            tie.

        Ties aren't likely, but just to keep everything deterministic.
        """
        return self.error, -self._sn

    @functools.cached_property
    def quick_error(self) -> float:
        """Product of variance and weight as a rough cost metric.

        :return: product of max dimension and weight

        This is the error used to determine if a cluster should be split in the
        cutting pre-clustering step. For that purpose, it is superior to sum squared
        error, because you *want* to isolate outliers in the cutting step.
        """
        if len(self.members) == 1:
            return 0.0
        return self._variance * self.w

    # TODO: create a test with some 0-weight members
    @functools.cached_property
    def is_splittable(self) -> bool:
        """Can the cluster be split?

        :return: True if the cluster can be split

        If the cluster contains at least two members with non-zero weight, those
        members will end up in separate clusters when split. 0-weight members are
        tracers. A cluster with only tracers is invalid.
        """
        qualifying_members = (x for x in self.members if x.w)
        try:
            _ = next(qualifying_members)
            _ = next(qualifying_members)
        except StopIteration:
            return False
        else:
            return True

    def split(self) -> Annotated[set[Cluster], "doubleton"]:
        """Split cluster into two clusters.

        :return: two new clusters
        :raises ValueError: if cluster has only one member

        Split the cluster into two clusters by the plane perpendicular to the axis of
        highest variance.

        The splitting is a bit funny due to innate characteristice of the stacked
        median. It is possible to get a split with members
            a) on one side of the splitting plane; and
            b) exactly on the splitting plane.
        See stacked_quantile module for details, but that case is covered here.
        """
        abc = self._direction_of_highest_variance

        def rel_dist(x: int) -> float:
            return np.dot(abc, self.members.vectors[x])

        sorted_along_dhv = sorted(self.ixs, key=rel_dist)
        return {
            Cluster(self.members, sorted_along_dhv[: len(self.ixs) // 2]),
            Cluster(self.members, sorted_along_dhv[len(self.ixs) // 2 :]),
        }

    def se(self, member_candidate: Member) -> float:
        """Get the cost of adding a member to this cluster.

        :param member_candidate: Member instance
        :return: cost of adding member to this cluster
        """
        return _get_squared_error(member_candidate.vs, self.exemplar)

    @functools.cached_property
    def sse(self) -> float:
        """Get the sum of squared errors of all members.

        :return: sum of squared errors of all members
        """
        return sum(self.se(member) * member.w for member in self.members)

    def process_queue(self) -> Cluster:
        """Process the add and sub queues and update exemplars.

        :return: self or a new cluster
        """
        if self.queue_add or self.queue_sub:
            new_members = self.members - self.queue_sub | self.queue_add
            # reset state in case we revert back to this cluster with sync()
            self.is_new = True
            self.queue_add.clear()
            self.queue_sub.clear()
            return Cluster(new_members)
        self.is_new = False
        return self
