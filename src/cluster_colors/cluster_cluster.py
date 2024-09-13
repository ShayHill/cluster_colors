"""Type for a single cluster.

:author: Shay Hill
:created: 2024-09-03
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Annotated

import numpy as np
from stacked_quantile import get_stacked_medians

from cluster_colors.cluster_member import Members

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cluster_colors.type_hints import FPArray, ProximityMatrix, Vector, Vectors


_sn_gen = itertools.count()


def _construct_is_low(low: float, high: float) -> Callable[[float], bool]:
    """Return a function that determines if a value is closer to below than above.

    :param low: exemplar of low values
    :param hight: exemplar of high values
    :return: function that determines if a value is closer to low than high
    """

    def is_low(value: float) -> bool:
        """Determine if a value is closer to low than high.

        :param value: float
        :return: True if value is closer to low than high
        """
        return abs(value - low) < abs(value - high)

    return is_low


def _split_floats(floats: Iterable[float]) -> int:
    """Find an index of a sorted list of floats that minimizes the sum of errors.

    :param floats: An iterable of float numbers.
    :return: An index of the list that minimizes the sum of errors.
    """
    floats = sorted(floats)
    if len(floats) < 2:
        msg = "Cannot split a list of floats with fewer than 2 elements"
        raise ValueError(msg)

    def converge(splitter: int) -> int:
        if splitter == 0:  # all floats are identical
            return 0
        below, above = floats[:splitter], floats[splitter:]
        is_low = _construct_is_low(sum(below) / len(below), sum(above) / len(above))
        new_splitter = len(list(itertools.takewhile(is_low, floats)))
        if new_splitter == splitter:
            return splitter
        return converge(new_splitter)

    return converge(len(floats) // 2)


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

    def __init__(self, members: Members, ixs: Iterable[int] | None = None) -> None:
        """Identify a cluster by the indices of its members.

        :param members: Members instance
        :param ixs: optional indices of members. If None, use all members.
        """
        self.members = members
        if ixs is None:
            self.ixs = np.arange(len(self.members), dtype=np.int32)
        else:
            self.ixs = np.array(sorted(ixs), dtype=np.int32)
        self.sn = next(_sn_gen)

    # ===========================================================================
    #   constructors
    # ===========================================================================

    @classmethod
    def from_vectors(
        cls, vectors: Vectors, pmatrix: ProximityMatrix | None = None
    ) -> Cluster:
        """Create a Cluster instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: A Cluster instance with members
            {Member([r0, g0, b0, w0]), Member([r1, g1, b1, w1]), ...}
        """
        members = Members.from_vectors(vectors, pmatrix=pmatrix)
        return cls(members)

    @classmethod
    def from_stacked_vectors(
        cls, stacked_vectors: Vectors, pmatrix: ProximityMatrix | None = None
    ) -> Cluster:
        """Create a Cluster instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :return: A Cluster instance with members
            {Member([r0, g0, b0, w0]), Member([r1, g1, b1, w1]), ...}
        """
        members = Members.from_stacked_vectors(stacked_vectors, pmatrix=pmatrix)
        return cls(members)

    # ===========================================================================
    #   Vector-like properties
    # ===========================================================================

    @property
    def weight(self) -> float:
        """Total weight of members.

        :return: total weight of members
        """
        return sum(self.members.weights[self.ixs])

    # ===========================================================================
    #   Cluster centers
    # ===========================================================================

    def _get_weighted_medoid(self, ixs: Iterable[int] | None = None) -> int:
        """Get the index of the mediod, respecting weights.

        :param ixs: optional subset of members indices. I can't see a use case for
            manually passing this, but it's here to break ties in property medoid.
        :return: index of the mediod, respecting weights
        """
        ixs_ = self.ixs if ixs is None else np.array(list(ixs), dtype=np.int32)
        if len(ixs_) == 1:
            return int(ixs_[0])
        return int(ixs_[np.argmin(self.members.weighted_pmatrix[ixs_].sum(axis=1))])

    @functools.cached_property
    def weighted_medoid(self) -> int:
        """Get cluster exemplar.

        :return: the index of the exemplar with the least cost

        If I strictly followed my own conventions, I'd just call this property `vs`,
        but this value acts as the exemplar when clustering, so I prefer to use this
        alias in my clustering code.
        """
        return self._get_weighted_medoid()

    @functools.cached_property
    def unweighted_medoid(self) -> int:
        """Get the index of the mediod, mostly ignoring weights.

        :return: index of the mediod

        If multiple members are tied for cost, use weights to break the tie. This
        will always be the case with two members, but it is theoretically possible
        with more members. That won't happen, but it's cheap to cover the case.
        """
        if len(self.ixs) < 3:
            return self._get_weighted_medoid()

        row_sums = self.members.pmatrix[self.ixs].sum(axis=1)
        min_cost = np.min(row_sums)
        arg_where_min = np.argwhere(row_sums == min_cost).flatten()
        arg_where_min = [self.ixs[x] for x in arg_where_min]

        if len(arg_where_min) == 1:
            return int(arg_where_min[0])
        return self._get_weighted_medoid(map(int, arg_where_min))

    @functools.cached_property
    def weighted_median(self) -> Vector:
        """Get the median of the cluster, respecting weights.

        :return: median of the cluster

        This is categorically different than the medoid, because the median is not a
        member of the cluster. So this property is not an index to the cluster
        members, but a vector is likely not coincident with any member.
        """
        weights = self.members.weights[self.ixs].reshape(-1, 1)
        return get_stacked_medians(self.members.vectors[self.ixs], weights)

    @property
    def as_vector(self) -> Vector:
        """Get the exemplar as a vector.

        :return: exemplar as a vector
        """
        return self.members.vectors[self.weighted_medoid]

    @property
    def as_stacked_vector(self) -> Vector:
        """Get the exemplar as a stacked vector.

        :return: exemplar as a stacked vector
        """
        weight = self.members.weights[self.weighted_medoid]
        return np.append(self.as_vector, weight)

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
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)
        return np.real(eigenvalues), np.real(eigenvectors)

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
        if len(self.ixs) == 1:
            return 0
        return float(np.sum(self.members.weighted_pmatrix[self.weighted_medoid]))

    @property
    def error_metric(self) -> tuple[float, int]:
        """Break ties in the error property.

        :return: the error and negative sn so older cluster will split in case of a
            tie.

        Ties aren't likely, but just to keep everything deterministic.
        """
        return self.error, -self.sn

    def get_merge_error(self, other: Cluster) -> float:
        """Get the complete linkage error of merging this cluster with another.

        :return: sum of squared errors of all members if merged with another cluster
        """
        return float(np.max(self.members.pmatrix[np.ix_(self.ixs, other.ixs)]))

    def get_merge_error_metric(self, other: Cluster) -> tuple[float, int]:
        """Break ties in the get_merge_error property.

        :return: the error and negative of max sn so older cluster will merge in
            case of a tie.

        Ties aren't likely, but just to keep everything deterministic.
        """
        return self.get_merge_error(other), -max(self.sn, other.sn)

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
        vecs = self.members.vectors

        def rel_dist(x: int) -> float:
            return np.dot(abc, self.members.vectors[x])

        scored = sorted([(rel_dist(x), tuple(vecs[x]), x) for x in self.ixs])
        split = _split_floats([s for s, *_ in scored])
        if split in {0, len(scored)}:
            split = len(scored) // 2
        return {
            Cluster(self.members, [x for *_, x in scored[:split]]),
            Cluster(self.members, [x for *_, x in scored[split:]]),
        }
