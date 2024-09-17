"""Type for a single cluster.

Each Cluster instance holds a Members instance and a list of indices to vectors in
that instance. Clusters have multiple errors and centers defined for various
purposes, including two centroids, weighted and unweighted medoid, used within the
clustering algorithm.

:author: Shay Hill
:created: 2024-09-03
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import numpy as np
from paragraphs import par
from stacked_quantile import get_stacked_medians

from cluster_colors.cluster_members import Members

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cluster_colors.type_hints import (
        CenterName,
        CentroidName,
        FPArray,
        ProximityMatrix,
        QualityMetric,
        Vector,
        VectorsLike,
    )


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

    This is used to find a natural split along the axis of highest variance, if such
    exists. During the k-medoids clustering, this little bit of intelligence will
    matter less, because the SuperclusterBase._reassign method will help address poor
    splits, but a few hundredths of a second is worth the effort during the median
    cut (`cut_colors`) process.
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
    """A cluster of indices to self.members.vectors.

    quality_metric: The property used to select which cluster to split. This is
        meaningles for agglomerative clustering.
    quality_centroid: The property used as a centroid when computing the quality
        metric.
    """

    quality_metric: QualityMetric = "sum_error"
    quality_centroid: CentroidName = "weighted_medoid"

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

    # ===========================================================================
    #   parameter-based properties
    # ===========================================================================

    @property
    def error(self) -> float:
        """Get the value of the split metric.

        :return: value of the split metric
        """
        return getattr(self, self.quality_metric)

    @property
    def centroid(self) -> int:
        """Get the value of the error centroid.

        :return: value of the error centroid
        """
        return getattr(self, self.quality_centroid)

    # ===========================================================================
    #   constructors
    # ===========================================================================

    @classmethod
    def from_vectors(
        cls, vectors: VectorsLike, pmatrix: ProximityMatrix | None = None
    ) -> Cluster:
        """Create a Cluster instance from a sequence or array of colors.

        :param vectors: An iterable of vectors
            [(r0, g0, b0), (r1, g1, b1), ...]
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: A Cluster instance with members
            {Member([r0, g0, b0, 1]), Member([r1, g1, b1, 1]), ...}
        """
        members = Members.from_vectors(vectors, pmatrix=pmatrix)
        return cls(members)

    @classmethod
    def from_stacked_vectors(
        cls, stacked_vectors: VectorsLike, pmatrix: ProximityMatrix | None = None
    ) -> Cluster:
        """Create a Cluster instance from an array of colors with a weight axis.

        :param stacked_vectors: An iterable of vectors with a weight axis
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :return: A Cluster instance with members
            {Member([r0, g0, b0, w0]), Member([r1, g1, b1, w1]), ...}
        """
        members = Members.from_stacked_vectors(stacked_vectors, pmatrix=pmatrix)
        return cls(members)

    # ===========================================================================
    #   vector-like properties
    # ===========================================================================

    @property
    def weight(self) -> float:
        """Total weight of members.

        :return: total weight of members
        """
        return sum(self.members.weights[self.ixs])

    @property
    def error_weight(self) -> float:
        """Total weight of members that are not the center.

        :return: total weight of members that are not the center

        This is for use in various metrics one might use for selecting clusters to
        split or join.
        """
        if len(self.ixs) == 1:
            return 0
        return self.weight - self.members.weights[self.centroid]

    def get_as_vector(self, which_center: CenterName | None = None) -> Vector:
        """Get the exemplar as a vector.

        :param which_center: optionally specify a cluster center attribute. Choices
            are 'weighted_median', 'weighted_medoid', or 'unweighted_medoid'. Default
            is 'weighted_median'.
        :return: cluster center as a vector
        """
        which_center = which_center or "weighted_median"
        if which_center == "weighted_median":
            return self.weighted_median
        if which_center == "weighted_medoid":
            return self.members.vectors[self.weighted_medoid]
        if which_center == "unweighted_medoid":
            return self.members.vectors[self.unweighted_medoid]
        msg = par(
            f"""center_name must be 'weighted_median', 'weighted_medoid', or
            'unweighted_medoid', but got {which_center}"""
        )
        raise ValueError(msg)

    def get_as_stacked_vector(self, which_center: CenterName | None = None) -> Vector:
        """Get the exemplar as a stacked vector.

        :return: [*center, weight] as a stacked vector
        :param which_center: optionally specify a cluster center attribute. Choices
            are 'weighted_median', 'weighted_medoid', or 'unweighted_medoid'. Default
            is 'weighted_median'.
        """
        vector = self.get_as_vector(which_center)
        weight = sum(self.members.weights[self.ixs])
        return np.append(vector, weight)

    # ===========================================================================
    #   cluster centers
    # ===========================================================================

    def _get_weighted_medoid(self, ixs: Iterable[int] | None = None) -> int:
        """Get the index of the mediod, respecting weights.

        :param ixs: optional subset of members indices. I can't see a use case for
            manually passing this, but it's here to break ties in property
            unweighted_medoid.
        :return: index of the mediod, respecting weights
        """
        ixs_ = self.ixs if ixs is None else np.array(list(ixs), dtype=np.int32)
        if len(ixs_) == 1:
            return int(ixs_[0])
        pmatrix = self.members.weighted_pmatrix[np.ix_(ixs_, ixs_)]
        return ixs_[np.argmin(pmatrix.sum(axis=1))]

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

        row_sums = self.members.pmatrix[np.ix_(self.ixs, self.ixs)].sum(axis=1)
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
        members, but a vector that is likely not coincident with any member.
        """
        weights = self.members.weights[self.ixs].reshape(-1, 1)
        return get_stacked_medians(self.members.vectors[self.ixs], weights)

    # ===========================================================================
    #   covariance matrix and eigenvectors
    # ===========================================================================

    @property
    def _covariance_matrix(self) -> FPArray:
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
        eigenvalues, eigenvectors = np.linalg.eig(self._covariance_matrix)
        return np.real(eigenvalues), np.real(eigenvectors)

    @property
    def _direction_of_highest_variance(self) -> FPArray:
        """Get the first Eigenvector of the covariance matrix.

        :return: first Eigenvector of the covariance matrix

        Return the normalized eigenvector with the largest eigenvalue.
        """
        eigenvalues, eigenvectors = self._np_linalg_eig
        return eigenvectors[:, np.argmax(eigenvalues)]

    # ===========================================================================
    #   quality metrics
    # ===========================================================================

    @functools.cached_property
    def sum_error(self) -> float:
        """Get the sum of proximity errors of all members.

        :return: sum of errors of all members
        """
        if len(self.ixs) == 1:
            return 0
        pmatrix = self.members.weighted_pmatrix
        weights = pmatrix[self.centroid, self.ixs]
        return float(np.sum(weights))

    @functools.cached_property
    def max_error(self) -> float:
        """Get the max of proximity errors of all members.

        :return: max of proximity errors of all members

        As elsewhere, weights are treated as frequencies, so the maximum error is the
        error of the most distant member, regardless of weight, even if that weight
        is 0.
        """
        if len(self.ixs) == 1:
            return 0
        pmatrix = self.members.pmatrix
        weights = pmatrix[self.centroid, self.ixs]
        return float(np.max(weights))

    @functools.cached_property
    def span(self) -> float:
        """Get the maximum proximity error between any two members.

        :return: maximum squared error of all members
        """
        if len(self.ixs) == 1:
            return 0
        pmatrix = self.members.pmatrix
        weights = pmatrix[np.ix_(self.ixs, self.ixs)]
        return float(np.max(weights))

    @functools.cached_property
    def avg_error(self) -> float:
        """Get the average error of the cluster members.

        :return: max_error / error_weight

        This metric is useful for splitting clusters with meaningful outliers, where
        sum_error might split heavy clusters, even when the members are close.
        """
        if self.error_weight == 0:
            return 0
        return self.sum_error / self.error_weight

    # ===========================================================================
    #   examine merge candidates
    # ===========================================================================

    def get_merge_span(self, other: Cluster) -> float:
        """Get the complete linkage error of merging this cluster with another.

        :return: max error between any two members of self | other
        """
        return float(np.max(self.members.pmatrix[np.ix_(self.ixs, other.ixs)]))

    # ===========================================================================
    #   the only verb
    # ===========================================================================

    def split(self) -> tuple[Cluster, Cluster]:
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
        return (
            Cluster(self.members, [x for *_, x in scored[:split]]),
            Cluster(self.members, [x for *_, x in scored[split:]]),
        )
