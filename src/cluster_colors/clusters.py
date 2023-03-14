"""The members, clusters, and groups of clusters.

:author: Shay Hill
:created: 2023-01-17
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import functools
from contextlib import suppress
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from colormath.color_conversions import convert_color  # type: ignore
from colormath.color_diff import delta_e_cie2000  # type: ignore
from colormath.color_objects import LabColor, sRGBColor  # type: ignore
from stacked_quantile import get_stacked_median, get_stacked_medians

from cluster_colors.distance_matrix import DistanceMatrix
from cluster_colors.type_hints import FPArray, StackedVectors, Vector, VectorLike

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt

# looks like python-colormath is abandoned. The code on PyPI will not work with the
# latest numpy because asscaler has been removed from numpy. This kludges it.


def patch_asscalar(a: npt.NDArray[np.float_]) -> float:
    """Alias for np.item().

    :param a: numpy array
    :return: input array as scalar
    """
    return a.item()


np.asscalar = patch_asscalar  # type: ignore

# </ patch numpy asscalar for colormath


def _get_squared_error(vector_a: VectorLike, vector_b: VectorLike) -> float:
    """Get squared distance between two vectors.

    :param vector_a: vector
    :param vector_b: vector
    :return: squared Euclidian distance from vector_a to vector_b
    """
    squared_error: np.floating[Any] = np.sum(np.subtract(vector_a, vector_b) ** 2)
    return float(squared_error)


class Member:
    """A member of a cluster.

    When clustering initial images arrays, the weight will only represent the number
    of times the color appears in the image. After removing some color or adding an
    alpha channel, the weight will also reflect the alpha channel, with transparent
    colors weighing less.
    """

    def __init__(self, weighted_vector: Vector) -> None:
        """Create a new Member instance.

        :param weighted_vector: a vector with a weight in the last axis
        :param ancestors: sets of ancestors to merge
        """
        self.as_array = weighted_vector

    @property
    def vs(self) -> tuple[float, ...]:
        """All value axes of the Member as a tuple.

        :return: tuple of values that are not the weight
        """
        return tuple(self.as_array[:-1])

    @property
    def w(self) -> float:
        """Weight of the Member.

        :return: weight of the Member
        """
        return self.as_array[-1]

    @classmethod
    def new_members(cls, stacked_vectors: StackedVectors) -> set[Member]:
        """Transform an array of rgb or rgbw colors into a set of _Member instances.

        :param stacked_vectors: a list of vectors with weight channels in the last axis
        :return: set of Member instances

        Silently drop colors without weight. It is possible to return an empty set if
        no colors have weight > 0.
        """
        return {Member(v) for v in stacked_vectors if v[-1] > 0}


_ClusterT = TypeVar("_ClusterT", bound="Cluster")


class Cluster:
    """A cluster of Member instances.

    :param members: Member instances

    Hold Members in a set. It is important for convergence that the exemplar is not
    updated each time a member is added or removed. Add members from other clusters to
    queue_add and self members to queue_sub. Do not update the members or
    process_queue until each clusters' members have be offered to all other clusters.

    When all clusters that should be moved have been inserted into queues, call
    process_queue and, if changes have occurred, create a new Cluster instance for
    the next round.

    This is almost a frozen class, but the queue_add, queue_sub, and exemplar_age
    attributes are mutable.
    """

    def __init__(self, members: Iterable[Member]) -> None:
        """Initialize a Cluster instance.

        :param members: Member instances
        :raise ValueError: if members is empty
        """
        if not members:
            msg = "Cannot create an empty cluster"
            raise ValueError(msg)
        self.members = set(members)
        self.exemplar_age = 0
        self.queue_add: set[Member] = set()
        self.queue_sub: set[Member] = set()
        # cache calls to self.se
        self._se_cache: dict[int, float] = {}
        self._vss: FPArray | None = None
        self._ws: FPArray | None = None

    @classmethod
    def from_stacked_vectors(
        cls: type[_ClusterT], stacked_vectors: FPArray
    ) -> _ClusterT:
        """Create a Cluster instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
        :return: A Cluster instance
        """
        return cls(Member.new_members(stacked_vectors))

    @functools.cached_property
    def as_array(self) -> FPArray:
        """Cluster as an array of member arrays.

        :return: array of member arrays [[x, y, z, w], [x, y, z, w], ...]
        """
        return np.array([member.as_array for member in self.members])

    def _set_vss_ws(self):
        """Set the cached vss and ws properties."""
        self._vss, self._ws = np.split(self.as_array, [-1], axis=1)

    @property
    def vss(self) -> FPArray:
        """Values for cluster members.

        :return: array of values e.g., (x, y, z) for each member (x, y, z, w)
        """
        if self._vss is None:
            self._set_vss_ws()
        assert self._vss is not None
        return self._vss

    @property
    def ws(self) -> FPArray:
        """Weights for cluster members.

        :return: array of weights e.g., (w,) for each member (x, y, z, w)
        """
        if self._ws is None:
            self._set_vss_ws()
        assert self._ws is not None
        return self._ws

    @functools.cached_property
    def vs(self) -> tuple[float, ...]:
        """Values for cluster as a member instance.

        :return: tuple of values (x, y, z, w)
        """
        vss, ws = self.vss, self.ws
        return tuple(get_stacked_medians(vss, ws))

    @functools.cached_property
    def w(self) -> float:
        """Total weight of members.

        :return: total weight of members
        """
        ws = self.ws
        return float(np.sum(ws))

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
    def exemplar_lab(self) -> tuple[float, float, float]:
        """The color description used for Cie distance.

        :return: LabColor instance
        """
        srgb = cast(Any, sRGBColor(*self.exemplar, is_upscaled=True))
        lab = cast(Any, convert_color(srgb, LabColor))
        return lab.lab_l, lab.lab_a, lab.lab_b

    @functools.cached_property
    def as_member(self) -> Member:
        """Get cluster as a Member instance.

        :return: Member instance with rgb and weight of exemplar
        """
        vector = np.array((*self.vs, self.w))
        return Member(cast(Vector, vector))

    @functools.cached_property
    def _np_linalg_eig(self) -> tuple[FPArray, FPArray]:
        """Cache the value of np.linalg.eig on the covariance matrix of the cluster.

        :return: tuple of eigenvalues and eigenvectors
        """
        vss, ws = self.vss, self.ws
        covariance_matrix: FPArray = np.cov(vss.T, aweights=ws.flatten())
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return eigenvalues.astype(float), eigenvectors.astype(float)

    @functools.cached_property
    def _variance(self) -> float:
        """Get the variance of the cluster.

        :return: variance of the cluster
        """
        return max(self._np_linalg_eig[0])

    @functools.cached_property
    def _direction_of_highest_variance(self) -> FPArray:
        """Get the first Eigenvector of the covariance matrix.

        :return: first Eigenvector of the covariance matrix

        Return the normalized eigenvector with the largest eigenvalue.
        """
        eigenvalues, eigenvectors = self._np_linalg_eig
        return eigenvectors[:, np.argmax(eigenvalues)]

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
        :raises ValueError: if cluster has only one member

        Split the cluster into two clusters by the plane perpendicular to the axis of
        highest variance.

        The splitting is a bit funny due to particulars of the stacked median. See
        stacked_quantile module for details.
        """
        if len(self.members) == 1:
            msg = "Cannot split a cluster with only one member"
            raise ValueError(msg)
        if len(self.members) == 2:
            a, b = self.members
            return {Cluster([a]), Cluster([b])}

        abc = self._direction_of_highest_variance

        def get_rel_dist(rgb: VectorLike) -> float:
            """Get relative distance of rgb from plane Ax + By + Cz + 0.

            :param rgb: color to get distance from plane
            :return: relative distance of rgb from plane
            """
            return float(np.dot(abc, rgb))

        scored = [(get_rel_dist(member.vs), member) for member in self.members]
        median_score = get_stacked_median(
            np.array([s for s, _ in scored]), np.array([m.w for _, m in scored])
        )
        left = {m for s, m in scored if s < median_score}
        right = {m for s, m in scored if s > median_score}
        center = {m for s, m in scored if s == median_score}
        if center and sum(m.w for m in left) < sum(m.w for m in right):
            left |= center
        else:
            right |= center
        return {Cluster(left), Cluster(right)}

    def se(self, member_candidate: Member) -> float:
        """Get the cost of adding a member to this cluster.

        :param member_candidate: Member instance
        :return: cost of adding member to this cluster
        """
        hash_ = hash(member_candidate)
        with suppress(KeyError):
            return self._se_cache[hash_]
        se = _get_squared_error(member_candidate.vs, self.exemplar)
        self._se_cache[hash_] = se
        return se

    @functools.cached_property
    def sse(self) -> float:
        """Get the sum of squared errors of all members.

        :return: sum of squared errors of all members
        """
        return sum(self.se(member) * member.w for member in self.members)

    def process_queue(self) -> Cluster:
        """Process the add and sub queues and update exemplars.

        :return: self
        """
        if self.queue_add or self.queue_sub:
            new_members = self.members - self.queue_sub | self.queue_add
            # clean this up in case it is cached somewhere
            self.exemplar_age = 0
            self.queue_add.clear()
            self.queue_sub.clear()
            return Cluster(new_members)
        self.exemplar_age += 1
        return self


def _get_cluster_delta_e_cie2000(cluster_a: Cluster, cluster_b: Cluster) -> float:
    """Get perceptual color distance between two clusters.

    :param cluster_a: Cluster
    :param cluster_b: Cluster
    :return: perceptual distance from cluster_a.exemplar to cluster_b.exemplar
    """
    labcolor_a = cast(Any, LabColor(*cluster_a.exemplar_lab))
    labcolor_b = cast(Any, LabColor(*cluster_b.exemplar_lab))
    return cast(float, delta_e_cie2000(labcolor_a, labcolor_b))


_ClustersT = TypeVar("_ClustersT", bound="Clusters")


class Clusters:
    """A set of Cluster instances with cached distances and queued updates.

    Maintains a cached matrix of squared distances between all Cluster exemplars.
    Created for cluster algorithms which passes members around *before* updating
    exemplars, so any changes identified must be staged in each Cluster's queue_add
    and queue_sub sets then applied with _Clusters.process_queues.
    """

    def __init__(self, clusters: Iterable[Cluster]) -> None:
        """Create a new Clusters instance."""
        self._clusters: set[Cluster] = set()
        self.spans: DistanceMatrix[Cluster]
        self.spans = DistanceMatrix(_get_cluster_delta_e_cie2000)
        self.add(*clusters)

    @classmethod
    def from_stacked_vectors(
        cls: type[_ClustersT], stacked_vectors: FPArray
    ) -> _ClustersT:
        """Create a Clusters instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
        :return: A Clusters instance
        """
        return cls({Cluster(Member.new_members(stacked_vectors))})

    def __iter__(self) -> Iterator[Cluster]:
        """Iterate over clusters.

        :return: iterator
        """
        return iter(self._clusters)

    def __len__(self) -> int:
        """Get number of clusters.

        :return: number of clusters
        """
        return len(self._clusters)

    def add(self, *cluster_args: Cluster) -> None:
        """Add clusters to the set.

        :param cluster_args: Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self._clusters.add(cluster)
            self.spans.add(cluster)

    def remove(self, *cluster_args: Cluster) -> None:
        """Remove clusters from the set and update the distance matrix.

        :param cluster_args: a Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self._clusters.remove(cluster)
            self.spans.remove(cluster)

    def sync(self, clusters: set[Cluster]) -> None:
        """Match the set of clusters to the given set.

        :param clusters: set of clusters

        This can be used to roll back changes to a previous cluster set. Come caches
        will be lost, but this keeps it simple. If you want to capture the state of a
        Clusters instance, just use `state = set(instance._clusters)`.
        """
        self.remove(*(self._clusters - clusters))
        self.add(*(clusters - self._clusters))

    def process_queues(self) -> None:
        """Apply queued updates to all Cluster instances."""
        processed = {c.process_queue() for c in self._clusters}
        self.sync(processed)

    @property
    def _has_clear_winner(self) -> bool:
        """Is one cluster heavier than the rest.

        :return: True if one cluster is heavier than the rest. Will almost always be
            true.

        It is up to child classes to decide if and how to fix a situation that has no
        clear winner.
        """
        if len(self) == 1:
            return True
        weights = [c.w for c in self]
        return weights.count(max(weights)) == 1

    def _maybe_split_cluster(self, min_error_to_split: float = 0) -> bool:
        """Split the cluster with the highest SSE. Return True if a split occurred.

        :param clusters: the clusters to split
        :param min_error_to_split: the cost threshold for splitting, default 0
        :return: Parent of split clusters if split occurred, None if no split occurred.

        Could potentially make multiple splits if max_error is a tie, but this is
        unlikely. The good news is that this will be deterministic.
        """
        candidates = [c for c in self if len(c.members) > 1]
        if not candidates:
            return False
        graded = [(c.sse, c) for c in candidates]
        max_error, cluster = max(graded, key=itemgetter(0))
        if max_error < min_error_to_split:
            return False
        for cluster in (c for g, c in graded if g == max_error):
            self.remove(cluster)
            self.add(*cluster.split())
        return True

    def _maybe_merge_cluster(self, merge_below_cost: float = np.inf) -> bool:
        """Merge the two clusters with the lowest exemplar span.

        :param clusters: the clusters to merge
        :param merge_below_cost: the cost threshold for merging, default
            infinity
        :return: True if a merge occurred

        Could potentially make multiple merges if min_cost is a tie, but this is
        unlikely. The good news is that this will be deterministic.
        """
        if len(self) < 2:
            return False
        min_cluster_a, min_cluster_b = self.spans.keymin()
        min_cost = self.spans(min_cluster_a, min_cluster_b)
        if min_cost > merge_below_cost:
            return False
        combined_members = min_cluster_a.members | min_cluster_b.members
        self.remove(min_cluster_a, min_cluster_b)
        self.add(Cluster(combined_members))
        return True

    ## Cluster Reassignment

    def _get_others(self, cluster: Cluster) -> set[Cluster]:
        """Identify other clusters with the potential to take members from cluster.

        :param cluster: the cluster offering its members to other clusters
        :return: other clusters with the potential to take members

        Two optimizations:

        1.  Don't compare old clusters with other old clusters.
            These clusters are old because they have not changed since the last time
            they were compared.

        2.  Don't compare clusters with a squared distance greater than four times
            the squared distance (twice the actual distance) to the farthest cluster
            member.
        """
        if len(cluster.members) == 1:
            return set()
        if cluster.exemplar_age == 0:
            others = {x for x in self._clusters if x is not cluster}
        else:
            others = {x for x in self._clusters if x.exemplar_age == 0}
        if not others:
            return others

        max_se = max(cluster.se(m) for m in cluster.members)
        return {x for x in others if self.spans(cluster, x) / 4 < max_se}

    def _offer_members(self, cluster: Cluster) -> None:
        """Look for another cluster with lower cost for members of input cluster.

        :param cluster: the cluster offering its members to other clusters
        :effect: moves members between clusters
        """
        others = self._get_others(cluster)
        if not others:
            return

        safe_cost = self.spans.min(cluster) / 4
        members = {m for m in cluster.members if cluster.se(m) > safe_cost}
        for member in members:
            best_cost = cluster.se(member)
            best_cluster = cluster
            for other in others:
                cost = other.se(member)
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = other
            if best_cluster is not cluster:
                cluster.queue_sub.add(member)
                best_cluster.queue_add.add(member)

    def _maybe_reassign_members(self) -> bool:
        """Pass members between clusters and update exemplars.

        :return: True if any changes were made
        """
        if len(self) < 2:
            return False
        if all(x.exemplar_age > 0 for x in self._clusters):
            return False
        for cluster in self._clusters:
            self._offer_members(cluster)
        return True
