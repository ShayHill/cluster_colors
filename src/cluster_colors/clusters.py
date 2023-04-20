"""The members, clusters, and groups of clusters.

:author: Shay Hill
:created: 2023-01-17
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

import numpy as np
from colormath.color_conversions import convert_color  # type: ignore
from colormath.color_diff import delta_e_cie2000  # type: ignore
from colormath.color_objects import LabColor, sRGBColor  # type: ignore
from stacked_quantile import get_stacked_median, get_stacked_medians

from cluster_colors.distance_matrix import DistanceMatrix
from cluster_colors.type_hints import FPArray, StackedVectors, Vector, VectorLike

_T = TypeVar("_T", bound=object)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt


def patch_asscalar(a: npt.NDArray[np.float_]) -> float:
    """Alias for np.item(). Patch np.asscalar for colormath.

    :param a: numpy array
    :return: input array as scalar


    looks like python-colormath is abandoned. The code on PyPI will not work with the
    latest numpy because asscaler has been removed from numpy. This kludges it.
    np.asscalar is not called in this module, but it is called when computing
    delta-e.
    """
    return a.item()


np.asscalar = patch_asscalar  # type: ignore


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
        self.vss, self.ws = np.split(self.as_array, [-1], axis=1)

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

        This is the error used to determine if a cluster should be split in the
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
        return _get_squared_error(member_candidate.vs, self.exemplar)

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
    dist_ab = cast(float, delta_e_cie2000(labcolor_a, labcolor_b))
    dist_ba = cast(float, delta_e_cie2000(labcolor_b, labcolor_a))
    return max(dist_ab, dist_ba)


_SuperclusterT = TypeVar("_SuperclusterT", bound="Supercluster")


class _State(NamedTuple):
    index_: int
    clusters: set[Cluster]
    min_span: float


class StatesCache:
    """Cache the clusters and minimum inter-cluster span for each index.

    A Supercluster instance splits the heaviest cluster, if there is a tie, some states
    will be skipped. E.g., given cluster weights [1, 2, 3, 4, 4], _split_clusters
    will split both clusters with weight == 4, skipping from a 5-cluster state to a
    7-cluster state. In this instance, state 6 will be None. State 0 will always be
    None, so the index of a state is the number of clusters in that state.
    """

    def __init__(self, supercluster: Supercluster) -> None:
        """Initialize the cache.

        :param supercluster: Supercluster instance

        If clusters instance has more than one cluster, all clusters will be combined
        into one large cluster, so most predicates will be true for at least one
        index (index 1). A single-cluster state has infinite inter-cluster span and
        should not exceed any maximum-cluster requirements. This package only
        includes predicates that pass 100% of the time with a 1-cluster state, but
        some common predicates (cluster SSE) do not have this guarantee. Implementing
        those will require a more sophisticated cache.
        """
        #TODO: join all clusters at the beginning. Do not preserve input divisions
        all_members: set[Member] = set()
        all_members = all_members.union(*(x.members for x in supercluster))
        self.cluster_sets = [None, {Cluster(all_members)}]
        self.min_spans = [None, float("inf")]
        self.capture_state(supercluster)
        self._hard_max = len(all_members)

    def capture_state(self, supercluster: Supercluster) -> None:
        """Capture the state of the Supercluster instance.

        :param supercluster: Supercluster instance to capture
        """
        while len(self.cluster_sets) <= len(supercluster.clusters):
            self.cluster_sets.append(None)
            self.min_spans.append(None)
        self.cluster_sets[len(supercluster.clusters)] = set(supercluster.clusters)
        self.min_spans[len(supercluster.clusters)] = supercluster.spans.valmin()

    def _enumerate(self) -> Iterator[_State]:
        """Iterate over the cached cluster states.

        :return: None
        :yield: _State tuples (index_, clusters, min_span) for each viable (non-None)
            state.
        """
        for i, (clusters, min_span) in enumerate(zip(self.cluster_sets, self.min_spans)):
            if clusters is not None:
                assert min_span is not None
                yield _State(i, clusters, min_span)

    def reverse_enumerate(self) -> Iterator[_State]:
        """Iterate backward over the cached cluster states.

        :return: None
        :yield: tuples (index_, clusters, min_span) for each viable (non-None) state.
        """
        enumerated = tuple(self._enumerate())
        yield from reversed(enumerated)

    def seek_ge(self, min_index: int) -> _State:
        """Start at min_index and move right to find a non-None state.

        :param min_index: minimum index to return
        :return: (index, clusters, and min_span) at or above index = min_index
        :raise StopIteration: if cache does not have at least min_index entries.
        """
        return next(s for s in self._enumerate() if s.index_ >= min_index)

    def seek_le(self, max_index: int) -> _State:
        """Start at max_index and move left to find a non-None state.

        :param max_index: maximum index to return
        :return: (index, clusters, and min_span) at or below index = max_index
        :raise ValueError: if maximum index 0 is requested. No clusters instance will
            ever have 0 clusters.
        :raise StopIteration: if cache does not have at least max_index entries.
        """
        if max_index == 0:
            msg = "no Supercluster instance has 0 clusters"
            raise ValueError(msg)
        enumerated = self._enumerate()
        prev = next(enumerated)  # will always be 1
        if max_index == 1:
            return prev
        here = next(enumerated)
        while here.index_ < max_index:
            prev = here
            here = next(enumerated)
            if here.index_ == max_index:
                return here
        return prev

    def seek_while(
        self, max_count: int | None = None, min_span: float | None = None
    ) -> _State:
        """Seek to the rightmost state that satisfies the given conditions.

        :param max_count: The maximum number of clusters to allow.
        :param min_span: The minimum span to allow. If this is low and no max_count
            is given, expect to split all the way down to singletons, which could
            take several seconds.
        :return: The number of clusters in the state that was found.
        :raises StopIteration: if all states satisfy condition. In this case, we
            won't know if we are at the rightmost state.

        When max count is one, prev will only ever have the value None. It is not
        possible to fail other tests in this state, so the `prev or state` return
        values will never be returned when prev is still None. This is because a
        single cluster has a minimum span of infinity.
        """
        max_count = max_count or self._hard_max
        max_count = min(max_count, self._hard_max)
        min_span = 0 if min_span is None else min_span
        enumerated = self._enumerate()
        prev = None
        for state in enumerated:
            if state.min_span < min_span:  # this is the first one that is too small
                return prev or state
            if state.index_ > max_count:  # overshot because tied clusters were split
                return prev or state
            if state.index_ == max_count:  # reached maximum count
                return state
            prev = state
        raise StopIteration


class Supercluster:
    """A set of Cluster instances with cached distances and queued updates.

    Maintains a cached matrix of squared distances between all Cluster exemplars.
    Created for cluster algorithms which passes members around *before* updating
    exemplars, so any changes identified must be staged in each Cluster's queue_add
    and queue_sub sets then applied with _Supercluster.process_queues.
    """

    def __init__(self, clusters: Iterable[Cluster]) -> None:
        """Create a new Supercluster instance.

        Will not merge clusters, so the minimum number of clusters will be the number
        of clusters at instantiation.
        """
        self.clusters: set[Cluster] = set()
        self.spans: DistanceMatrix[Cluster]
        self.spans = DistanceMatrix(_get_cluster_delta_e_cie2000)
        self.add(*clusters)

        self._states = StatesCache(self)
        self._next_to_split: set[Cluster] = set()

    @property
    def next_to_split(self) -> set[Cluster]:
        """Return the next set of clusters to split.

        :return: set of clusters with sse == max(sse)
        :raise ValueError: if no clusters are available to split

        These will be the clusters (multiple if tie, which should be rare) with the
        highest sse.
        """
        self._next_to_split &= self.clusters
        if self._next_to_split:
            return self._next_to_split

        candidates = [c for c in self if len(c.members) > 1]
        if not candidates:
            msg = "All clusters have only one member."
            raise ValueError(msg)
        max_error = max(c.sse for c in candidates)
        self._next_to_split = {c for c in candidates if c.sse == max_error}
        return self._next_to_split

    @classmethod
    def from_stacked_vectors(
        cls: type[_SuperclusterT], stacked_vectors: FPArray
    ) -> _SuperclusterT:
        """Create a Supercluster instance from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
        :return: A Supercluster instance
        """
        return cls({Cluster(Member.new_members(stacked_vectors))})

    def __iter__(self) -> Iterator[Cluster]:
        """Iterate over clusters.

        :return: iterator
        """
        return iter(self.clusters)

    def __len__(self) -> int:
        """Get number of clusters.

        :return: number of clusters
        """
        return len(self.clusters)

    def add(self, *cluster_args: Cluster) -> None:
        """Add clusters to the set.

        :param cluster_args: Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self.clusters.add(cluster)
            self.spans.add(cluster)

    def remove(self, *cluster_args: Cluster) -> None:
        """Remove clusters from the set and update the distance matrix.

        :param cluster_args: a Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self.clusters.remove(cluster)
            self.spans.remove(cluster)

    def sync(self, clusters: set[Cluster]) -> None:
        """Match the set of clusters to the given set.

        :param clusters: set of clusters

        This can be used to roll back changes to a previous cluster set. Come caches
        will be lost, but this keeps it simple. If you want to capture the state of a
        Supercluster instance, just use `state = set(instance._clusters)`.
        """
        self.remove(*(self.clusters - clusters))
        self.add(*(clusters - self.clusters))

    def process_queues(self) -> None:
        """Apply queued updates to all Cluster instances."""
        processed = {c.process_queue() for c in self.clusters}
        self.sync(processed)

    # ------------------------------------------------------------------------ #
    #
    #  split clusters
    #
    # ------------------------------------------------------------------------ #

    def _split_cluster(self, cluster: Cluster):
        """Split one cluster.

        Overload this method to implement a custom split strategy or to add a
        convergence step after splitting.
        """
        self.remove(cluster)
        self.add(*cluster.split())

    def _split_clusters(self):
        """Split one or more clusters.

        :param clusters: clusters of presumably equal error. The state after all
            splits will be stored in self._states. Intermediate states will be stored
            as None in split states.

        The overwhelming majority of the time, this will be exactly one cluster, but
        if more that one cluster share the same error, they will be split in
        parallel.
        """
        for cluster in tuple(self.next_to_split):
            self._split_cluster(cluster)
        self._states.capture_state(self)

    def split_until(self, max_count: int | None = None, min_span: float | None = None):
        """Split enough to break one or both conditions, then back up one step.

        :param max_count: maximum number of clusters
        :param min_span: minimum span between clusters (in delta-e)
        """
        try:
            self.sync(self._states.seek_while(max_count, min_span).clusters)
        except StopIteration:
            self._split_clusters()
            self.split_until(max_count, min_span)

    def split_to_at_most(self, count: int):
        """An alias for split_until(max_count=count) to clarify intent.

        :param count: maximum number of clusters
        """
        self.split_until(max_count=count)

    def split_to_delta_e(self, min_delta_e: float):
        """An alias for split_until(min_span=min_delta_e) to clarify intent.

        :param min_delta_e: minimum span between clusters (in delta-e)
        """
        self.split_until(min_span=min_delta_e)

    # ------------------------------------------------------------------------ #
    #
    #  return sorted clusters or examplars
    #
    # ------------------------------------------------------------------------ #

    @property
    def _no_two_clusters_have_same_weight(self) -> bool:
        """Do all clusters have a unique weight?

        :return: True if any two clusters have the same weight

        This ensures the biggest cluster is the biggest cluster, not "one of the
        biggest clusters". Also ensures that sorting clusters by weight is
        deterministic and non-arbitrary.
        """
        if len(self) == 1:
            return True
        weights = {c.w for c in self}
        return len(weights) == len(self)

    def _merge_to_break_ties(self):
        """Revert to previous state until no two clusters have the same weight.

        This will always succeed because there will always be a state with only one
        cluster.
        """
        while not self._no_two_clusters_have_same_weight:
            self.sync(self._states.seek_le(len(self) - 1).clusters)

    def get_rsorted_clusters(self) -> list[Cluster]:
        """Return clusters from largest to smallest, breaking ties.

        :return: a reverse-sorted (by weight) list of clusters

        This may not return the same clusters as the iterator, because the iterator
        will not break ties. Tie-breaking will rarely be needed, but this method
        makes sure things are 100% deterministic and non-arbitrary.
        """
        return sorted(self.clusters, key=lambda c: c.w, reverse=True)

    def get_rsorted_exemplars(self) -> list[tuple[float, ...]]:
        """Return clusters from largest to smallest, breaking ties.

        :return: a reverse-sorted (by weight) list of cluster exemplars

        This may not return the same clusters as the iterator, because the iterator
        will not break ties. Tie-breaking will rarely be needed, but this method
        makes sure things are 100% deterministic and non-arbitrary.
        """
        return [x.exemplar for x in self.get_rsorted_clusters()]

    # ------------------------------------------------------------------------ #
    #
    #  compare clusters and queue members for reassignment
    #
    # ------------------------------------------------------------------------ #

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
            others = {x for x in self.clusters if x is not cluster}
        else:
            others = {x for x in self.clusters if x.exemplar_age == 0}
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

        safe_cost = self.spans.min_from_item(cluster) / 4
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
        if all(x.exemplar_age > 0 for x in self.clusters):
            return False
        for cluster in self.clusters:
            self._offer_members(cluster)
        return True

    # ------------------------------------------------------------------------ #
    #
    #  treat it like a cluster
    #
    # ------------------------------------------------------------------------ #

    @property
    def as_one_cluster(self) -> Cluster:
        """Return a cluster that contains all members of all clusters.

        :return: a cluster that contains all members of all clusters

        This is a pathway to a Supercluster instance sum weight, sum exemplar, etc.
        """
        members_list = [x.members for x in self.clusters]
        return Cluster(members_list[0].union(*members_list[1:]))
