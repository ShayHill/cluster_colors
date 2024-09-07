"""The members, clusters, and groups of clusters.

:author: Shay Hill
:created: 2023-01-17
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Annotated, NamedTuple, TypeVar, Literal

import numpy as np
from basic_colormath import get_delta_e_lab, get_sqeuclidean, rgb_to_lab
from paragraphs import par
from stacked_quantile import get_stacked_median, get_stacked_medians

from cluster_colors.cluster_member import Member
from cluster_colors.distance_matrix import DistanceMatrix
from cluster_colors.cluster_cluster import Cluster

_RGB = tuple[float, float, float]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from cluster_colors.type_hints import FPArray, VectorLike, Vector






def _get_cluster_delta_e_cie2000(cluster_a: Cluster, cluster_b: Cluster) -> float:
    """Get perceptual color distance between two clusters.

    :param cluster_a: Cluster
    :param cluster_b: Cluster
    :return: perceptual distance from cluster_a.exemplar to cluster_b.exemplar
    """
    dist_ab = get_delta_e_lab(cluster_a.lab, cluster_b.lab)
    dist_ba = get_delta_e_lab(cluster_b.lab, cluster_a.lab)
    return max(dist_ab, dist_ba)


_SuperclusterT = TypeVar("_SuperclusterT", bound="Supercluster")


class _State(NamedTuple):
    """The information required to revert to a previous state."""

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

        Most predicates will be true for at least one index (index 1). A
        single-cluster state has infinite inter-cluster span and should not exceed
        any maximum-cluster requirements. This package only includes predicates that
        pass 100% of the time with a 1-cluster state, but some common predicates
        (cluster SSE) do not have this guarantee. Implementing those will require a
        more intricate cache.
        """
        self.cluster_sets: list[set[Cluster] | None] = []
        self.min_spans: list[float | None] = []
        self.capture_state(supercluster)
        (cluster,) = supercluster.clusters
        self._hard_max = sum(1 for x in cluster if x.w)

    def capture_state(self, supercluster: Supercluster) -> None:
        """Capture the state of the Supercluster instance.

        :param supercluster: Supercluster instance to capture
        """
        while len(self.cluster_sets) <= len(supercluster.clusters):
            self.cluster_sets.append(None)
            self.min_spans.append(None)
        self.cluster_sets[len(supercluster.clusters)] = set(supercluster.clusters)
        self.min_spans[len(supercluster.clusters)] = supercluster.spans.valmin()

    def fwd_enumerate(self) -> Iterator[_State]:
        """Iterate over the cached cluster states.

        :return: None
        :yield: _State tuples (index_, clusters, min_span) for each viable (non-None)
            state.
        """
        for i, (clusters, min_span) in enumerate(
            zip(self.cluster_sets, self.min_spans, strict=True)
        ):
            if clusters is not None:
                if min_span is None:
                    msg = "min_span is None for non-None clusters"
                    raise ValueError(msg)
                yield _State(i, clusters, min_span)

    def rev_enumerate(self) -> Iterator[_State]:
        """Iterate backward over the cached cluster states.

        :return: None
        :yield: tuples (index_, clusters, min_span) for each viable (non-None) state.
        """
        enumerated = tuple(self.fwd_enumerate())
        yield from reversed(enumerated)

    def seek_ge(self, min_index: int) -> _State:
        """Start at min_index and move right to find a non-None state.

        :param min_index: minimum index to return
        :return: (index, clusters, and min_span) at or above index = min_index
        :raise StopIteration: if cache does not have at least min_index entries.
        """
        return next(s for s in self.fwd_enumerate() if s.index_ >= min_index)

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
        enumerated = self.fwd_enumerate()
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
        enumerated = self.fwd_enumerate()
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


def _get_all_members(*cluster_args: Cluster) -> set[Member]:
    """Return the union of all members of the given clusters.

    :param cluster_args: The clusters to get members from.
    :return: The union of all members of the given clusters.
    """
    try:
        member_sets = (x.members for x in cluster_args)
        all_members = next(member_sets).union(*member_sets)
    except StopIteration:
        return set()
    else:
        return all_members


class Supercluster:
    """A set of Cluster instances with cached distances and queued updates.

    Maintains a cached matrix of squared distances between all Cluster exemplars.
    Created for cluster algorithms which pass members around *before* updating
    exemplars, so any changes identified must be staged in each Cluster's queue_add
    and queue_sub sets then applied with Supercluster.process_queues.
    """

    def __init__(self, *members: Iterable[Member]) -> None:
        """Create a new Supercluster instance.

        :param members: initial members. All are combined into one cluster. Multiple
        arguments allowed.
        """
        all_members = set(members[0]).union(*(set(ms) for ms in members[1:]))
        self.clusters: set[Cluster] = set()
        self.spans: DistanceMatrix[Cluster]
        self.spans = DistanceMatrix(_get_cluster_delta_e_cie2000)
        self._add(Cluster(all_members))

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

        candidates = [c for c in self if c.is_splittable]
        if not candidates:
            msg = "No clusters can be split"
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
        return cls(Member.new_members(stacked_vectors))

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

    def _add(self, *cluster_args: Cluster) -> None:
        """Add clusters to the set.

        :param cluster_args: Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self.clusters.add(cluster)
            self.spans.add(cluster)

    def _remove(self, *cluster_args: Cluster) -> None:
        """Remove clusters from the set and update the distance matrix.

        :param cluster_args: a Cluster, accepts multiple args
        """
        for cluster in cluster_args:
            self.clusters.remove(cluster)
            self.spans.remove(cluster)

    def exchange(
        self, subtractions: Iterable[Cluster], additions: Iterable[Cluster]
    ) -> None:
        """Exchange clusters in the set and update the distance matrix.

        :param subtractions: clusters to remove
        :param additions: clusters to add
        :raise ValueError: if the exchange would result in a missing members
        """
        sub_members = _get_all_members(*subtractions)
        add_members = _get_all_members(*additions)
        if sub_members != add_members:
            msg = par(
                """Exchange would result in missing or extra members: {sub_members}
                != {add_members}"""
            )
            raise ValueError(msg)
        self._remove(*subtractions)
        self._add(*additions)

    def sync(self, clusters: set[Cluster]) -> None:
        """Match the set of clusters to the given set.

        :param clusters: set of clusters

        This can be used to roll back changes to a previous cluster set. Come caches
        will be lost, but this keeps it simple. If you want to capture the state of a
        Supercluster instance, just use `state = set(instance._clusters)`.
        """
        self.exchange(self.clusters - clusters, clusters - self.clusters)

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
        """Split one cluster."""
        self.exchange({cluster}, cluster.split())

    def _split_clusters(self):
        """Split one or more clusters.

        :param clusters: clusters of presumably equal error. The state after all
            splits will be stored in self._states. Intermediate states will be stored
            as None in split states.

        The overwhelming majority of the time, this will be exactly one cluster, but
        if more that one cluster share the same error, they will be split in
        parallel.

        Overload this method to implement a custom split strategy or to add a
        convergence step after splitting.
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
        if cluster.is_new:
            others = {x for x in self.clusters if x is not cluster}
        else:
            others = {x for x in self.clusters if x.is_new}
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
        if len(self) in {0, 1}:
            return False
        if all(not x.is_new for x in self.clusters):
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
    def as_cluster(self) -> Cluster:
        """Return a cluster that contains all members of all clusters.

        :return: a cluster that contains all members of all clusters

        This is a pathway to a Supercluster instance sum weight, sum exemplar, etc.
        """
        (cluster,) = next(self._states.fwd_enumerate()).clusters
        return cluster
