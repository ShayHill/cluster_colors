"""The members, clusters, and groups of clusters.

:author: Shay Hill
:created: 2023-01-17
"""

from __future__ import annotations

import itertools as it
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from cluster_colors.cluster_cluster import Cluster
from cluster_colors.cluster_member import Members

_CachedState = tuple[tuple[int, ...], ...]

_RGB = tuple[float, float, float]


if TYPE_CHECKING:
    from collections.abc import Callable

    from cluster_colors.type_hints import FPArray, ProximityMatrix, Vectors


class FailedToSplitError(Exception):
    """Exception raised when no clusters can be split."""

    def __init__(self, message: str | None = None) -> None:
        """Create a new AllClustersAreSingletonsError instance."""
        message_ = message or "Cannot split any cluster. All clusters are singletons."
        self.message = message_
        super().__init__(self.message)


class FailedToMergeError(Exception):
    """Exception raised when no clusters can be merged."""

    def __init__(self, message: str | None = None) -> None:
        """Create a new CannotMergeSingleCluster instance."""
        message_ = message or "Cannot merge any cluster. All members in one cluster."
        self.message = message_
        super().__init__(self.message)


_SuperclusterT = TypeVar("_SuperclusterT", bound="_Supercluster")


class _Supercluster(ABC):
    """A set of Cluster instances.

    Cache states (sets of cluster indices given a number of clusters) when splitting
    or merging.

    A supercluster that starts as one large cluster will cache states as that cluster
    and its descendants are split, and merging from any state in that cluster will be
    loading a previouly cached state.

    Similarly, a supercluster that starts as singletons will cache states as those
    singletons and their descendants are merged, and splitting from any state in that
    cluster will be loading a previously cached state.

    The result of this is that a supercluster started as one large cluster will never
    merge (only split and un-split) and a supercluster started as singletons will
    never split (only merge and un-merge). The only thing required to make this a
    divisive or agglomerative class is to implement the _initialize_clusters method
    to return either a single cluster or a cluster for each member.

    There is a slight trick in implementing this method, as the clusters are kept as
    a `dict[Cluster, None]` but are otherwise treated as a set. This is to maintain
    insertion order, which is important for tie-breaking with the same error metric
    (older clusters are split first in case of a tie). Ties are unlikely to happen
    when clustering colors, but the class is available for clustering sequential
    integers as (1,) arrays.
    """

    def __init__(self, members: Members) -> None:
        """Create a new Supercluster instance.

        :param members: Members instance

        Clusters are kept in a dictionary to maintain insertion order to make
        tie-breaking with the same error metric deterministic.
        """
        self.members = members
        self.clusters = self._initialize_clusters()
        self._cached_states: list[_CachedState] = []
        self._cache_current_state()

    @abstractmethod
    def _initialize_clusters(self) -> dict[Cluster, None]:
        """Create clusters from the members."""
        ...

    # ===========================================================================
    #   constructors
    # ===========================================================================

    @classmethod
    def from_vectors(
        cls: type[_SuperclusterT],
        vectors: Vectors,
        pmatrix: ProximityMatrix | None = None,
    ) -> _SuperclusterT:
        """Create a new Supercluster instance from stacked vectors.

        :param stacked_vectors: members as a numpy array (n, m+1) with the last
            column as the weight.
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        """
        members = Members.from_vectors(vectors, pmatrix=pmatrix)
        return cls(members)

    @classmethod
    def from_stacked_vectors(
        cls: type[_SuperclusterT],
        stacked_vectors: Vectors,
        pmatrix: ProximityMatrix | None = None,
    ) -> _SuperclusterT:
        """Create a new Supercluster instance from stacked vectors.

        :param stacked_vectors: members as a numpy array (n, m+1) with the last
            column as the weight.
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        """
        members = Members.from_stacked_vectors(stacked_vectors, pmatrix=pmatrix)
        return cls(members)

    # ===========================================================================
    #   properties
    # ===========================================================================

    @property
    def n(self) -> int:
        """Return the number of clusters in the Supercluster instance."""
        return len(self.clusters)

    @property
    def as_stacked_vectors(self) -> Vectors:
        """Return the members as a numpy array, sorted heaviest to lightest."""
        as_stacked_vectors = np.array([c.as_stacked_vector for c in self.clusters])
        return as_stacked_vectors[np.argsort(as_stacked_vectors[:, -1])][::-1]

    @property
    def as_vectors(self) -> FPArray:
        """Return the members as a numpy array, sorted heaviest to lightest."""
        return self.as_stacked_vectors[:, :-1]

    # ===========================================================================
    #   cacheing and state management
    # ===========================================================================

    def _cache_current_state(self) -> None:
        """Cache the current state of the Supercluster instance.

        Call this at init and after every split or merge. These calls are already in
        the existing methods.
        """
        try:
            _ = self._get_cached_state(self.n)
        except IndexError:
            self._cached_states.append(tuple(tuple(c.ixs) for c in self.clusters))

    def _get_cached_state(self, n: int) -> _CachedState:
        """Get the cached state of the Supercluster with n clusters.

        :param n: number of clusters in the state
        :return: the state with n clusters
        :raise IndexError: if the state has not been cached

        This uses an indexing mechanic that will work with either divisive or
        agglomerative clustering.
        """
        idx = abs(n - len(self._cached_states[0]))
        try:
            return self._cached_states[idx]
        except IndexError as e:
            msg = f"State {n} has not been cached."
            raise IndexError(msg) from e

    def _restore_cached_state(self, state: _CachedState) -> None:
        """Restore a previous state of the Supercluster instance.

        :param state: state to restore
        :raise IndexError: if the state has not been cached

        Retains shared clusters between the current state and cached state to
        preserve cached values and relative values of cluster serial numbers.
        """
        state_list = list(state)
        for cluster in tuple(self.clusters):
            ixs = tuple(cluster.ixs)
            if ixs in state:
                state_list.remove(ixs)
            else:
                del self.clusters[cluster]
        for ixs in state_list:
            self.clusters[Cluster(self.members, ixs)] = None

    def _restore_state_to_n(self, n: int) -> None:
        """Restore the Supercluster instance to n clusters.

        :param n: desired number of clusters
        """
        if n == self.n:
            return
        state = self._get_cached_state(n)
        self._restore_cached_state(state)

    def _restore_state_as_close_as_possible_to_n(self, n: int) -> None:
        """Restore the Supercluster to the nearest state to n clusters.

        :param n: desired number of clusters

        If as state has not been cached with the desired number of clusters, get as
        close as possible.
        """
        with suppress(IndexError):
            self._restore_state_to_n(n)
            return
        state = self._cached_states[-1]
        if len(state) == self.n:
            return
        self._restore_cached_state(state)

    # ===========================================================================
    #   select clusters to split or merge
    # ===========================================================================

    def _get_next_to_split(self) -> Cluster:
        """Return the next set of clusters to split.

        :return: set of clusters with sse == max(sse)
        :raise ValueError: if no clusters are available to split

        These will be the clusters (multiple if tie, which should be rare) with the
        highest sse.
        """
        return max(self.clusters, key=lambda c: c.error_metric)

    def _get_next_to_merge(self) -> tuple[Cluster, Cluster]:
        """Return the next set of clusters to merge.

        :return: set of clusters with sse == min(sse)
        :raise ValueError: if no clusters are available to merge

        These will be the clusters (multiple if tie, which should be rare) with the
        lowest sse.
        """
        pairs = it.combinations(self.clusters, 2)
        return min(pairs, key=lambda p: p[0].get_merge_error_metric(p[1]))

    # ===========================================================================
    #   perform splits and merges
    # ===========================================================================

    def _split_to_n(self, n: int) -> None:
        """Split or restore the Supercluster instance to n clusters.

        :param n: number of clusters
        """
        self._restore_state_as_close_as_possible_to_n(n)
        while self.n < n:
            cluster = self._get_next_to_split()
            del self.clusters[cluster]
            self.clusters.update({c: None for c in cluster.split()})
            self._reassign()
            self._cache_current_state()

    def _merge_to_n(self, n: int) -> None:
        """Merge or restore the Supercluster instance to n clusters.

        :param n: number of clusters
        """
        self._restore_state_as_close_as_possible_to_n(n)
        while self.n > n:
            cluster_a, cluster_b = self._get_next_to_merge()
            merged = Cluster(self.members, cluster_a.ixs + cluster_b.ixs)
            del self.clusters[cluster_a]
            del self.clusters[cluster_b]
            self.clusters[merged] = None

    # ===========================================================================
    #   public methods
    # ===========================================================================

    def set_n(self, n: int) -> None:
        """Set the number of clusters in the Supercluster instance.

        :param n: number of clusters
        """
        self._split_to_n(n)
        self._merge_to_n(n)

    def split(self):
        """Split the cluster with the highest sum error.

        This sets the state of the Supercluster instance. If the state is already
        >=n, nothing happens.
        """
        if len(self.clusters) == len(self.members):
            raise FailedToSplitError
        self._split_to_n(self.n + 1)

    def merge(self):
        """Merge the two clusters with the lowest sum error.

        This sets the state of the Supercluster instance. If the state is already
        <=n, nothing happens.
        """
        if len(self.clusters) == 1:
            raise FailedToMergeError
        self._merge_to_n(self.n - 1)

    def split_amrap(self, predicate: Callable[[_Supercluster], bool]) -> None:
        """Split as many reps as possible before the predicate is False.

        :param predicate: function that takes a Supercluster instance and returns a
            boolean. It is critical that this predicate returns true when
            self.n == 1

        Split until the predicate is False, then back up one step. If the predicate
        is False when starting, back up to a state where the predicate is True. If
        all clusters are singletons, and the predicate is still True, the
        Supercluster instance will be left in an "atomized" (one member per cluster)
        state.
        """
        while predicate(self):
            self.split()
        while not predicate(self):
            self.merge()

    def get_min_proximity(self) -> float:
        """Return the minimum intercluster proximity."""
        if self.n == 1:
            return 0
        ixs = [c.weighted_medoid for c in self.clusters]
        pmat = self.members.pmatrix_with_inf_diagonal
        return float(np.min(pmat[np.ix_(ixs, ixs)]))

    def set_min_proximity(self, min_proximity: float):
        """Split as far as possible while maintaining a minimum inter-cluster span.

        Split until the condition is broken, then back up one step. If the condition
        is broken at the start, back up all the way to a 1-cluster state before
        splitting.
        """

        def predicate(supercluster: _Supercluster) -> bool:
            return supercluster.get_min_proximity() < min_proximity

        self.split_amrap(predicate)

    def _reassign(self, _previous_medoids: set[tuple[int, ...]] | None = None):
        """Reassign members based on proximity to cluster medoids.

        :param _previous_medoids: set of cluster medoids that have already been seen.
            For recursion use only

        Recursively redistribute members between clusters until no member can be
        moved to a different cluster to reduce the total error.

        Convergence uses the cluster medoid, not the cluster exemplar. This allows
        the clusters a bit more mobility, so the separation of two heavy,
        nearly-identical clusters is not destiny.

        A record of previous states prevents infinite recursion between a few states.
        It is conceivable that conversion could fail in other cases. The recursion
        limit is set to the Python's recursion limit.

        This will only ever be called for divisive clustering.
        """
        medoids = [c.unweighted_medoid for c in self.clusters]

        previous_states = _previous_medoids or set()
        state = tuple(sorted(medoids))
        if state in previous_states:
            return
        previous_states.add(state)

        which_medoid = np.argmin(self.members.pmatrix[medoids], axis=0)
        for i, cluster in enumerate(tuple(self.clusters)):
            new_where = np.argwhere(which_medoid == i)
            new = list(map(int, new_where.flatten()))
            if new != list(cluster.ixs):
                del self.clusters[cluster]
                self.clusters[Cluster(self.members, new)] = None

        with suppress(RecursionError):
            self._reassign(previous_states)


class DivisiveSupercluster(_Supercluster):
    """A set of Cluster instances for divisive clustering."""

    def _initialize_clusters(self) -> dict[Cluster, None]:
        return {Cluster(self.members): None}


class AgglomerativeSupercluster(_Supercluster):
    """A set of Cluster instances for agglomerative clustering."""

    def _initialize_clusters(self) -> dict[Cluster, None]:
        return {Cluster(self.members, [i]): None for i in range(len(self.members))}
