"""The members, clusters, and groups of clusters.

:author: Shay Hill
:created: 2023-01-17
"""

from __future__ import annotations

from paragraphs import par
import bisect
import functools
import itertools as it
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
    Callable,
)

import numpy as np
from basic_colormath import get_delta_e_lab, get_sqeuclidean, rgb_to_lab
from numpy import typing as npt
from paragraphs import par
from stacked_quantile import get_stacked_median, get_stacked_medians

from cluster_colors.cluster_cluster import Cluster
from cluster_colors.cluster_member import Member, Members
from cluster_colors.distance_matrix import DistanceMatrix

_RGB = tuple[float, float, float]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from cluster_colors.type_hints import (
        FPArray,
        ProximityMatrix,
        StackedVectors,
        Vector,
        VectorLike,
    )


class AllClustersAreSingletonsError(Exception):
    """Exception raised when no clusters can be split."""

    def __init__(
        self, message: str = "Cannot split any cluster. All clusters are singletons."
    ) -> None:
        self.message = message
        super().__init__(self.message)


_SuperclusterT = TypeVar("_SuperclusterT", bound="Supercluster")


class Supercluster:
    """A set of Cluster instances with cached distances and queued updates.

    Maintains a cached matrix of squared distances between all Cluster exemplars.
    Created for cluster algorithms which pass members around *before* updating
    exemplars, so any changes identified must be staged in each Cluster's queue_add
    and queue_sub sets then applied with Supercluster.process_queues.
    """

    def __init__(self, members: Members) -> None:
        """Create a new Supercluster instance.

        TODO: update Supercluster.__init__ docstring
        :param members: initial members. All are combined into one cluster. Multiple
        arguments allowed.

        TODO: document why clusters are kept in a dictionary, not a set
        """
        self.members = members
        self.clusters = {Cluster(members): None}
        self._states: list[list[tuple[int, ...]]] = []
        self._cache_current_state()

    @classmethod
    def from_stacked_vectors(
        cls: type[_SuperclusterT],
        stacked_vectors: StackedVectors,
        pmatrix: ProximityMatrix | None = None,
    ) -> _SuperclusterT:
        members = Members.from_stacked_vectors(stacked_vectors, pmatrix=pmatrix)
        return cls(members)

    @property
    def n(self) -> int:
        """Return the number of clusters in the Supercluster instance."""
        return len(self.clusters)

    @property
    def cached_n(self) -> int:
        """Return the number of cached states."""
        return len(self._states)

    @property
    def as_cluster(self) -> Cluster:
        """Return the members as a numpy array."""
        ixs = [c.exemplar for c in self.clusters]
        return Cluster(self.members, ixs=ixs)

    @property
    def as_vectors(self) -> FPArray:
        """Return the members as a numpy array, sorted heaviest to lightest."""
        return self.as_stacked_vectors[:, :-1]

    @property
    def as_stacked_vectors(self) -> StackedVectors:
        """Return the members as a numpy array, sorted heaviest to lightest."""
        as_stacked_vectors = self.as_cluster.as_stacked_vectors
        return as_stacked_vectors[np.argsort(as_stacked_vectors[:, -1])][::-1]

    def _get_state(self, n: int) -> list[tuple[int, ...]]:
        """Get the cached state of the Supercluster with n clusters.

        :param n: number of clusters in the state
        :return: the state with n clusters
        :raise IndexError: if the state has not been cached

        This is a 1-based index. The first state is the state with one cluster.
        """
        try:
            return self._states[n - 1]
        except IndexError as e:
            msg = par(
                f"""State {n} has not been cached. The maximum cached state is
                {len(self._states)}."""
            )
            raise IndexError(msg) from e

    def _cache_current_state(self) -> None:
        """Cache the current state of the Supercluster instance."""
        if self.cached_n == self.n - 1:
            self._states.append([tuple(c.ixs) for c in self.clusters])
        elif self.cached_n < self.n:
            msg = "Previous state not cached."
            raise ValueError(msg)


    def get_min_intercluster_proximity(self) -> float:
        """Return the minimum span between clusters."""
        if self.n == 1:
            return 0
        ixs = [c.exemplar for c in self.clusters]
        pmat = self.members.pmatrix_with_inf_diagonal
        return float(np.min(pmat[np.ix_(ixs, ixs)]))

    def split_until(self, predicate: Callable[[Supercluster], bool]) -> None:
        """Split clusters until splitting makes the predicate false.

        :param predicate: function that takes a Supercluster instance and returns a
            boolean. It is critical that this predicate returns true when
            self.n == 1

        Split until the predicate is False, then back up one step. If the predicate
        is False when starting, back up to a state where the predicate is True. If
        all clusters are singletons, and the predicate is still True, the
        Supercluster instance will be left in an "atomized" state.
        """
        while predicate(self):
            try:
                self.split_next_cluster()
            except AllClustersAreSingletonsError:
                return
        while not predicate(self):
            self.set_n(self.n - 1)

    def set_min_proximity(self, min_proximity: float):
        """Split as far as possible while maintaining a minimum inter-cluster span.

        Split until the condition is broken, then back up one step. If the condition
        is broken at the start, back up all the way to a 1-cluster state before
        splitting.
        """

        def predicate(supercluster: Supercluster) -> bool:
            return supercluster.get_min_intercluster_proximity() < min_proximity

        return self.split_until(predicate)

    def _split_to_n_or_more(self, n: int) -> None:
        """Split until there are at least n clusters.

        :param n: minimum number of clusters

        If the Supercluster instance already has n or more clusters, nothing happens.
        """
        self._restore_state_to_at_most_n(n)
        while self.n < n:
            cluster = self._next_to_split
            del self.clusters[cluster]
            self.clusters.update({c: None for c in cluster.split()})
            self._converge()
            self._cache_current_state()

    def _restore_state_to_at_most_n(self, n: int) -> None:
        """Restore the cached state closest to n clusters.

        :param n: maximum number of clusters
        """
        n = min(n, self.n)
        self._restore_state(n)

    def _restore_state(self, n: int) -> None:
        """Restore a previous state of the Supercluster instance.

        :param state: state to restore
        :raise IndexError: if the state has not been cached
        """
        if n == self.n:
            return
        state = self._get_state(n)
        for cluster in tuple(self.clusters):
            ixs = tuple(cluster.ixs)
            if ixs in state:
                state.remove(ixs)
            else:
                del self.clusters[cluster]
        for ixs in state:
            self.clusters[Cluster(self.members, ixs=ixs)] = None

    def set_n(self, n: int) -> None:
        self._split_to_n_or_more(n)
        self._restore_state(n)

    @property
    def _next_to_split(self) -> Cluster:
        """Return the next set of clusters to split.

        :return: set of clusters with sse == max(sse)
        :raise ValueError: if no clusters are available to split

        These will be the clusters (multiple if tie, which should be rare) with the
        highest sse.
        """
        candidate = max(self.clusters, key=lambda c: c.error_metric)
        if candidate.error == 0:
            raise AllClustersAreSingletonsError()
        return candidate

    def split_next_cluster(self):
        """Split the cluster with the highest sum error.

        This sets the state of the Supercluster instance. If the state is already
        >=n, nothing happens.
        """
        self._split_to_n_or_more(self.n + 1)

    def _converge(self, _previous_states: set[tuple[int, ...]] | None = None):
        """Redistribute members between clusters.

        :param _previous_states: set of cluster states that have already been seen.
            For recursion use only

        Recursively redistribute members between clusters until no member can be
        moved to a different cluster to reduce the total error.

        Convergence uses the cluster medoid, not the cluster exemplar. This allows
        the clusters a bit more mobility, so the separation of two heavy,
        nearly-identical clusters is not destiny.

        A record of previous states prevents infinite recursion between a few states.
        It is conceivable that conversion could fail in other cases. The recursion
        limit is set to the Python's recursion limit.
        """
        medoids = [c.medoid for c in self.clusters]

        previous_states = _previous_states or set()
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
            self._converge(previous_states)
