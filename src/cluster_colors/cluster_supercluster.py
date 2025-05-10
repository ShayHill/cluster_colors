"""The members, clusters, and groups of clusters.

SuperclusterBase is an abstract class that can split or merge clusters.

SuperclusterBase caches states (lists of cluster indices, each defining a Cluster,
given a number of clusters) when splitting or merging.

A supercluster that starts as one large cluster will cache states as that cluster
and its descendants are split, and merging from any state in that cluster will be
loading a previouly cached state.

Similarly, a supercluster that starts as singletons will cache states as those
singletons and their descendants are merged, and splitting from any state in that
cluster will be loading a previously cached state.

The result of this is that a supercluster started as one large cluster will never
merge (only split and un-split) and a supercluster started as singletons will
never split (only merge and un-merge). The only thing required to make this a
divisive or agglomerative class is to implement the `_initialize_clusters` method
to return either a single cluster or a cluster for each member.

:author: Shay Hill
:created: 2023-01-17
"""

from __future__ import annotations

import itertools as it
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from paragraphs import par

from cluster_colors import exceptions
from cluster_colors.cluster_cluster import Cluster
from cluster_colors.cluster_members import Members

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cluster_colors.type_hints import (
        CenterName,
        CentroidName,
        FPArray,
        IndicesLike,
        ProximityMatrix,
        QualityMetric,
        Vectors,
        VectorsLike,
    )

from typing import cast

_CachedState = tuple[tuple[int, ...], ...]
_SuperclusterT = TypeVar("_SuperclusterT", bound="SuperclusterBase")
_GetterT = TypeVar("_GetterT", bound="Callable[[SuperclusterBase], float]")
_SetterT = TypeVar("_SetterT", bound="Callable[[SuperclusterBase, float], None]")


def check_empty_supercluster_get(method: _GetterT) -> _GetterT:
    """Decorator to raise EmptySuperclusterError if self.n is 0.

    Raise this exception for methods that are invalid on an empty supercluster.
    """

    def wrapper(self: SuperclusterBase) -> float:
        m_name = method.__name__
        if self.n == 0:
            msg = par(
                f"""The supercluster is empty. Attempt to execute method '{m_name}'
                failed."""
            )
            raise exceptions.EmptySuperclusterError(msg)
        return method(self)

    return cast("_GetterT", wrapper)


def check_empty_supercluster_set(method: _SetterT) -> _SetterT:
    """Decorator to raise EmptySuperclusterError if self.n is 0.

    Raise this exception for methods that are invalid on an empty supercluster.
    """

    def wrapper(self: SuperclusterBase, val: float):
        m_name = method.__name__
        if self.n == 0:
            msg = par(
                f"""The supercluster is empty. Attempt to execute method '{m_name}'
                failed."""
            )
            raise exceptions.EmptySuperclusterError(msg)
        return method(self, val)

    return cast("_SetterT", wrapper)


class SuperclusterBase:
    """A list of Cluster instances.

    quality_metric: The property used to select which cluster to split. This is
        meaningles for agglomerative clustering. There is no such class attribute for
        merge quality, because I have only implemented one merge-quality metric:
        span.
        Literal["sum_error", "max_error", "avg_error", "span"]

    quality_centroid: The property used as a centroid when computing the quality
        metric.
        Literal["weighted_medoid", "unweighted_medoid"]

    assignment_centroid: The property used as a centroid when assigning members to a
        cluster. This may be different from the quality centroid, as SuperclusterBase
        deviates from k-medoids by splitting a cluster by median cut instead of by
        randomly selecting exemplars from the cluster.
        Literal["weighted_medoid", "unweighted_medoid"]

    clustering_method: The method used to split or merge clusters.
        Literal["divisive", "agglomerative"]
    """

    quality_metric: QualityMetric = "sum_error"
    quality_centroid: CentroidName = "weighted_medoid"
    assignment_centroid: CentroidName = "weighted_medoid"
    clustering_method: Literal["divisive", "agglomerative"] = "divisive"

    def __init__(self, members: Members, ixs: IndicesLike | None = None) -> None:
        """Create a new Supercluster instance.

        :param members: Members instance
        :param ixs: optional list of indices to use from the Members instance. This
            can be used to create a subset of clusters from another Subcluster
            instance using the same Members instance.
        """
        self.members = members
        self.members = members
        if ixs is None:
            self.ixs = np.arange(len(self.members))
        else:
            self.ixs = np.array(sorted(map(int, ixs)), dtype=np.intp)

        class _Cluster(Cluster):
            quality_metric = self.quality_metric
            quality_centroid = self.quality_centroid

        self.cluster_type = _Cluster

        self.clusters: list[Cluster] = self._initialize_clusters()
        self._cached_states: list[_CachedState] = []
        self._cache_current_state()

    def _initialize_clusters(self) -> list[Cluster]:
        """Create clusters from the members."""
        if len(self.ixs) == 0:
            return []
        match self.clustering_method:
            case "divisive":
                return [self.cluster_type(self.members, self.ixs)]
            case "agglomerative":
                return [self.cluster_type(self.members, [i]) for i in self.ixs]
        msg = par(
            f"""SuperclusterBase class attribute `clustering_type` must be 'divisive'
            or 'agglomerative', not {self.clustering_type}."""
        )
        raise ValueError(msg)

    # ===========================================================================
    #   lookups
    # ===========================================================================

    def find_member(self, member_idx: int) -> int:
        """Return an indes to the cluster that contains member_idx."""
        return next(i for i, c in enumerate(self.clusters) if member_idx in c.ixs)

    # ===========================================================================
    #   constructors
    # ===========================================================================

    @classmethod
    def from_vectors(
        cls: type[_SuperclusterT],
        vectors: VectorsLike,
        pmatrix: ProximityMatrix | None = None,
    ) -> _SuperclusterT:
        """Create a SuperclusterBase instance from a sequence or array of colors.

        :param vectors: An iterable of vectors
            [(r0, g0, b0), (r1, g1, b1), ...]
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: A SuperclusterBase instance with members
            {Member([r0, g0, b0, 1]), Member([r1, g1, b1, 1]), ...}
        """
        members = Members.from_vectors(vectors, pmatrix=pmatrix)
        return cls(members)

    @classmethod
    def from_stacked_vectors(
        cls: type[_SuperclusterT],
        stacked_vectors: VectorsLike,
        pmatrix: ProximityMatrix | None = None,
    ) -> _SuperclusterT:
        """Create a Cluster instance from an array of colors with a weight axis.

        :param stacked_vectors: An iterable of vectors with a weight axis
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
            [(r0, g0, b0, w0), (r1, g1, b1, w1), ...]
        :return: A SuperclusterBase instance with members
            {Member([r0, g0, b0, w0]), Member([r1, g1, b1, w1]), ...}
        """
        members = Members.from_stacked_vectors(stacked_vectors, pmatrix=pmatrix)
        return cls(members)

    # ===============================================================================
    #   filtered copies
    # ===============================================================================

    def _fill_inc_or_exc(
        self,
        cluster_ixs: Iterable[int],
        member_cluster_ixs: Iterable[int],
        member_ixs: Iterable[int],
    ) -> set[int]:
        """Return a set of member indices to include or exclude.

        :param cluster_ixs: indices of clusters to include or exclude
        :param member_cluster_ixs: indices of members to include or exclude. For each
            given member, the cluster that contains that member will be included or
            excluded.
        :param member_ixs: indices of members to include or exclude.
        :return: set of member indices
        """
        clusters = set(cluster_ixs)
        clusters.update(self.find_member(m_i) for m_i in member_cluster_ixs)
        members = set(it.chain(*[self.clusters[c_i].ixs for c_i in clusters]))
        return members | set(member_ixs)

    def copy(
        self: _SuperclusterT,
        *,
        inc_clusters: Iterable[int] = (),
        inc_member_clusters: Iterable[int] = (),
        inc_members: Iterable[int] = (),
        exc_clusters: Iterable[int] = (),
        exc_member_clusters: Iterable[int] = (),
        exc_members: Iterable[int] = (),
        reindex: bool = False,
    ) -> _SuperclusterT:
        """Create a SuperclusterBase instance from a subset of clusters.

        :param inc_clusters: indices of clusters to include in the subset
        :param inc_member_clusters: indices of members to include in the subset. For
            each given member, the cluster that contains that member will be
            included.
        :param inc_members: indices of members to include in the subset.
        :param exc_clusters: indices of clusters to exclude from the subset
        :param exc_member_clusters: indices of members to exclude from the
            subset. For each given member, the cluster that contains that member will
            be excluded.
        :param exc_members: indices of members to exclude from the subset.
        :param reindex: Optionally reindex the members of the subset.
        :return: a new SuperclusterBase instance with members from the subset

        If no arguments are given for `inc_`, excluded members will be excluded from
        the entire supercluster. If any `inc_` arguments are passed, only members
        explicitly included will be included, and the `exc_` arguments will be
        excluded from that `inc_` subset.

        If reindex=True, reindex the members of the subset. Filter members.vectors,
        members.weights, and members.pmatrix. Self.ixs will be range(len(subset)).
        This will preserve the calculations in the pmatrix, but will otherwise be as
        if the superset had never existed.

        If reindex=False (the default), do not reindex the members of the subset.
        The number of member indices in the superset will be <= the length of the
        members.vectors. Self.ixs will be sequential, but will have gaps. This will
        carry around some extra data, but indices to the parent supercluster members
        will be preserved.
        """
        include = self._fill_inc_or_exc(inc_clusters, inc_member_clusters, inc_members)
        exclude = self._fill_inc_or_exc(exc_clusters, exc_member_clusters, exc_members)

        if include:
            ixs = sorted(include - exclude)
        else:
            ixs = sorted(set(self.ixs) - exclude)

        if reindex is False:
            return self.__class__(self.members, ixs)

        subset_vectors = self.members.vectors[ixs]
        subset_weights = self.members.weights[ixs]
        subset_pmatrix = self.members.pmatrix[np.ix_(ixs, ixs)]
        subset_members = Members(
            subset_vectors, weights=subset_weights, pmatrix=subset_pmatrix
        )
        return self.__class__(subset_members)

    # ===========================================================================
    #   properties
    # ===========================================================================

    @property
    def n(self) -> int:
        """Return the number of clusters in the Supercluster instance."""
        return len(self.clusters)

    def get_as_stacked_vectors(self, which_center: CenterName | None = None) -> Vectors:
        """Return the members as a numpy array, sorted heaviest to lightest.

        :param which_center: optionally specify a cluster center attribute. Choices
            are 'weighted_median', 'weighted_medoid', or 'unweighted_medoid'. Default
            is 'weighted_median'.
        :return as_stacked_vectors: members as a numpy array (n, m+1) with the last
            column as the weight.
        """
        if self.n == 0:
            return self.members.stacked_vectors[:0]
        as_stacked_vectors = np.array(
            [c.get_as_stacked_vector(which_center) for c in self.clusters]
        )
        return as_stacked_vectors[np.argsort(as_stacked_vectors[:, -1])][::-1]

    def get_as_vectors(self, which_center: CenterName | None = None) -> FPArray:
        """Return the members as a numpy array, sorted heaviest to lightest.

        :param which_center: optionally specify a cluster center attribute. Choices
            are 'weighted_median', 'weighted_medoid', or 'unweighted_medoid'. Default
            is 'weighted_median'.
        """
        if self.n == 0:
            return self.members.vectors[:0]
        return self.get_as_stacked_vectors(which_center)[:, :-1]

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
        current_state = tuple(tuple(c.ixs) for c in self.clusters)
        new_state = [x for x in state if x not in current_state]
        self.clusters = [c for c in self.clusters if tuple(c.ixs) in state]
        self.clusters.extend([self.cluster_type(self.members, x) for x in new_state])

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

        Avoid picking singletons, which can tie for max_error == 0 with larger
        clusters if the larger clusters have 0-weight members.

        This function should never be called if all clusters are singletons, a
        FailedToSplitError should be raised before that happens.
        """
        candidates = (c for c in self.clusters if len(c.ixs) > 1)
        return max(candidates, key=lambda c: c.error)

    def _get_next_to_merge(self) -> tuple[Cluster, Cluster]:
        """Return the next set of clusters to merge.

        :return: set of clusters with sse == min(sse)
        :raise ValueError: if no clusters are available to merge
        """
        if len(self.clusters) == 1:
            raise exceptions.FailedToMergeError
        pairs = it.combinations(self.clusters, 2)
        return min(pairs, key=lambda p: p[0].get_merge_span(p[1]))

    # ===========================================================================
    #   perform splits and merges
    # ===========================================================================

    def _split_to_n(self, n: int) -> None:
        """Split or restore the Supercluster instance to n clusters.

        :param n: number of clusters
        """
        self._restore_state_as_close_as_possible_to_n(n)

        while self.n < n:
            if self.n == len(self.ixs):
                raise exceptions.FailedToSplitError
            cluster = self._get_next_to_split()
            self.clusters.remove(cluster)
            self.clusters.extend(cluster.split())
            self._reassign()
            self._cache_current_state()

    def _merge_to_n(self, n: int) -> None:
        """Merge or restore the Supercluster instance to n clusters.

        :param n: number of clusters
        """
        self._restore_state_as_close_as_possible_to_n(n)
        while self.n > n:
            if self.n == 1:
                raise exceptions.FailedToMergeError
            pair_to_merge = self._get_next_to_merge()
            merged_ixs = np.concatenate([x.ixs for x in pair_to_merge])
            merged = self.cluster_type(self.members, merged_ixs)
            self.clusters = [c for c in self.clusters if c not in pair_to_merge]
            self.clusters.append(merged)

    # ===========================================================================
    #   common public methods
    # ===========================================================================

    def set_n(self, n: int) -> None:
        """Set the number of clusters in the Supercluster instance.

        :param n: number of clusters
        """
        if self.n == 0:
            msg = "No clusters to merge or split."
            raise exceptions.EmptySuperclusterError(msg)
        self._split_to_n(n)
        self._merge_to_n(n)

    def split(self):
        """Split the cluster with the highest sum error.

        This sets the state of the Supercluster instance. If the state is already
        >=n, nothing happens.
        """
        if self.n == 0:
            msg = "No clusters to split."
            raise exceptions.FailedToSplitError(msg)
        self._split_to_n(self.n + 1)

    def merge(self):
        """Merge the two clusters with the lowest sum error.

        This sets the state of the Supercluster instance. If the state is already
        <=n, nothing happens.
        """
        if self.n == 0:
            msg = "No clusters to merge."
            raise exceptions.FailedToMergeError(msg)
        self._merge_to_n(self.n - 1)

    # ===========================================================================
    #   split or merge to satisfy a condition
    #
    #   For every condition defined here, the condition will be satisfied when a
    #   one-cluster state is reached (if not before), but other conditions may be
    #   passed in that to do satisfy this. In those instances, the splitting or
    #   merging will silenty give up when the minimum or maximum number of clusters
    #   is reached.
    # ===========================================================================

    def split_till_true(self, predicate: Callable[[SuperclusterBase], bool]) -> None:
        """Split clusters till predicate is True.

        :param predicate: function that takes a SuperclusterBase and returns a
            boolean. Presumably, if the predicate is True, a quality metric is good.
            If False, that metric is bad.

        If the predicate is True at the current state, merge till it is False, then
        split back to the first True. This will put the SuperclusterInstance into a
        state where the predicate is True, but one merge would make it False.

        This is for predicates that improve as the number of clusters increases.
        """
        with suppress(exceptions.FailedToMergeError):
            while predicate(self):
                self.merge()
        with suppress(exceptions.FailedToSplitError):
            while not predicate(self):
                self.split()

    @check_empty_supercluster_get
    def get_max_sum_error(self) -> float:
        """Return the maximum sum of errors of any cluster."""
        return max(c.span for c in self.clusters)

    @check_empty_supercluster_set
    def set_max_sum_error(self, max_sum_error: float):
        """Split as far as necessary to get below the threshold.

        :param min_proximity: maximum sum of errors of any cluster
        """

        def predicate(supercluster: SuperclusterBase) -> bool:
            return supercluster.get_max_sum_error() <= max_sum_error

        self.split_till_true(predicate)

    @check_empty_supercluster_get
    def get_max_span(self) -> float:
        """Return the minimum maximum cost of any cluster."""
        return max(c.span for c in self.clusters)

    @check_empty_supercluster_set
    def set_max_span(self, max_span: float):
        """Split as far as necessary to get below the threshold.

        :param min_max_error: maximum span of any cluster
        """

        def predicate(supercluster: SuperclusterBase) -> bool:
            return supercluster.get_max_span() <= max_span

        self.split_till_true(predicate)

    @check_empty_supercluster_get
    def get_max_max_error(self) -> float:
        """Return the maximum max_error of any cluster."""
        return max(c.max_error for c in self.clusters)

    @check_empty_supercluster_set
    def set_max_max_error(self, max_max_error: float):
        """Split as far as necessary to get below the threshold.

        :param min_max_error: maximum max_error of any cluster
        """

        def predicate(supercluster: SuperclusterBase) -> bool:
            return supercluster.get_max_max_error() <= max_max_error

        self.split_till_true(predicate)

    @check_empty_supercluster_get
    def get_max_avg_error(self) -> float:
        """Return the maximum avg_error of any cluster."""
        return max(c.avg_error for c in self.clusters)

    @check_empty_supercluster_set
    def set_max_avg_error(self, max_avg_error: float):
        """Split as far as necessary to get below the threshold.

        :param max_avg_error: maximum avg_error of any cluster
        """

        def predicate(supercluster: SuperclusterBase) -> bool:
            return supercluster.get_max_avg_error() <= max_avg_error

        self.split_till_true(predicate)

    # ===========================================================================
    #   the reassignment step for divisive clustering
    # ===========================================================================

    def _reassign(self, _previous_medoids: set[tuple[int, ...]] | None = None):
        """Reassign members based on proximity to cluster medoids.

        :param _previous_medoids: set of cluster medoids that have already been seen.
            For recursion use only

        Recursively redistribute members between clusters until no member can be
        moved to a different cluster to reduce the total error.

        A record of previous states prevents infinite recursion between a few states.
        It is conceivable that conversion could fail in other cases. The recursion
        limit is set to the Python's recursion limit.

        This will only ever be called for divisive clustering.
        """
        medoids = np.array([c.unweighted_medoid for c in self.clusters])
        previous_states = _previous_medoids or set()
        state = tuple(sorted(medoids))
        if state in previous_states:
            return
        previous_states.add(state)

        which_medoid = np.argmin(
            self.members.pmatrix[np.ix_(medoids, self.ixs)], axis=0
        )

        for i, cluster in enumerate(tuple(self.clusters)):
            new_where = np.argwhere(which_medoid == i)
            new_ixs = self.ixs[new_where.flatten()]
            new = list(map(int, new_ixs))
            if new != list(cluster.ixs):
                self.clusters.remove(cluster)
                self.clusters.append(self.cluster_type(self.members, new))

        with suppress(RecursionError):
            self._reassign(previous_states)


class DivisiveSupercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric: QualityMetric = "avg_error"
    quality_centroid: CentroidName = "weighted_medoid"
    assignment_centroid: CentroidName = "weighted_medoid"
    clustering_method: Literal["divisive", "agglomerative"] = "divisive"


class AgglomerativeSupercluster(SuperclusterBase):
    """A SuperclusterBase that uses agglomerative clustering."""

    quality_metric: QualityMetric = "avg_error"
    quality_centroid: CentroidName = "weighted_medoid"
    assignment_centroid: CentroidName = "weighted_medoid"
    clustering_method: Literal["divisive", "agglomerative"] = "divisive"
