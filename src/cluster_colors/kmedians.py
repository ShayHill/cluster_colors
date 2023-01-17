#!/usr/bin/env python3
# last modified: 230117 17:04:55
"""Cluster stacked vectors.

Designed for divisive clustering, so start with one cluster and divide until some
quality conditions are met.

Will also merge clusters. This is not an inverse of cluster division, so four
divisions then one merge will not result in three divisions. Could be used for
agglomerative clustering all the way up, but is here mostly for tie breaking when the
largest cluster is sought.

I've included some optimizations to make this workable with image colors, but dealing
with very small sets was a priority.

:author: Shay Hill
:created: 2022-09-14
"""

from __future__ import annotations

from copy import deepcopy
from operator import itemgetter
from typing import Iterable

import numpy as np

from cluster_colors.clusters import Cluster, Clusters, Member
from cluster_colors.type_hints import FPArray

_MAX_ITERATIONS = 1000


class _ClusterSplitter:
    """Split the cluster in self.clusters with the highest cost.

    Examine all clusters in self.clusters. Split the cluster with the highest cost
    along the axis of highest variance.
    """

    def __init__(self, clusters: Clusters):
        self.clusters = clusters

    def __call__(self, min_error_to_split: float = 0) -> bool:
        """Split the cluster with the highest SSE. Return True if a split occurred.

        :param min_error_to_split: the cost threshold for splitting
        :return: True if a split occurred

        Could potentially make multiple splits if max_error is a tie, but this is
        unlikely. The good news is that this will be deterministic.
        """
        candidates = [c for c in self.clusters if len(c.members) > 1]
        if not candidates:
            return False
        graded = [(c.sse, c) for c in candidates]
        max_error, cluster = max(graded, key=itemgetter(0))
        if max_error < min_error_to_split:
            return False
        for cluster in (c for g, c in graded if g == max_error):
            self.clusters.remove(cluster)
            self.clusters.add(*cluster.split())
        return True


class _ClusterMerger:
    """Merge clusters in self.clusters with the exemplar span.

    Examine all clusters in self.clusters. Find the two closest clusters, A and B.
    Remove these clusters and add a new cluster with their combined members.
    """

    def __init__(self, clusters: Clusters):
        """Initialize the merger with a set of clusters.

        :param clusters: the clusters to merge
        """
        self.clusters = clusters

    def __call__(self, merge_below_cost: float = np.inf) -> bool:
        """Merge the two clusters with the lowest exemplar span.

        :param merge_below_cost: the cost threshold for merging
        :return: True if a merge occurred.
        """
        if len(self.clusters) < 2:
            return False
        min_cluster_a, min_cluster_b = self.clusters.spans.keymin()
        min_cost = self.clusters.spans(min_cluster_a, min_cluster_b)
        if min_cost > merge_below_cost:
            return False
        combined_members = min_cluster_a.members | min_cluster_b.members
        self.clusters.remove(min_cluster_a, min_cluster_b)
        self.clusters.add(Cluster(combined_members))
        return True


class _ClusterReassigner:
    """Reassign members to the closest cluster exemplar."""

    def __init__(self, clusters: Clusters) -> None:
        """Initialize the reassigner with a set of clusters.

        :param clusters: the clusters whose members may be reassigned
        """
        self.clusters = clusters

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
        spans = self.clusters.spans
        return {x for x in others if spans(cluster, x) / 4 < max_se}

    def _offer_members(self, cluster: Cluster) -> None:
        """Look for another cluster with lower cost for members of input cluster.

        :param cluster: the cluster offering its members to other clusters
        :effect: moves members between clusters
        """
        others = self._get_others(cluster)
        if not others:
            return

        safe_cost = self.clusters.spans.min(cluster) / 4
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

    def __call__(self) -> bool:
        """Pass members between clusters and update exemplars.

        :return: True if any changes were made
        """
        if len(self.clusters) < 2:
            return False
        if all(x.exemplar_age > 0 for x in self.clusters):
            return False
        for cluster in self.clusters:
            self._offer_members(cluster)
        return True


class KMediansClusters(Clusters):
    """Clusters for kmedians clustering."""

    def __init__(self, clusters: Iterable[Cluster]) -> None:
        super().__init__(clusters)
        self.splitter = _ClusterSplitter(self)
        self.merger = _ClusterMerger(self)
        self.reassigner = _ClusterReassigner(self)

    @classmethod
    def from_stacked_vectors(cls, stacked_vectors: FPArray) -> KMediansClusters:
        """Create a KMediansClusters from an iterable of colors.

        :param stacked_vectors: An iterable of vectors with a weight axis
        :return: A KMediansClusters instance
        """
        return cls({Cluster(Member.new_members(stacked_vectors))})

    def converge(self) -> None:
        """Reassign members until no changes occur."""
        iterations = 0
        # if any(x.queue_add for x in self.clusters):
        # self.clusters.process_queues()
        while self.reassigner() and iterations < _MAX_ITERATIONS:
            self.process_queues()
            iterations += 1

    def split_above_sse(self, max_sse: float) -> bool:
        """Split while the maximum SSE is above the threshold.

        :param max_sse: the SSE threshold for splitting
        :return: True if a split occurred
        """
        did_split = False
        while self.splitter(max_sse):
            self.converge()
            did_split = True
        return did_split

    def split_to_count(self, count: int) -> bool:
        """Split clusters until len(clusters) == count.

        :param count: the target number of clusters
        :return: True if a split occurred
        """
        did_split = False
        while self.splitter() and len(self) < count:
            self.converge()
            did_split = True
        return did_split

    def split_to_se(self, min_se: float) -> bool:
        """Split clusters until a split results in two exemplars closer than min_se.

        :param min_se: the minimum squared distance between exemplars
        :return: True if a split occurred
        """
        dummy = deepcopy(self)
        while dummy.spans.valmin() > min_se and dummy.splitter():
            dummy.converge()
        if dummy.spans.valmin() > min_se:
            clusters_found = len(dummy)
        else:
            clusters_found = len(dummy) - 1
        return self.split_to_count(clusters_found)

    def merge_below_se(self, min_se: float) -> bool:
        """Merge clusters until the min squared error between exemplars is reached.

        :param min_se: the squared exemplar span for merging
        :return: True if a merge occurred
        """
        did_merge = False
        while self.merger(min_se):
            self.converge()
            did_merge = True
        return did_merge

    def merge_to_count(self, count: int) -> bool:
        """Merge clusters until len(clusters) == count.

        :param count: the target number of clusters
        :return: True if a merge occurred
        """
        did_merge = False
        while len(self) > count and self.merger():
            self.converge()
            did_merge = True
        return did_merge

    @property
    def _has_clear_winner(self) -> bool:
        """Is one cluster heavier than the rest?

        :return: True if one cluster is heavier than the rest. Will almost always be
        true.
        """
        if len(self) == 1:
            return True
        weights = [c.w for c in self]
        return weights.count(max(weights)) == 1

    def merge_to_find_winner(self) -> None:
        """Merge clusters until there is a clear winner."""
        while not self._has_clear_winner and self.merger():
            self.converge()
