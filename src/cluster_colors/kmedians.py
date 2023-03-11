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

from typing import TYPE_CHECKING

from cluster_colors.clusters import Cluster, Clusters, Member

if TYPE_CHECKING:
    from cluster_colors.type_hints import FPArray

_MAX_ITERATIONS = 1000


class KMediansClusters(Clusters):
    """Clusters for kmedians clustering."""

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
        while self._maybe_reassign_members() and iterations < _MAX_ITERATIONS:
            self.process_queues()
            iterations += 1

    def split_above_sse(self, max_sse: float) -> bool:
        """Split while the maximum SSE is above the threshold.

        :param max_sse: the SSE threshold for splitting
        :return: True if a split occurred
        """
        did_split = False
        while self._maybe_split_cluster(max_sse):
            self.converge()
            did_split = True
        return did_split

    def split_to_count(self, count: int) -> bool:
        """Split clusters until len(clusters) == count.

        :param count: the target number of clusters
        :return: True if a split occurred
        """
        did_split = False
        while self._maybe_split_cluster() and len(self) < count:
            self.converge()
            did_split = True
        return did_split

    def split_to_se(self, min_se: float):
        """Split clusters until a split results in two exemplars closer than min_se.

        :param min_se: the minimum squared distance between exemplars

        It is difficult to know when to stop splitting, so this method splits one
        time too many then un-splits the last split.
        """
        prev_state: set[Cluster] | None = None
        while self.spans.valmin() > min_se:
            prev_state = set(self._clusters)
            if self._maybe_split_cluster():
                self.converge()
            else:
                break
        if prev_state is not None and self.spans.valmin() < min_se:
            self.sync(prev_state)

    def merge_below_se(self, min_se: float) -> bool:
        """Merge clusters until the min squared error between exemplars is reached.

        :param min_se: the squared exemplar span for merging
        :return: True if a merge occurred
        """
        did_merge = False
        while self._maybe_merge_cluster(min_se):
            self.converge()
            did_merge = True
        return did_merge

    def merge_to_count(self, count: int) -> bool:
        """Merge clusters until len(clusters) == count.

        :param count: the target number of clusters
        :return: True if a merge occurred
        """
        did_merge = False
        while len(self) > count and self._maybe_merge_cluster():
            self.converge()
            did_merge = True
        return did_merge

    @property
    def _has_clear_winner(self) -> bool:
        """Is one cluster heavier than the rest?.

        :return: True if one cluster is heavier than the rest. Will almost always be
        true.
        """
        if len(self) == 1:
            return True
        weights = [c.w for c in self]
        return weights.count(max(weights)) == 1

    def merge_to_find_winner(self) -> None:
        """Merge clusters until there is a clear winner."""
        while not self._has_clear_winner and self._maybe_merge_cluster():
            self.converge()
