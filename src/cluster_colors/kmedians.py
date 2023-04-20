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

from cluster_colors.clusters import Cluster, Supercluster

_MAX_ITERATIONS = 1000


class KMedSupercluster(Supercluster):
    """Clusters for kmedians clustering."""

    def converge(self) -> None:
        """Reassign members until no changes occur."""
        iterations = 0
        # if any(x.queue_add for x in self.clusters):
        while self._maybe_reassign_members() and iterations < _MAX_ITERATIONS:
            self.process_queues()
            iterations += 1

    def _split_cluster(self, cluster: Cluster):
        """Split one cluster."""
        self.remove(cluster)
        self.add(*cluster.split())
        self.converge()
