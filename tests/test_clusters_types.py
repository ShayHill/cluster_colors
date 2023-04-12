"""Test individual methods in the base Member, Cluster, and Clusters classes.

:author: Shay Hill
:created: 2023-04-12
"""

from cluster_colors.clusters import Cluster, Clusters
import numpy as np

class TestClusters:
    def test_as_one_cluster(self):
        """Test the as_one_cluster method."""
        one_cluster = Cluster.from_stacked_vectors(np.array([[1, 2, 3, 1], [4, 5, 6, 1]]))
        two_cluster = Cluster.from_stacked_vectors(np.array([[7, 8, 9, 1], [1, 3, 5, 1]]))
        clusters = Clusters([one_cluster])
        clusters.add(two_cluster)
        assert len(clusters.as_one_cluster.members) == 4

