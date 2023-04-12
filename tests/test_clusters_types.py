"""Test individual methods in the base Member, Cluster, and Clusters classes.

:author: Shay Hill
:created: 2023-04-12
"""

from cluster_colors.clusters import Cluster, Clusters, Member
import numpy as np

class TestClusters:
    def test_as_one_cluster(self):
        """Test the as_one_cluster method."""
        one_cluster = Cluster.from_stacked_vectors(np.array([[1, 2, 3], [4, 5, 6]]))
        clusters = Clusters([one_cluster])
        assert clusters.as_one_cluster == one_cluster

