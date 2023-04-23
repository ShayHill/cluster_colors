"""Test individual methods in the base Member, Cluster, and Supercluster classes.

:author: Shay Hill
:created: 2023-04-12
"""

from cluster_colors.clusters import Cluster, Supercluster
import numpy as np

class TestSupercluster:
    def test_as_one_cluster(self):
        """Test the as_one_cluster method."""
        one_cluster = Cluster.from_stacked_vectors(np.array([[1, 2, 3, 1], [4, 5, 6, 1]]))
        two_cluster = Cluster.from_stacked_vectors(np.array([[7, 8, 9, 1], [1, 3, 5, 1]]))
        clusters = Supercluster(one_cluster, two_cluster)
        assert len(clusters.as_cluster.members) == 4

