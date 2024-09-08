"""Test individual methods in the base Member, Cluster, and Supercluster classes.

:author: Shay Hill
:created: 2023-04-12
"""

from cluster_colors.clusters import Cluster, Supercluster
from cluster_colors.cluster_member import Members
import numpy as np


class TestSupercluster:
    def test_as_one_cluster(self):
        """Test the as_one_cluster method."""
        members = Members.from_stacked_vectors(np.random.rand(6, 4))
        clusters = Supercluster(members)
        assert len(clusters.as_cluster) == 1
