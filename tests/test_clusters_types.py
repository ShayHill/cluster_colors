"""Test individual methods in the base Member, Cluster, and Supercluster classes.

:author: Shay Hill
:created: 2023-04-12
"""

from cluster_colors.cluster_supercluster import Cluster, DivisiveSupercluster
from cluster_colors.cluster_members import Members
import numpy as np


class TestSupercluster:
    def test_as_one_cluster(self):
        """Test the as_one_cluster method."""
        members = Members.from_stacked_vectors(np.random.rand(6, 4))
        clusters = DivisiveSupercluster(members)
        assert len(clusters.get_as_stacked_vectors()) == 1
