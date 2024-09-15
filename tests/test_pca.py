"""Test module pca.py

:author: Shay Hill
:created: 2023-03-14
"""

# pyright: reportPrivateUsage=false

import math

import numpy as np

from cluster_colors import cluster_supercluster


class TestPCA:
    def test_linear(self):
        """When all points are on a line, axis points along line."""
        line = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [14, 14, 1]])
        cluster_ = cluster_supercluster.Cluster.from_stacked_vectors(line)
        x, y = cluster_._direction_of_highest_variance
        assert math.isclose(x, y)
        assert x != 0

    def test_square(self):
        """When all points form a square, axis points up or across."""
        square = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1]])
        cluster_ = cluster_supercluster.Cluster.from_stacked_vectors(square)
        x, y = cluster_._direction_of_highest_variance
        assert x == 0 or y == 0
        assert x == 1 or y == 1

    def test_wide_rect(self):
        """When all points form a long rectangle, axis points across."""
        wide_rect = np.array([[1, 1, 1], [5, 1, 1], [5, 2, 1], [1, 2, 1]])
        cluster_ = cluster_supercluster.Cluster.from_stacked_vectors(wide_rect)
        np.testing.assert_allclose(cluster_._direction_of_highest_variance, (1, 0))

    def test_tall_rect(self):
        """When all points form a tall rectangle, axis points up."""
        tall_rect = np.array([[1, 1, 1], [2, 1, 1], [2, 5, 1], [1, 5, 1]])
        cluster_ = cluster_supercluster.Cluster.from_stacked_vectors(tall_rect)
        np.testing.assert_allclose(cluster_._direction_of_highest_variance, (0, 1))
