#!/usr/bin/env python3
# last modified: 230117 08:54:10
"""Test functions in triangulate_image.kmedians.py

:author: Shay Hill
:created: 2022-09-16
"""

from typing import Iterable

import matplotlib
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib import pyplot as plt
from PIL import Image

# pyright: reportPrivateUsage=false
import cluster_colors.clusters
from cluster_colors import kmedians
from cluster_colors.clusters import Cluster, Member
from cluster_colors.paths import TEST_DIR
from cluster_colors.stack_vectors import add_weight_axis, stack_vectors
from cluster_colors.type_hints import FPArray

matplotlib.use("WebAgg")


@pytest.fixture
def thin_cluster() -> cluster_colors.clusters.Cluster:
    """A cluster with all members along the (0,0,0) to (1,1,1) line."""
    colors = np.array([[x, x, x, x] for x in range(10)])
    members = cluster_colors.clusters.Member.new_members(colors)
    return cluster_colors.rgb_clusters.Cluster(members)


class TestMemberNewMembers:
    """Test Member.new_members."""

    def test_one_member_per_color(self) -> None:
        """Return 256 members given 256 colors."""
        colors = stack_vectors(np.random.randint(1, 256, (256, 4), dtype=np.uint8))
        members = kmedians.Member.new_members(colors)
        assert len(members) == len(colors)


class TestClusterExemplar:
    """Test triangulate_image._Cluster.exemplar property"""

    def test_exemplar(self) -> None:
        """Return weighted average of member.rgb values."""
        cluster = cluster_colors.clusters.Cluster(
            {Member(np.array([1, 2, 3, 2])), Member(np.array([4, 5, 6, 1]))}
        )
        assert cluster.exemplar == (1, 2, 3)


class TestCluster:
    def test_split(self) -> None:
        """Return 256 clusters given 256 colors."""
        members = Member.new_members(np.random.randint(1, 255, (50, 3)))  # type: ignore
        cluster = Cluster(members)
        clusters = cluster.split()
        clusters = set.union(*(c.split() for c in clusters))  # type: ignore
        # show_clusters(clusters)  # type: ignore


class TestCallers:
    """Test the public functions that use Clusters and Members."""

    # def test_get_cluster_tuple(self) -> None:
    #     """Return exemplar and members for a cluster."""
    #     colors = np.array([[x, x, x, x] for x in range(1, 10)])
    #     members = Member.new_members(colors)
    #     cluster = Cluster(members)
    #     exemplar, members = cluster_rgb._get_cluster_tuple(cluster)
    #     assert exemplar == (7, 7, 7)
    #     assert {tuple(x) for x in members} == {tuple(x) for x in colors}


def show_clusters(clusters: Iterable[Cluster]) -> None:
    """Display clusters as a scatter plot.

    :param clusters: list of sets of (x, y) coordinates

    Make each cluster a different color.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))  # type: ignore
    colors = stack_vectors(colors)  # type: ignore
    for cluster, color in zip(clusters, colors):
        points = cluster.as_array[:, :2]
        x, y = zip(*points)
        plt.scatter(x, y, color=color)  # type: ignore
    plt.show()  # type: ignore


# def test_resets_exemplar_age(self) -> None:
# """Reset exemplar_age to 0 when exemplar is accessed."""
# cluster = cluster_colors.clusters.Cluster(
# {cluster_colors.clusters.Member((1, 2, 3), 2), cluster_colors.clusters.Member((4, 5, 6), 1)}
# )
# cluster.exemplar_age = 1
# _ = cluster.exemplar
# assert cluster.exemplar_age == 0

# # class TestClusterAxis:
# # def test_1d(self, thin_cluster) -> None:
# # """Return the axis of the cluster."""
# # aaa = thin_cluster.axis_of_highest_variance
# # breakpoint()
