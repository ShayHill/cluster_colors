"""Test functions in triangulate_image.kmedians.py

:author: Shay Hill
:created: 2022-09-16
"""

from typing import Iterable

import numpy as np
import pytest
from matplotlib import pyplot as plt  # type: ignore

# pyright: reportPrivateUsage=false
import cluster_colors.clusters
from cluster_colors.clusters import Cluster, Member
from cluster_colors.stack_vectors import  stack_vectors



@pytest.fixture
def thin_cluster() -> Cluster:
    """A cluster with all members along the (0,0,0) to (1,1,1) line."""
    colors = np.array([[x, x, x, x] for x in range(10)])
    members = cluster_colors.clusters.Member.new_members(colors)
    return Cluster(members)


class TestMemberNewMembers:
    """Test Member.new_members."""

    def test_one_member_per_color(self) -> None:
        """Return 256 members given 256 colors."""
        colors = stack_vectors(np.random.randint(1, 256, (256, 4), dtype=np.uint8))
        members = Member.new_members(colors)
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
        show_clusters(clusters)  # type: ignore


def show_clusters(clusters: Iterable[Cluster]) -> None:
    """Display clusters as a scatter plot.

    :param clusters: list of sets of (x, y) coordinates

    Make each cluster a different color.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))  # type: ignore
    colors = stack_vectors(colors)  # type: ignore
    for cluster, color in zip(clusters, colors):
        points = cluster.as_array[:, :2]
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.scatter(xs, ys, color=color)  # type: ignore
    plt.show()  # type: ignore


