"""Test functions in triangulate_image.kmedians.py

:author: Shay Hill
:created: 2022-09-16
"""

import numpy as np
from matplotlib import pyplot as plt

# pyright: reportPrivateUsage=false
from cluster_colors.cluster_members import Members
from cluster_colors.cluster_supercluster import (
    AgglomerativeSupercluster,
    Cluster,
    DivisiveSupercluster,
    SuperclusterBase,
)


class TestClusterExemplar:
    """Test triangulate_image._Cluster.exemplar property"""

    def test_weighted_medoid(self) -> None:
        """Return weighted average of member.rgb values."""
        vectors = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        members = Members.from_stacked_vectors(vectors)
        cluster = Cluster(members, [0, 2])
        np.testing.assert_array_equal(cluster.weighted_medoid, 2)

    def test_unweighted_medoid(self) -> None:
        """Return the member with lowest cost, ignoring weights."""
        vectors = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        members = Members.from_stacked_vectors(vectors)
        cluster = Cluster(members, [0, 1, 2])
        np.testing.assert_array_equal(cluster.unweighted_medoid, 1)

    def test_weighted_median(self) -> None:
        """Return the weighted median of the members."""
        vectors = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3]])
        members = Members.from_stacked_vectors(vectors)
        cluster = Cluster(members, [0, 1, 2])
        np.testing.assert_array_equal(cluster.weighted_median, (5.5, 6.5, 7.5))


class TestCluster:

    def test_split(self) -> None:
        """Return 256 clusters given 256 colors."""
        vectors = np.random.rand(50, 4) * 255
        members = Members.from_stacked_vectors(vectors)
        cluster = Cluster(members, range(50))
        child_a, child_b = cluster.split()
        assert set(child_a.ixs) | set(child_b.ixs) == set(cluster.ixs)

    def test_plot_2d_clusters(self) -> None:
        """Display clusters as a scatter plot."""
        vectors = np.random.rand(150, 3) * 255
        supercluster = DivisiveSupercluster.from_stacked_vectors(vectors)
        for _ in range(4):
            supercluster.split()
        plot_2d_clusters(supercluster)

    def test_plot_2d_clusters_agglomerative(self) -> None:
        """Display clusters as a scatter plot.

        This test occasionally fails with a _tkinter.TclError. I'm not sure why, but
        re-running with the same output works, so it's not something I'm going to try
        to fix. So, I catch the error, print it, and move on.
        """
        vectors = np.random.rand(150, 3) * 255
        supercluster = AgglomerativeSupercluster.from_stacked_vectors(vectors)
        supercluster.set_n(5)

        try:
            plot_2d_clusters(supercluster)
        except Exception as e:
            print(e)


def plot_2d_clusters(supercluster: SuperclusterBase) -> None:
    """Display clusters as a scatter plot.

    :param supercluster: list of sets of (x, y) coordinates

    Make each cluster a different color.
    """
    clusters = supercluster.clusters
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    # colors2 = stack_vectors(colors)

    clusters = sorted(clusters, key=lambda c: c.weight)[::-1]
    colors = np.zeros((len(clusters), 4))
    colors[:, 3] = 1
    colors[:, :2] = supercluster.get_as_vectors() / 255
    colors[:, 2] = 1 - np.max(colors[:, :2], axis=1)

    for cluster, color in zip(clusters, colors):
        points = cluster.members.vectors[cluster.ixs]
        exemplar = cluster.get_as_vector("unweighted_medoid")
        plt.scatter([exemplar[0]], [exemplar[1]], color="black", s=100)  # type: ignore
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.scatter(xs, ys, color=color)  # type: ignore
    plt.show()  # type: ignore
