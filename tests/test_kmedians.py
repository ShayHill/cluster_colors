"""Test methods in KMedDivisiveSupercluster

:author: Shay Hill
/created: 2023-03-14
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

import itertools as it
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from cluster_colors.cluster_supercluster import (
    AgglomerativeSupercluster,
    DivisiveSupercluster,
)
from cluster_colors.exceptions import FailedToSplitError
from cluster_colors.vector_stacker import stack_vectors

ColorsArray = Annotated[npt.NDArray[np.floating], (-1, 3)]


@pytest.fixture(
    scope="function",
    params=[np.random.randint(0, 255, (100, 4), dtype=np.uint8) for _ in range(10)],
)
def colors(request: pytest.FixtureRequest) -> ColorsArray:
    return stack_vectors(request.param)


class TestKMedians:
    def test_get_rsorted_clusters(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(16)
        _ = clusters.get_as_stacked_vectors()

    def test_split_to_n(self, colors: ColorsArray):
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(10)
        assert clusters.get_as_stacked_vectors().shape == (10, 4)

    def test_get_rsorted_exemplars(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(16)
        assert clusters.get_as_vectors().shape == (16, 3)

    def test_merge_to_n(self, colors: ColorsArray):
        clusters = AgglomerativeSupercluster.from_stacked_vectors(colors[:24])
        clusters.set_n(10)
        assert clusters.get_as_stacked_vectors().shape == (10, 4)


class TestSubset:
    def test_subset_init_reindex_false(self, colors: ColorsArray):
        """Vectors and pmatrix are unchanged."""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters_m = len(clusters.ixs)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7))
        subset_ixs = sorted(it.chain(*[c.ixs for c in clusters.clusters[:4]]))
        assert subset.members.vectors.shape == (clusters_m, 3)
        assert subset.members.weights.shape == (clusters_m,)
        assert subset.members.pmatrix.shape == (clusters_m, clusters_m)
        np.testing.assert_array_equal(subset_ixs, subset.ixs)

    def test_subset_init_reindex_true(self, colors: ColorsArray):
        """Vectors and pmatrix are filtered. Indices are sequential."""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7), reindex=True)
        subset_ixs = sorted(it.chain(*[c.ixs for c in clusters.clusters[:4]]))
        subset_cnt = len(subset_ixs)
        assert subset.members.vectors.shape == (subset_cnt, 3)
        assert subset.members.weights.shape == (subset_cnt,)
        assert subset.members.pmatrix.shape == (subset_cnt, subset_cnt)
        np.testing.assert_array_equal(subset.ixs, range(subset_cnt))

    def test_subset_split_reindex_false(self, colors: ColorsArray):
        """Nothing should change"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters_m = len(clusters.ixs)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7))
        subset.set_n(4)
        subset_ixs = sorted(it.chain(*[c.ixs for c in clusters.clusters[:4]]))
        assert subset.members.vectors.shape == (clusters_m, 3)
        assert subset.members.weights.shape == (clusters_m,)
        assert subset.members.pmatrix.shape == (clusters_m, clusters_m)
        np.testing.assert_array_equal(subset_ixs, subset.ixs)

    def test_subset_split_reindex_true(self, colors: ColorsArray):
        """Nothing should change"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7), reindex=True)
        subset.set_n(4)
        subset_ixs = sorted(it.chain(*[c.ixs for c in clusters.clusters[:4]]))
        subset_cnt = len(subset_ixs)
        assert subset.members.vectors.shape == (subset_cnt, 3)
        assert subset.members.weights.shape == (subset_cnt,)
        assert subset.members.pmatrix.shape == (subset_cnt, subset_cnt)

    def test_split_past_m_reindex_false(self, colors: ColorsArray):
        """Raise FailedToSplitError if trying to split past m"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7))
        subset_m = len(subset.ixs)
        subset.set_n(subset_m)
        with pytest.raises(FailedToSplitError):
            subset.split()

    def test_split_past_m_reindex_true(self, colors: ColorsArray):
        """Raise FailedToSplitError if trying to split past m"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        subset = clusters.copy(exc_clusters=(4, 5, 6, 7), reindex=True)
        subset_m = len(subset.ixs)
        subset.set_n(subset_m)
        with pytest.raises(FailedToSplitError):
            subset.split()

    def test_exc_clusters_vs_exc_member_clusters(self, colors: ColorsArray):
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        exc_clusters = (4, 5, 6, 7)
        exc_member_clusters = (clusters.clusters[x].ixs[0] for x in exc_clusters)
        subset_a = clusters.copy(exc_clusters=exc_clusters)
        subset_b = clusters.copy(exc_member_clusters=exc_member_clusters)
        subset_c = clusters.copy(
            exc_clusters=exc_clusters, exc_member_clusters=exc_member_clusters
        )
        np.testing.assert_array_equal(subset_a.ixs, subset_b.ixs)
        np.testing.assert_array_equal(subset_b.ixs, subset_c.ixs)

    def test_make_empty(self, colors: ColorsArray):
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(8)
        empty = clusters.copy(exc_clusters=(0, 1, 2, 3, 4, 5, 6, 7))
        assert empty.n == 0


class TestPredicates:
    def test_set_max_sum_error(self, colors: ColorsArray):
        """Split as far as necessary to get below the threshold"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        for _ in range(10):
            clusters.split()
        max_sum_error = 48000
        clusters.set_max_sum_error(max_sum_error)
        assert clusters.get_max_sum_error() <= max_sum_error
        clusters.merge()
        assert clusters.get_max_sum_error() > max_sum_error

    def test_set_max_span(self, colors: ColorsArray):
        """Split as far as necessary to get below the threshold"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        for _ in range(10):
            clusters.split()
        max_span = 2400
        clusters.set_max_span(max_span)
        assert clusters.get_max_span() <= max_span
        clusters.merge()
        assert clusters.get_max_span() > max_span

    def test_set_max_max_error(self, colors: ColorsArray):
        """Split as far as necessary to get below the threshold"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        for _ in range(10):
            clusters.split()
        max_max_error = 1200
        clusters.set_max_max_error(max_max_error)
        assert clusters.get_max_max_error() <= max_max_error
        clusters.merge()
        assert clusters.get_max_max_error() > max_max_error

    def test_set_avg_error(self, colors: ColorsArray):
        """Split as far as necessary to get below the threshold"""
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        for _ in range(10):
            clusters.split()
        max_avg_error = 0.5
        clusters.set_max_avg_error(max_avg_error)
        assert clusters.get_max_avg_error() <= max_avg_error
        clusters.merge()
        assert clusters.get_max_avg_error() > max_avg_error
