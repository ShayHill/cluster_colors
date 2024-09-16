"""Test methods in KMedDivisiveSupercluster

:author: Shay Hill
:created: 2023-03-14
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from cluster_colors.cluster_supercluster import (
    AgglomerativeSupercluster,
    DivisiveSupercluster,
)
from cluster_colors.vector_stacker import stack_vectors

ColorsArray = Annotated[npt.NDArray[np.float64], (-1, 3)]


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

    def test_from_cluster_subset(self, colors: ColorsArray):
        clusters = DivisiveSupercluster.from_vectors(colors)
        clusters.set_n(8)
        subset = DivisiveSupercluster.from_cluster_subset(*clusters.clusters[:4])
        subset_cnt = sum(len(c.ixs) for c in clusters.clusters[:4])
        assert subset.members.vectors.shape == (subset_cnt, 3)
        assert subset.members.weights.shape == (subset_cnt,)
        assert subset.members.pmatrix.shape == (subset_cnt, subset_cnt)


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
