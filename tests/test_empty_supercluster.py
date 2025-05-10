"""Test that cost metrics raise an empty supercluster error when they have no data.

:author: Shay Hill
:created: 2024-10-07
"""

from typing import Annotated

import numpy as np
import pytest
from numpy import typing as npt

from cluster_colors import exceptions
from cluster_colors.cluster_members import Members
from cluster_colors.cluster_supercluster import DivisiveSupercluster
from cluster_colors.vector_stacker import stack_vectors

ColorsArray = Annotated[npt.NDArray[np.floating], (-1, 3)]


@pytest.fixture(
    scope="function",
    params=[np.random.randint(0, 255, (100, 4), dtype=np.uint8) for _ in range(10)],
)
def colors(request: pytest.FixtureRequest) -> ColorsArray:
    return stack_vectors(request.param)


class TestKMedians:
    def test_init(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        assert empty_supercluster.members == members
        assert empty_supercluster.clusters == []

    def test_as_stacked_vectors(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        assert empty_supercluster.get_as_stacked_vectors().shape == (0, 4)

    def test_as_vectors(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        assert empty_supercluster.get_as_vectors().shape == (0, 3)

    def test_split(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        with pytest.raises(exceptions.FailedToSplitError) as exc:
            empty_supercluster.split()
        assert exc.match("No clusters to split.")

    def test_merge(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        with pytest.raises(exceptions.FailedToMergeError) as exc:
            empty_supercluster.merge()
        assert exc.match("No clusters to merge.")

    def test_set_n(self, colors: ColorsArray):
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        with pytest.raises(exceptions.EmptySuperclusterError) as exc:
            empty_supercluster.set_n(3)
        assert exc.match("No clusters to merge or split.")

    def test_get_max_sum(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        with pytest.raises(exceptions.EmptySuperclusterError):
            _ = empty_supercluster.get_max_sum_error()

    def test_set_max_sum(self, colors: ColorsArray):
        """Test that the clusters are sorted by the number of colors in them"""
        members = Members(colors)
        empty_supercluster = DivisiveSupercluster(members, [])
        with pytest.raises(exceptions.EmptySuperclusterError):
            empty_supercluster.set_max_sum_error(0)
