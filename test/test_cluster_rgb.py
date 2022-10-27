#!/usr/bin/env python3
# last modified: 221026 20:56:44
"""Test functions in triangulate_image.cluster_rgb.py

:author: Shay Hill
:created: 2022-09-16
"""

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

# pyright: reportPrivateUsage=false
import cluster_colors.rgb_members_and_clusters
from cluster_colors import cluster_rgb
from cluster_colors.stack_colors import stack_colors


@pytest.fixture
def thin_cluster() -> cluster_colors.rgb_members_and_clusters.Cluster:
    """A cluster with all members along the (0,0,0) to (1,1,1) line."""
    colors = np.array([[x, x, x, x] for x in range(10)])
    members = cluster_colors.rgb_members_and_clusters.Member.new_members(colors)
    return cluster_colors.rgb_members_and_clusters.Cluster(members)


class TestMemberNewMembers:
    """Test Member.new_members."""

    def test_one_member_per_color(self) -> None:
        """Return 256 members given 256 colors."""
        colors = stack_colors(np.random.randint(1, 256, (256, 4), dtype=np.uint8))
        members = cluster_rgb.Member.new_members(colors)
        assert len(members) == len(colors)




# class TestMemberColorToRgbw:
# """Test triangulate_image._Member._color_to_rgbw() staticmethod."""

# def test_assign_to_rgb(self):
# """Add a weight of 255 to a 3-value vector."""
# assert cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw((1, 2, 3)) == (1, 2, 3, 255)

# def test_assign_to_rgba(self):
# """Weight of a 4-value vector is 255 - the last value."""
# assert cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw((1, 2, 3, 4)) == (1, 2, 3, 251)

# def test_float_input(self):
# """Raise a TypeError if input cannot be cast to int without loss."""
# with pytest.raises(TypeError):
# _ = cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw((1.1, 2, 3))

# def test_dot0_input(self):
# """Do not raise a TypeError if float input can be cast to int without loss."""
# assert cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw((1.0, 2, 3)) == (1, 2, 3, 255)

# def test_values_in_range(self):
# """Raise a ValueError if input vector is not in [0..255]."""
# with pytest.raises(ValueError):
# _ = cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw((1, 2, 300))

# def test_three_or_four_values(self):
# """Raise a ValueError if input vector is not 3 or 4 values."""
# too_many = cast(tuple[int, int, int], (1, 2, 3, 4, 5))
# too_few = cast(tuple[int, int, int], (1, 2))
# with pytest.raises(ValueError):
# _ = cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw(too_many)
# with pytest.raises(ValueError):
# _ = cluster_colors.rgb_members_and_clusters.Member._color_to_rgbw(too_few)


# class TestMemberGetMemberInstances:
# """Test triangulate_image._Member.new_members() classmethod."""

# def test_pass_list_of_rgb(self):
# """Return a list of _Member instances from a list of colors."""
# assert cluster_colors.rgb_members_and_clusters.Member.new_members([(1, 2, 3), (4, 5, 6)]) == {
# cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 255),
# cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 255),
# }

# def test_pass_list_of_rgba(self):
# """Return a list of _Member instances from a list of colors."""
# assert cluster_colors.rgb_members_and_clusters.Member.new_members(
# [(1, 2, 3, 4), (4, 5, 6, 7)]
# ) == {cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 251), cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 248)}

# def test_merges_rgb_dupliates(self):
# """Combine duplicate colors into a single _Member instance."""
# assert cluster_colors.rgb_members_and_clusters.Member.new_members([(1, 2, 3), (1, 2, 3)]) == {
# cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 510),
# }

# def test_flatten_multi_dimensional_rgb(self) -> None:
# """Flatten a 3D array of colors into a list of _Member instances."""
# assert cluster_colors.rgb_members_and_clusters.Member.new_members(
# np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
# ) == {
# cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 510),
# cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 510),
# }

# def test_flatten_multi_dimensional_rgba(self) -> None:
# """Flatten a 3D array of rgba colors into a list of _Member instances."""
# colors = np.array([[[1, 2, 3, 4], [4, 5, 6, 7]], [[1, 2, 3, 4], [4, 5, 6, 7]]])
# assert cluster_colors.rgb_members_and_clusters.Member.new_members(colors) == {
# cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 502),
# cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 496),
# }


# class TestClusterExemplar:
# """Test triangulate_image._Cluster.exemplar property"""

# def test_exemplar(self) -> None:
# """Return weighted average of member.rgb values."""
# cluster = cluster_colors.rgb_members_and_clusters.Cluster(
# {cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 2), cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 1)}
# )
# assert cluster.exemplar == (1, 2, 3)

# def test_resets_exemplar_age(self) -> None:
# """Reset exemplar_age to 0 when exemplar is accessed."""
# cluster = cluster_colors.rgb_members_and_clusters.Cluster(
# {cluster_colors.rgb_members_and_clusters.Member((1, 2, 3), 2), cluster_colors.rgb_members_and_clusters.Member((4, 5, 6), 1)}
# )
# cluster.exemplar_age = 1
# _ = cluster.exemplar
# assert cluster.exemplar_age == 0

# # class TestClusterAxis:
# # def test_1d(self, thin_cluster) -> None:
# # """Return the axis of the cluster."""
# # aaa = thin_cluster.axis_of_highest_variance
# # breakpoint()
