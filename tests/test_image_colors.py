"""Test image_colors module."""

from cluster_colors import image_colors
from cluster_colors.paths import TEST_DIR
from cluster_colors.clusters import Supercluster

_TEST_IMAGE = TEST_DIR / 'sugar-shack-barnes.jpg'

class TestGetBiggestColor:

    def test_display(self):
        """Test display_biggest_color function."""
        quarter_colorspace_se = 16**2
        colors = image_colors.stack_image_colors(_TEST_IMAGE)
        clusters = Supercluster.from_stacked_vectors(colors)
        clusters.set_n(2)

        _ = clusters.set_min_proximity(quarter_colorspace_se)
        # breakpoint()
        # TODO: restore show_clusters
        # image_colors.show_clusters(clusters, "sugar-shack-barnes")
