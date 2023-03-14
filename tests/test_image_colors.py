"""Test image_colors module."""

from cluster_colors import image_colors
from cluster_colors.paths import TEST_DIR
from cluster_colors.kmedians import KMediansClusters

_TEST_IMAGE = TEST_DIR / 'sugar-shack-barnes.jpg'

class TestGetBiggestColor:

    # def test_run(self):
    #     """Test get_biggest_color function."""
    #     colors = image_colors.stack_image_colors(_TEST_IMAGE)
    #     biggest_color = image_colors.get_biggest_color(colors)
    #     assert biggest_color == (22.0, 26.0, 20.0)

    def test_display(self):
        """Test display_biggest_color function."""
        quarter_colorspace_se = 16**2
        colors = image_colors.stack_image_colors(_TEST_IMAGE)
        clusters = KMediansClusters.from_stacked_vectors(colors)

        _ = clusters.split_to_delta_e(quarter_colorspace_se) 
        image_colors.show_clusters(clusters, "sugar-shack-barnes")
