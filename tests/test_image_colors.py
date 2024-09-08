"""Test image_colors module."""

from cluster_colors import image_colors
from cluster_colors.paths import TEST_DIR
from cluster_colors.kmedians import KMedSupercluster
from cluster_colors.clusters import Supercluster
import numpy as np

_TEST_IMAGE = TEST_DIR / 'sugar-shack-barnes.jpg'

class TestGetBiggestColor:

    def test_display(self):
        """Test display_biggest_color function."""
        quarter_colorspace_se = 16**2
        colors = image_colors.stack_image_colors(_TEST_IMAGE)
        clusters = Supercluster.from_stacked_vectors(colors)

        _ = clusters.split_to_intercluster_proximity(np.inf)
        # TODO: restore show_clusters
        # image_colors.show_clusters(clusters, "sugar-shack-barnes")
