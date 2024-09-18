"""Test image_colors module."""

from cluster_colors import image_colors
from cluster_colors.cluster_supercluster import DivisiveSupercluster
from cluster_colors.paths import BINARIES_DIR, TEST_DIR

_TEST_IMAGE = TEST_DIR / "sugar-shack-barnes.jpg"


class TestGetBiggestColor:

    def test_display(self):
        """Test display_biggest_color function."""
        colors = image_colors.stack_image_colors(_TEST_IMAGE)
        clusters = DivisiveSupercluster.from_stacked_vectors(colors)
        clusters.set_n(2)
        clusters.set_max_max_error(64**2)
        image_colors.show_clusters(clusters, BINARIES_DIR / _TEST_IMAGE.stem)
