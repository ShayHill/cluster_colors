"""Create and use cluster images from image colors.

:author: Shay Hill
:created: 2022-11-07
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from cluster_colors.config import CACHE_DIR
from cluster_colors.cut_colors import cut_colors
from cluster_colors.kmedians import KMediansClusters
from cluster_colors.paths import BINARIES_DIR
from cluster_colors.pool_colors import pool_colors
from cluster_colors.vector_stacker import stack_vectors

if TYPE_CHECKING:
    from cluster_colors.type_hints import FPArray, NBits, StackedVectors


def _stack_image_colors_no_cache(
    filename: Path | str, num_colors: int = 512, pool_bits: NBits = 6
) -> StackedVectors:
    """Stack pixel colors and reduce the number of colors in an image.

    :param filename: the path to an image file
    :param num_colors: the number of colors to reduce to. The default of 512 will
        cluster quickly down to medium-sized clusters.
    :param pool_bits: the number of bits to pool colors by. The default of 6 is a
    good value. You can probably just ignore this parameter, but it's here to
        eliminate a "magic number" from the code.
    :return: an array of colors with weights
    """
    img = Image.open(filename)
    img = img.convert("RGBA")
    colors = stack_vectors(np.array(img))
    colors = pool_colors(colors, pool_bits)
    return cut_colors(colors, num_colors)


def stack_image_colors(
    filename: Path | str,
    num_colors: int = 512,
    pool_bits: NBits = 6,
    *,
    ignore_cache: bool = False,
) -> StackedVectors:
    """Load cache or stack pixel colors and reduce the number of colors in an image.

    :param filename: the path to an image file
    :param num_colors: the number of colors to reduce to. The default of 512 will
        cluster quickly down to medium-sized clusters.
    :param pool_bits: the number of bits to pool colors by. The default of 6 is a
    good value. You can probably just ignore this parameter, but it's here to
        eliminate a "magic number" from the code.
    :param ignore_cache: if True, ignore any cached results and recompute the colors.
    :return: an array of colors with weights

    This is a pre-processing step for the color clustering. Stacking is necessary,
    and the pooling and cutting will allow clustering in a reasonable amount of time.
    """
    cache_path = CACHE_DIR / f"{Path(filename).stem}_{num_colors}_{pool_bits}.npy"
    if not ignore_cache and cache_path.exists():
        return np.load(cache_path)

    colors = _stack_image_colors_no_cache(filename, num_colors, pool_bits)
    np.save(cache_path, colors)
    return colors


def get_biggest_color(stacked_colors: StackedVectors) -> tuple[float, ...]:
    """Get the color with the highest weight.

    :param stacked_colors: an array of colors with weight axes
    :return: the color with the highest weight

    Cluster into large clusters, then return the exemplar of the biggest cluster.
    """
    quarter_colorspace_se = 64**2
    clusters = KMediansClusters.from_stacked_vectors(stacked_colors)
    clusters.split_to_delta_e(quarter_colorspace_se)
    return clusters.get_rsorted_exemplars()[0]


def get_image_clusters(
    filename: Path | str,
    num_colors: int = 512,
    pool_bits: NBits = 6,
    *,
    ignore_cache: bool = False,
) -> KMediansClusters:
    """Get all colors in an image as a single KMediansClusters instance.

    :param filename: the path to an image file
    :param num_colors: the number of colors to reduce to. The default of 512 will
        cluster quickly down to medium-sized clusters.
    :param pool_bits: the number of bits to pool colors by. The default of 6 is a
    good value. You can probably just ignore this parameter, but it's here to
        eliminate a "magic number" from the code.
    :param ignore_cache: if True, ignore any cached results and recompute the colors.
    :return: a KMediansClusters instance containing all the colors in the image
    """
    stacked_colors = stack_image_colors(
        filename, num_colors, pool_bits, ignore_cache=ignore_cache
    )
    return KMediansClusters.from_stacked_vectors(stacked_colors)


def show_clusters(clusters: KMediansClusters, filename_stem: str) -> None:
    """Create a png with the exemplar of each cluster.

    :param clusters: the clusters to show
    :param filename_stem: the filename stem to use for the output file
    """
    width = 1000
    sum_weight = sum(c.w for c in clusters)
    stripes: list[FPArray] = []
    for cluster in clusters:
        stripe_width = max(round(cluster.w / sum_weight * width), 1)
        stripes.append(
            np.tile(cluster.vs, (800, stripe_width))
            .reshape(800, stripe_width, 3)
            .astype(np.uint8)
        )
    # combine stripes into one array
    image = np.concatenate(stripes, axis=1)

    image = Image.fromarray(image)
    image.save(BINARIES_DIR / f"{filename_stem}-{len(clusters)}.png")
