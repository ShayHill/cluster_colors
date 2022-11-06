from __future__ import annotations

import pickle
from _operator import attrgetter
from pathlib import Path
from typing import Optional

import numpy as np

from cluster_colors.kmedians import KMediansClusters
from cluster_colors.cut_colors import cut_colors
from cluster_colors.paths import PICKLE_DIR, TEST_DIR
from cluster_colors.pool_colors import pool_colors
from cluster_colors.stack_vectors import stack_vectors
from cluster_colors.type_hints import FPArray, NBits, Pixels, StackedColors


def stack_image_colors(
    filename: Path | str,
    num_colors: float = 512,
    pool_bits: NBits = 6,
    ignore_cache: bool = False,
) -> StackedColors:
    """Stack pixel colors and reduce the number of colors in an image.

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
    cache_path = PICKLE_DIR / f"{Path(filename).stem}-{num_colors}.pkl"
    if not ignore_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    img = Image.open(filename)
    img = img.convert("RGBA")
    colors = stack_vectors(np.array(img))
    colors = pool_colors(colors, pool_bits)
    colors = cut_colors(colors, num_colors)

    # save the cache to PICKLE_DIR
    with open(cache_path, "wb") as f:
        pickle.dump(colors, f)

    return colors


def get_biggest_color(stacked_colors: StackedColors) -> tuple[float, ...]:
    """Get the color with the highest weight."""
    quarter_colorspace_se = 64**2
    clusters = KMediansClusters.from_stacked_vectors(stacked_colors)
    _ = clusters.split_to_se(quarter_colorspace_se)
    clusters.merge_to_find_winner()
    winner = max(clusters, key=attrgetter("w"))
    return winner.exemplar


def image_from_array(arr):
    """Create a PIL Image from a numpy array."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save("test.png")


def temp_r(clusters):
    return sum(x.sse for x in clusters.clusters)

    return x.sse / (x._variance * len(x.members))


def display_clusters(clusters):
    """Create a 1200 by 800 pixel array with a vertical stripe for each cluster.exemplar."""
    clusters = sorted(clusters.clusters, key=attrgetter("weight"), reverse=True)
    exemplars = [c.exemplar for c in clusters]
    stripes = [np.tile(x, (800, 100)).reshape(800, 100, 3) for x in exemplars]
    return np.hstack(stripes)


def _show_palette(colors: Pixels, weight: Optional[float] = None) -> tuple[float, ...]:
    """Get the color with the highest weight."""
    quarter_colorspace_se = 4**2
    clusters = _get_clusters(colors, weight)
    _ = clusters.split_to_se(quarter_colorspace_se)
    arr = display_clusters(clusters)
    image_from_array(arr)


def _get_base_colors(
    colors: Pixels, weight: Optional[float] = None, cut_to: int = 512
) -> FPArray:
    """Reduce colors to at or around 512 colors.

    :param colors: an array of colors
    :param weight: a single value to be applied to all colors. Colors with an alpha
        channel will already have a weight, so the default of None will be
        appropriate for colors without an alpha channel.

    :return: a shorter array of colors with weights
    """
    colors = stack_vectors(colors, weight)
    colors = pool_colors(colors, 6)
    colors = cut_colors(colors, cut_to)
    return colors


# TODO: delete this if it's not used
def _get_clusters( stacked_colors: StackedColors) -> KMediansClusters:
    """Create a KMediansClusters instance with one cluster holding all colors.

    :param stacked_colors: an array of colors with weight axes
    :return: a KMediansClusters object
    """
    clusters = KMediansClusters.from_stacked_vectors(stacked_colors)
    return clusters


if __name__ == "__main__":
    # open image, convert to array, and pass color array to get_biggest_color
    from PIL import Image

    # img = Image.open("test/sugar-shack-barnes.jpg")
    import time

    start = time.time()
    colors = stack_image_colors(TEST_DIR / "sugar-shack-barnes.jpg")
    print(time.time() - start)
    # _show_palette(colors)
    get_biggest_color(colors)
    # try:
    # image = image.quantize(colors=count_colors, method=MAXCOVERAGE)
    # except ValueError:
    # image = image.quantize(colors=count_colors, method=FASTOCTREE)
    # count_colors = len(image.getcolors())
    # if count_colors < count_colors:
    # image = image.quantize(count_colors, method=2)

    # result = image.convert('P', colors=count_colors)
    # breakpoint()
    # colors = np.array(image)

    # # colors = np.array(image.getpalette()).reshape((-1, 3))
