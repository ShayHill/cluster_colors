#!/usr/bin/env python3
# last modified: 230118 08:37:20
"""Create and use cluster images from image colors

:author: Shay Hill
:created: 2022-11-07
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from _operator import attrgetter
from PIL import Image

from cluster_colors.cut_colors import cut_colors
from cluster_colors.kmedians import KMediansClusters
from cluster_colors.paths import PICKLE_DIR
from cluster_colors.pool_colors import pool_colors
from cluster_colors.stack_vectors import stack_vectors
from cluster_colors.type_hints import NBits, StackedVectors


def stack_image_colors(
    filename: Path | str,
    num_colors: int = 512,
    pool_bits: NBits = 6,
    ignore_cache: bool = False,
) -> StackedVectors:
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


def get_biggest_color(stacked_colors: StackedVectors) -> tuple[float, ...]:
    """Get the color with the highest weight.

    :param stacked_colors: an array of colors with weight axes
    :return: the color with the highest weight

    Cluster into large clusters, then return the exemplar of the biggest cluster.
    """
    quarter_colorspace_se = 64**2
    clusters = KMediansClusters.from_stacked_vectors(stacked_colors)
    _ = clusters.split_to_se(quarter_colorspace_se)
    clusters.merge_to_find_winner()
    winner = max(clusters, key=attrgetter("w"))
    return winner.exemplar


# def show_clusters(clusters: KMediansClusters, filename_stem: str) -> None:
#     width = 1000
#     sum_weight = sum(c.w for c in clusters)
#     stripes: list[FPArray] = []
#     for cluster in clusters:
#         stripe_width = max(round(cluster.w / sum_weight * width), 1)
#         stripes.append(
#             np.tile(cluster.vs, (800, stripe_width))
#             .reshape(800, stripe_width, 3)
#             .astype(np.uint8)
#         )
#     # combine stripes into one array
#     image = np.concatenate(stripes, axis=1)

#     # image = Image.fromarray(np.hstack(*stripes))
#     image = Image.fromarray(image)
#     image.save(BINARIES_DIR / f"{filename_stem}-{len(clusters)}.png")


# if __name__ == "__main__":
#     # open image, convert to array, and pass color array to get_biggest_color

#     # img = Image.open("test/sugar-shack-barnes.jpg")
#     from cluster_colors.paths import TEST_DIR, BINARIES_DIR
#     import time

#     start = time.time()
#     colors = stack_image_colors(TEST_DIR / "sugar-shack-barnes.jpg")
#     print(time.time() - start)

#     clusters = KMediansClusters.from_stacked_vectors(colors)
#     _ = clusters.split_to_se(32**2)
#     show_clusters(clusters, "sugar-shack-barnes")

#     get_biggest_color(colors)
