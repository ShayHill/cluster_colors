"""Create and use cluster instances from image colors.

Some functions will take an image filename, others will take an array of colors.
Arrays of colors must be (r, g, b, w) where w is the weight of the color. When
creating a color array from an image, the w value will be the pixel alpha. For an
input image where every pixel is fully opaque, the w value, after stacking, will be
proportionate to the number of pixels where an (r, g, b) color appears.

:author: Shay Hill
:created: 2022-11-07
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from basic_colormath import get_delta_e_matrix
from PIL import Image

from cluster_colors.cluster_supercluster import SuperclusterBase
from cluster_colors.config import CACHE_DIR
from cluster_colors.cut_colors import cut_colors
from cluster_colors.pool_colors import pool_colors
from cluster_colors.vector_stacker import stack_vectors

if TYPE_CHECKING:
    import os

    from numpy import typing as npt

    from cluster_colors.type_hints import CenterName, NBits, Vectors, VectorsLike

_SuperclusterT = TypeVar("_SuperclusterT", bound=SuperclusterBase)


def stack_pool_cut_colors(
    colors: VectorsLike, num: int | None = None, pool_bits: NBits | None = None
) -> Vectors:
    """Reduce the number of colors to <= num by pooling and cutting.

    :param colors: an array (m, 4) of weighted colors (r, g, b, w)
    :param num: the number of colors to reduce to. Default is 512.
    :param pool_bits: the number of bits to pool colors by. Default is 6. Leave it.
    :return: an array of colors with weights
    """
    num = num or 512
    pool_bits = pool_bits or 6
    float_colors = np.array(colors, dtype=np.floating)
    float_colors = stack_vectors(np.array(float_colors))
    float_colors = pool_colors(float_colors, pool_bits)
    return cut_colors(float_colors, num)


def get_color_supercluster(
    return_type: type[_SuperclusterT],
    colors: VectorsLike,
    num: int | None = None,
    pool_bits: NBits | None = None,
) -> _SuperclusterT:
    """Create a SuperclusterBase instance from an array of colors.

    :param colors: an array (m, 4) of weighted colors (r, g, b, w)
    :param num: the number of colors to reduce to. Default is 512.
    :param pool_bits: the number of bits to pool colors by. Default is 6. Leave it.
    :return: a SuperclusterBase instance
    """
    stacked_colors = stack_pool_cut_colors(colors, num, pool_bits)
    pmatrix = get_delta_e_matrix(stacked_colors[:, :3])
    return return_type.from_stacked_vectors(stacked_colors, pmatrix=pmatrix)


def stack_pool_cut_image_colors(
    filename: Path | str,
    num: int | None = None,
    pool_bits: NBits | None = None,
    *,
    ignore_cache: bool = False,
) -> Vectors:
    """Load cache or stack pixel colors and reduce the number of colors in an image.

    :param filename: the path to an image file
    :param num: the number of colors to reduce to. Default is 512.
    :param pool_bits: the number of bits to pool colors by. Default is 6. Leave it.
    :param ignore_cache: if True, ignore any cached results and recompute the colors.
    :return: an array of colors with weights

    This is a pre-processing step for the color clustering. Stacking is necessary,
    and the pooling and cutting will allow clustering in a reasonable amount of time.
    """
    cache_path = CACHE_DIR / f"{Path(filename).stem}_{num}_{pool_bits}.npy"
    if not ignore_cache and cache_path.exists():
        return np.load(cache_path)

    img = Image.open(filename)
    img = img.convert("RGBA")
    colors = stack_pool_cut_colors(np.array(img), num, pool_bits)
    np.save(cache_path, colors)
    return colors


def get_image_supercluster(
    return_type: type[_SuperclusterT],
    filename: Path | str,
    num: int | None = None,
    pool_bits: NBits | None = None,
    *,
    ignore_cache: bool = False,
) -> _SuperclusterT:
    """Get all colors in an image as a single KMedSupercluster instance.

    :param return_type: the type of SuperclusterBase to return.
    :param filename: the path to an image file
    :param num: the number of colors to reduce to. Default is 512.
    :param pool_bits: the number of bits to pool colors by. Default is 6. Leave it.
    :param ignore_cache: if True, ignore any cached results and recompute the colors.
    :return: a KMedSupercluster instance containing all the colors in the image
    """
    stacked_colors = stack_pool_cut_image_colors(
        filename, num, pool_bits, ignore_cache=ignore_cache
    )
    pmatrix = get_delta_e_matrix(stacked_colors[:, :3])
    return return_type.from_stacked_vectors(stacked_colors, pmatrix=pmatrix)


def show_color_supercluster(
    supercluster: SuperclusterBase,
    filename: str | os.PathLike[str],
    which_center: CenterName | None = None,
) -> None:
    """Create a png with the exemplar of each cluster.

    :param supercluster: the clusters to show
    :param filename: the filename to use for the output file. The number of clusters
        will be added as an infix.
    :param which_center: optionally specify a cluster center attribute. Choices are
        'weighted_median', 'weighted_medoid', or 'unweighted_medoid'. Default is
        'weighted_median'.
    """
    width = 1000
    sum_weight = sum(supercluster.members.weights)
    stripes: list[npt.NDArray[np.uint8]] = []
    for cluster in supercluster.clusters:
        stripe_width = max(round(cluster.weight / sum_weight * width), 1)
        stripes.append(
            np.tile(cluster.get_as_vector(which_center), (800, stripe_width))
            .reshape(800, stripe_width, 3)
            .astype(np.uint8)
        )
    # combine stripes into one array
    image = np.concatenate(stripes, axis=1)

    image = Image.fromarray(image)

    output_path = Path(filename)
    output_name = f"{output_path.stem}-{len(supercluster.clusters)}.png"

    image.save(output_path.parent / output_name)
