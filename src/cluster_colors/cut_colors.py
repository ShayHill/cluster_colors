"""Cluster colors with median cut.

Repeatedly subdivide the color space by splitting along the longest axis (not
constrained to x, y, or z. splits with the optimal plane).

This median cut has a few optimizations, so it will produce a nice heirarchal set of
clusters with some modification, though at this point no hierarchies are preserved.
The idea is to use this to reduce the number of colors in an image to around 512.
From 512, they can be again merged into a single cluster and again split to find a
starting point for kmedians.

:author: Shay Hill
:created: 2022-10-21
"""

import numpy as np

from cluster_colors.cluster_cluster import Cluster
from cluster_colors.type_hints import Vectors


def _split_every_cluster(clusters: set[Cluster]) -> set[Cluster]:
    """Recursively split every cluster.

    :param clusters: A set of clusters.
    :returns: A set of clusters.

    Recursively split every cluster with no regard for error. Will only *not* split
    a cluster if it only has one member.
    """
    splittable = {c for c in clusters if len(c.ixs) > 1}
    if not splittable:
        return clusters
    for cluster in splittable:
        clusters.remove(cluster)
        clusters.update(cluster.split())
    return clusters


def _split_largest_cluster(clusters: set[Cluster], num: int) -> set[Cluster]:
    """Split one cluster per call.

    :param clusters: A set of clusters.
    :returns: A set of clusters.
    """
    if len(clusters) >= num:
        return clusters

    next_split = max(clusters, key=lambda c: c.span)
    if next_split.span == 0:
        return clusters
    clusters.remove(next_split)
    clusters.update(next_split.split())
    return clusters


def cut_colors(colors: Vectors, num: int) -> Vectors:
    """Merge colors into a set of num colors.

    :param colors: a (-1, 4) array of unique rgb values with weights
    :param num: the number of colors to split into
    :returns: a (-1, 4) array of unique rgb values with weights

    Put all colors into one cluster, split that cluster into num clusters, then
    return a median color for each cluster.

    Splits every cluster until roughly half the requested number of clusters have
    been created, then starts cherry picking. This idea was proposed and tested in a
    paper I ran into online, but I can't find it now.
    """
    if len(colors) <= num:
        return colors

    clusters = {Cluster.from_stacked_vectors(colors)}
    while len(clusters) < num // 4:
        clusters = _split_every_cluster(clusters)

    while len(clusters) < num:
        len_clusters = len(clusters)
        clusters = _split_largest_cluster(clusters, num)
        if len(clusters) == len_clusters:
            break

    return np.array([c.get_as_stacked_vector() for c in clusters])
