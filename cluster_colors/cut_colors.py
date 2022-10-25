#!/usr/bin/env python3
# last modified: 221023 09:52:32
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
from operator import attrgetter

import numpy as np

from cluster_colors.rgb_members_and_clusters import Cluster, Member
from cluster_colors.type_hints import StackedColors


def _split_every_cluster(clusters: set[Cluster], max_num: int) -> set[Cluster]:
    """Recursively split every cluster.

    :param clusters: A set of clusters.
    :returns: A set of clusters.

    Recursively split every cluster with no regard for error. Will only *not* split
    a cluster if it only has one member.
    """
    splittable = {c for c in clusters if len(c.members) > 1}
    if not splittable or len(splittable) + len(clusters) > max_num:
        return clusters
    for cluster in splittable:
        clusters.remove(cluster)
        clusters.update(cluster.split())
    return _split_every_cluster(clusters, max_num)



def _split_largest_cluster(clusters: set[Cluster], num: int) -> set[Cluster]:
    """Split one cluster per call.

    :param clusters: A set of clusters.
    :returns: A set of clusters.
    """
    if len(clusters) >= num:
        return clusters
    max_error = max(c.quick_error for c in clusters)
    if max_error == 0:
        return clusters
    for cluster in [c for c in clusters if c.quick_error == max_error]:
        clusters.remove(cluster)
        clusters.update(cluster.split())
    return _split_every_cluster(clusters, num)


def cut_colors(colors: StackedColors, num: int) -> StackedColors:
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
    clusters = {Cluster(Member.new_members(colors))}
    clusters = _split_every_cluster(clusters, num // 2)
    while len(clusters) < num:
        if not any(len(c.members) > 1 for c in clusters):
            break
        clusters = _split_largest_cluster(clusters, num)
    return np.array([c.as_member for c in clusters])

def time_cut_colors():
    """Time the cut_colors function."""
    from cluster_colors import stack_colors
    from cluster_colors import pool_colors
    import time
    start = time.time()
    colors = np.random.randint(0, 255, (100_000, 3), dtype=np.uint8)
    colors = pool_colors.pool_colors(colors)
    aaa = cut_colors(colors, 512)
    print (time.time() - start)

if __name__ == "__main__":
    time_cut_colors()
