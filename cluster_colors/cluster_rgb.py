#!/usr/bin/env python3
# last modified: 221001 16:12:25
"""Cluster RGB values, no optimization.

Cluster into an indeterminate number of groups. Will be fine for a small number of
colors.

# TODO: replace all uses of "cost" with "error"

:author: Shay Hill
:created: 2022-09-14
"""

from __future__ import annotations

import functools
import itertools
import math
from contextlib import suppress
from operator import attrgetter, itemgetter
from typing import Annotated, Any, Callable, Iterable, Optional, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from cluster_colors.rgb_members_and_clusters import Cluster, Member, get_squared_error

_MAX_ITERATIONS = 1000


_T = TypeVar("_T")

_RgbF = tuple[float, float, float]
_RgbI = tuple[int, int, int]

_ThreeInts: TypeAlias = tuple[int, int, int]
_FourInts: TypeAlias = tuple[int, int, int, int]

_ColorArray = (
    Annotated[npt.NDArray[np.uint8], "(..., 3) | (..., 4)"] | Iterable[Iterable[int]]
)


class _ErrorGetter:
    """Call the cached methods in _Cluster with cluster instances as arguments."""

    @staticmethod
    def get_half_exemplar_span(cluster_a: Cluster, cluster_b: Cluster) -> float:
        """The span between two clusters."""
        return cluster_a.get_half_squared_error(cluster_b.exemplar)

    @staticmethod
    def get_member_error(member: Member, cluster: Cluster) -> float:
        """What is the cost for having member in cluster?"""
        return cluster.get_squared_error(member.rgb)


class _ClusterSplitter(_ErrorGetter):
    """Split the cluster in self.clusters with the highest cost.

    Examine all clusters in self.clusters. Find the member with the highest cost
    among all clusters. Create a new cluster with just that member. Remove the member
    from the original cluster.
    """

    def __init__(self, clusters: set[Cluster]) -> None:
        self.clusters = clusters

    def __call__(self, min_error_to_split: float) -> bool:
        """Split the cluster with the highest SSE. Return True if a split occurred.

        :param min_error_to_split: the cost threshold for splitting
        """
        candidates = {c for c in self.clusters if len(c.members) > 1}
        if not candidates:
            return False
        graded = ((c.sum_squared_error, c) for c in candidates)
        max_error, cluster = max(graded, key=itemgetter(0))
        if max_error < min_error_to_split:
            return False
        self.clusters |= cluster.split()
        self.clusters.remove(cluster)
        return True


class _ClusterMerger(_ErrorGetter):
    """Merge clusters in self.clusters with the lowest cost.

    Examine all clusters in self.clusters. Find the two closest clusters, A and B.
    Remove these clusters and add a new cluster with their combined members.
    """

    def __init__(self, clusters: set[Cluster]) -> None:
        self.clusters = clusters

    def _find_min_cost(self) -> tuple[float, Cluster, Cluster]:
        """Find the two clusters with the closest exemplars.

        :return: (cluster_a, cluster_b)
        """
        min_cost = float("inf")
        min_cluster_a: Optional[Cluster] = None
        min_cluster_b: Optional[Cluster] = None
        for cluster_a, cluster_b in itertools.combinations(self.clusters, 2):
            cost = cluster_a.get_squared_error(cluster_b.exemplar)
            if cost < min_cost:
                min_cost = cost
                min_cluster_a = cluster_a
                min_cluster_b = cluster_b
        if min_cluster_a is None or min_cluster_b is None:
            raise RuntimeError("min_cluster_a or min_cluster_b is None")
        return min_cost, min_cluster_a, min_cluster_b

    def __call__(self, merge_below_cost: float) -> bool:
        """Merge the two clusters with the lowest cost. Return True if a merge occurred.

        :param merge_below_cost: the cost threshold for merging
        """
        if len(self.clusters) < 2:
            return False
        min_cost, min_cluster_a, min_cluster_b = self._find_min_cost()
        if min_cost > merge_below_cost:
            return False
        combined_members = min_cluster_a.members | min_cluster_b.members
        self.clusters.remove(min_cluster_b)
        self.clusters.remove(min_cluster_b)
        self.clusters.add(Cluster(combined_members))
        return True


class _ClusterReassigner(_ErrorGetter):
    """Reassign members to the closest cluster exemplar."""

    def __init__(self, clusters: set[Cluster]) -> None:
        self.clusters = clusters

    def _get_safe_cost(self, cluster: Cluster, others: set[Cluster]) -> float:
        """Determine the threshold below which members cannot be removed.

        If a member is closer than half the distance to the nearest exemplar, then no
        other exemplar has an opportunity to claim it.
        """
        return min(self.get_half_exemplar_span(cluster, x) for x in others)

    def _offer_members(self, cluster: Cluster) -> None:
        """Look for another cluster with lower cost for members of input cluster."""
        if cluster.exemplar_age == 0:
            others = self.clusters - {cluster}
        else:
            others = {x for x in self.clusters if x.exemplar_age == 0}
        if not others:
            return

        safe_cost = self._get_safe_cost(cluster, others)
        for member in cluster.members:
            cost_in_cluster = functools.partial(self.get_member_error, member)
            current_cost = cost_in_cluster(cluster)
            if current_cost <= safe_cost:
                continue
            best_fit = min(others, key=cost_in_cluster)
            if cost_in_cluster(best_fit) < current_cost:
                cluster.queue_sub.add(member)
                best_fit.queue_add.add(member)

    def __call__(self) -> bool:
        """Pass members between clusters and update exemplars."""
        if len(self.clusters) < 2:
            return False
        if all(x.exemplar_age > 0 for x in self.clusters):
            return False
        for cluster in self.clusters:
            self._offer_members(cluster)
        return True


class _Clusters(_ErrorGetter):
    def __init__(
        self,
        colors: _ColorArray,
        min_span: float = 0,
        max_span: float = math.inf,
    ) -> None:
        self.clusters = {Cluster(Member.new_members(colors))}

        self.splitter = _ClusterSplitter(self.clusters)
        self.merger = _ClusterMerger(self.clusters)
        self.reassigner = _ClusterReassigner(self.clusters)

    @property
    def clusters(self) -> set[Cluster]:
        return self.new_clusters | self.old_clusters

    def _process_queues(self) -> None:
        """Apply queued changes to clusters."""
        prev_clusters = tuple(self.clusters)
        self.clusters.clear()
        self.clusters |= {c.process_queue() for c in prev_clusters}

    def converge(self) -> None:
        """Reassign members until no changes occur."""
        iterations = 0
        while self.reassigner() and iterations < _MAX_ITERATIONS:
            self._process_queues()
            iterations += 1

    def split_to_threshold(self, split_above_cost: float) -> None:
        """Split clusters until the max_span is reached."""
        while self.splitter(split_above_cost):
            self._process_queues()
            self.converge()

    def split_to_count(self, count: int) -> None:
        """Split clusters until len(clusters) == count."""
        while len(self.clusters) < count:
            _ = self.splitter(0)
            self._process_queues()
            self.converge()

    def merge_to_threshold(self, merge_below_cost: float) -> None:
        """Merge clusters until the min_span is reached."""
        while self.merger(merge_below_cost):
            self._process_queues()
            self.converge()

    def merge_to_count(self, count: int) -> None:
        """Merge clusters until len(clusters) == count."""
        while len(self.clusters) > count:
            _ = self.merger(0)
            self._process_queues()
            self.converge()

    @property
    def _is_clear_winner(self) -> bool:
        """Is one cluster heavier than the rest?"""
        if len(self.clusters) == 1:
            return True
        weights = [c.weight for c in self.clusters]
        return weights.count(max(weights)) == 1

    def merge_to_find_winner(self) -> None:
        """Merge clusters until there is a clear winner."""
        while not self._is_clear_winner:
            _ = self.merger(0)
            self._process_queues()
            self.converge()


# ==============================================================================
#  Public
# ==============================================================================

_ClusterTuple = tuple[_RgbF, set[tuple[float, float, float, float]]]


def _get_cluster_tuple(cluster: Cluster) -> _ClusterTuple:
    """Return the cluster's exemplar and m.rgba for each member."""
    return cluster.exemplar, {m.rgbt for m in cluster.members}


def _get_color_clusters(
    colors: _ColorArray,
    merge_below_cost: float = 0,
    split_above_cost: float = 32,
    max_clusters: Optional[int] = None,
) -> set[Cluster]:
    """Cluster colors.

    :param colors: an iterable of colors
    :param merge_below_cost: the cost threshold for merging
    :param split_above_cost: the cost threshold for splitting
    :param num_clusters: the number of clusters to return
    :return: a dictionary of clusters
    """
    clusters = _Clusters(colors)
    clusters.converge()
    clusters.split_to_threshold(split_above_cost)
    clusters.merge_to_threshold(merge_below_cost)
    if max_clusters is not None:
        clusters.merge_to_count(max_clusters)
    return clusters.clusters


def get_biggest_color(
    colors: _ColorArray,
    merge_below_cost: float = 0,
    split_above_cost: float = 64,
    max_clusters: Optional[int] = None,
) -> _ClusterTuple:
    """Return the dominant color (and rgba constituents).

    :param colors: an iterable of colors
    :param merge_below_cost: the cost threshold for merging
    :param split_above_cost: the cost threshold for splitting
    :param num_clusters: the number of clusters to return
    :return: a tuple of the dominant color and a set of rgba constituents
    """
    clusters = _get_color_clusters(
        colors, merge_below_cost, split_above_cost, max_clusters
    )
    return _get_cluster_tuple(max(clusters, key=attrgetter("weight")))


if __name__ == "__main__":
    # open image, convert to array, and pass color array to get_biggest_color
    from PIL import Image
    from PIL.Image import MAXCOVERAGE, FASTOCTREE

    img = Image.open("sugar-shack-barnes.jpg")
    img.quantize(256)
    colors = img.getcolors(256)
    breakpoint()
    count_colors = 160**2
    # try:
    # image = image.quantize(colors=count_colors, method=MAXCOVERAGE)
    # except ValueError:
    # image = image.quantize(colors=count_colors, method=FASTOCTREE)
    # count_colors = len(image.getcolors())
    # if count_colors < count_colors:
    # image = image.quantize(count_colors, method=2)

    # result = image.convert('P', colors=count_colors)
    # breakpoint()
    colors = np.array(image)

    # colors = np.array(image.getpalette()).reshape((-1, 3))

    # colors = img.quantize(colors=256).getcolors(img.width * img.height)
    # reakpoint()
    # colors = np.array(img)
    # colors = [x[1]+(x[0],) for x in colors]

    # time how long it takes to get the dominant color
    import time

    start = time.time()
    color = get_biggest_color(colors)
    end = time.time()
    print(f"Time: {end - start:.2f} seconds")

    print(color)
