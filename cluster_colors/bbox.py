#!/usr/bin/env python3
# last modified: 221027 12:52:30
"""Simple BBox class

Create a square bounding sphere around a set of points.

Creates a degenerate bounding box when an intersection of two disjoint bounding boxes
is requested. This just gives a None-like BBox instance that will intersect with
nothing and contain nothing. This should remove a lot of the exception handling
outside the class.

:author: Shay Hill
:created: 2022-10-26
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np

from cluster_colors.type_hints import FPArray


@dataclass
class BBox:
    def __init__(
        self, mins: Optional[FPArray] = None, maxs: Optional[FPArray] = None
    ) -> None:
        if mins is None or maxs is None or np.any(mins > maxs):
            # make sure everything fails if not checked for _is_degenerate
            self._mins, self._maxs, self._dims = None, None, None
            self._is_degenerate = True
        else:
            self._mins, self._maxs = mins, maxs
            self._dims = maxs - mins
            self._is_degenerate = False

    @property
    def mins(self) -> FPArray:
        if self._mins is None:
            raise ValueError("BBox is degenerate")
        return self._mins

    @property
    def maxs(self) -> FPArray:
        if self._maxs is None:
            raise ValueError("BBox is degenerate")
        return self._maxs

    @property
    def dims(self) -> FPArray:
        if self._dims is None:
            raise ValueError("BBox is degenerate")
        return self._dims

    @classmethod
    def maybe_vector_bbox(cls, vectors: Annotated[FPArray, (-1, -1)]) -> BBox:
        """Create a loose BBox from a set of vectors

        :param vectors: (n, d) array of vectors
        :return: bounding box with equal sides.

        If vectors are 2D, the box is square, if 3D, the box is a cube. This is to
        loosely approximate a bounding circle or sphere.

        If only one vector is given, the return None. This is used for clusters, and
        a cluster with only one member cannot offer members to other clusters.
        """
        assert vectors.shape[0] > 0
        if vectors.shape[0] == 1:
            return cls(vectors[0], vectors[0])
        mins = np.min(vectors, axis=0)
        maxs = np.max(vectors, axis=0)
        center = (mins + maxs) / 2
        max_dim = np.max(maxs - mins)

        loose_mins = center - max_dim / 2
        loose_maxs = center + max_dim / 2
        return cls(loose_mins, loose_maxs)

    def __contains__(self, vector: FPArray) -> bool:
        if self._is_degenerate:
            return False
        return np.all(np.logical_and(vector >= self.mins, vector <= self.maxs))

    def intersection(self, other: BBox) -> BBox:
        if self._is_degenerate or other._is_degenerate:
            return BBox()
        mins = np.maximum(self.mins, other.mins)
        maxs = np.minimum(self.maxs, other.maxs)
        return BBox(mins, maxs)

    def point_intersection(self, point: FPArray) -> BBox:
        """Intersection of self and a BBox instance of the same dims centered on point.

        The box returned will be the region potentially shared by 
        """
        if self._is_degenerate:
            return BBox()
        mins = point - self.dims / 2
        maxs = point + self.dims / 2
        other_bbox = BBox(mins, maxs)
        return self.intersection(other_bbox)


    def intersects(self, other: BBox) -> bool:
        return not self.intersection(other)._is_degenerate

class BoundingSphere:
    def __init__(self, center: FPArray, radius: float) -> None:
        self.center = center
        self.radius = radius

    def __contains__(self, vector: FPArray) -> bool:
        """determine if a point lies within a sphere. do not use norm"""
        return np.linalg.norm(vector - self.center) <= self.radius

    def intersects(self, other: BoundingSphere) -> bool:
        return np.linalg.norm(other.center - self.center) <= self.radius + other.radius

    def intersection(self, other: BoundingSphere) -> BoundingSphere:
        if not self.intersects(other):
            return BoundingSphere(self.center, 0)
