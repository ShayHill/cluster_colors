"""One member of a cluster.

I left some flexibility in this library to work with vectors of any length, but there
are specialized methods for working with RGB colors. These will fail if the vector is
not 3 values long.

:author: Shay Hill
:created: 2024-09-01
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import get_sqeuclidean_matrix

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cluster_colors.type_hints import FPArray, ProximityMatrix, Vectors, VectorsLike


class Members:
    """A list of cluster members with a proximity matrix."""

    def __init__(
        self,
        vectors: VectorsLike,
        *,
        weights: Iterable[float] | None = None,
        pmatrix: ProximityMatrix | None = None,
    ) -> None:
        """Create a new Members instance.

        :param vectors: array (n, m) of vectors
        :param weights: optional array (n,) of weights
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance

        The vectors must be sorted in case any members has equal proximity to
        multiple centers. When this happens, the centroid with the lowest
        lexigraphical order is chosen.
        """
        vectors = np.asarray(vectors)
        sort_indices = np.lexsort(vectors.T[::-1])

        self.vectors = vectors[sort_indices]
        if weights is None:
            self.weights = np.ones(len(self.vectors))
        else:
            self.weights = np.array(list(weights))[sort_indices]
        if pmatrix is None:
            self.pmatrix = get_sqeuclidean_matrix(self.vectors)
        else:
            self.pmatrix = pmatrix[sort_indices, sort_indices]

    def __len__(self) -> int:
        """Number of members in the Members instance.

        :return: number of members
        """
        return len(self.vectors)

    @functools.cached_property
    def weighted_pmatrix(self) -> ProximityMatrix:
        """Proximity matrix with weights applied.

        :return: proximity matrix such that sum(pmatrix[i, (j, k, ...)]) is the cost
            of members[i] in a cluster with members[i, j, k, ...]
        """
        weight_columns = np.tile(self.weights, (len(self.weights), 1))
        return self.pmatrix * weight_columns

    @functools.cached_property
    def pmatrix_with_inf_diagonal(self) -> ProximityMatrix:
        """Proximity matrix with infinity on the diagonal.

        :return: proximity matrix with infinity on the diagonal. The is useful for
            finding the minumum proximity between members that is *not* the distance
            between a member and itself.
        """
        pmatrix_copy = self.pmatrix.copy()
        np.fill_diagonal(pmatrix_copy, np.inf)
        return pmatrix_copy

    # ===========================================================================
    #   constructors
    # ===========================================================================

    @classmethod
    def from_vectors(
        cls, vectors: Vectors, *, pmatrix: FPArray | None = None
    ) -> Members:
        """Create a Members instance from stacked_vectors.

        :param vectors: (n, m) a list of vectors
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: Members instance
        """
        return cls(vectors, pmatrix=pmatrix)

    @classmethod
    def from_stacked_vectors(
        cls, stacked_vectors: Vectors, *, pmatrix: FPArray | None = None
    ) -> Members:
        """Create a Members instance from stacked_vectors.

        :param stacked_vectors: (n, m + 1) a list of vectors with weight channels in
            the last axis
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: Members instance
        """
        return cls(
            stacked_vectors[:, :-1], weights=stacked_vectors[:, -1], pmatrix=pmatrix
        )
