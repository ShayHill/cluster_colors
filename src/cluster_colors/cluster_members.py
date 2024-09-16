"""Members of a cluster represented as a list of vectors with a proximity matrix.

The clustering algorithms in this package do not calculate proximity, but only refer
to a given proximity matrix. This means that more expensive proximity calculations
can be used with the minimum penalty. Each Supercluster and the Clusters it contains
share a single Members instance. This instance holds all vector information and a few
variants of the proximity matrix. A Cluster is defined by the indices of the members
it contains.

:author: Shay Hill
:created: 2024-09-01
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import get_sqeuclidean_matrix
from paragraphs import par

from cluster_colors.vector_stacker import add_weight_axis, stack_vectors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cluster_colors.type_hints import FPArray, ProximityMatrix, VectorsLike


class Members:
    """A list of cluster vectors with a proximity matrix."""

    def __init__(
        self,
        vectors: VectorsLike,
        *,
        weights: Iterable[float] | float | None = None,
        pmatrix: ProximityMatrix | None = None,
    ) -> None:
        """Create a new Members instance.

        :param vectors: array (n, m) of vectors
        :param weights: optional array (n,) of weights
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :raises ValueError: if input vectors and pmatrix do not align after stacking

        The `weights` argument is optional, but each member must have a weight. If
        `weights` is None, the weights are assumed to be the last axis of the vectors
        argument. This is consistent with `stacked_vectors` or image colors as
        `(r,g,b,a)`. If weights is a scalar, all members are given that weight. This
        is useful for clustering non-color vectors or other instances where all
        vectors should be treated with the same priority.

        The vectors must be sorted in case any members has equal proximity to
        multiple centers. When this happens, the centroid with the lowest
        lexigraphical order is chosen.

        Init will sort input vectors and sort any proximity matrix passed to align to
        the sorted vectors.

        Init will also stack vectors to remove any duplicates, but *cannot* update an
        input pmatrix to match. If you pass identical vectors and a pmatrix, you will
        get a ValueError.
        """
        self._stacked_vectors = np.asarray(vectors)

        if weights is None:
            pass
        elif isinstance(weights, int | float):
            self._stacked_vectors = add_weight_axis(self._stacked_vectors, weights)
        else:
            weights = np.asarray(list(weights)).reshape(-1, 1)
            self._stacked_vectors = np.hstack((self._stacked_vectors, weights))

        # sort input vectors
        sort_indices = np.lexsort(self._stacked_vectors.T[::-1])
        self._stacked_vectors = self._stacked_vectors[sort_indices]

        # stack input vectors to remove duplicates
        ensure_stacked = stack_vectors(self._stacked_vectors)
        if self._stacked_vectors.shape != ensure_stacked.shape and pmatrix is not None:
            msg = par(
                """Input pmatrix shape does not conform to input vectors after
                stacking duplicates."""
            )
            raise ValueError(msg)
        self._stacked_vectors = ensure_stacked

        if pmatrix is None:
            self.pmatrix = get_sqeuclidean_matrix(self.vectors)
        else:
            self.pmatrix = pmatrix[np.ix_(sort_indices, sort_indices)]

    def __len__(self) -> int:
        """Number of members in the Members instance.

        :return: number of members
        """
        return len(self.vectors)

    @property
    def vectors(self) -> FPArray:
        """Array of vectors.

        :return: array of vectors
        """
        return self._stacked_vectors[:, :-1]

    @property
    def weights(self) -> FPArray:
        """Array of weights.

        :return: array of weights
        """
        return self._stacked_vectors[:, -1]

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
        cls, vectors: VectorsLike, *, pmatrix: FPArray | None = None
    ) -> Members:
        """Create a Members instance from stacked_vectors.

        :param vectors: (n, m) a list of vectors
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: Members instance
        """
        return cls(vectors, weights=1, pmatrix=pmatrix)

    @classmethod
    def from_stacked_vectors(
        cls, stacked_vectors: VectorsLike, *, pmatrix: FPArray | None = None
    ) -> Members:
        """Create a Members instance from stacked_vectors.

        :param stacked_vectors: (n, m + 1) a list of vectors with weight channels in
            the last axis
        :param pmatrix: optional proximity matrix. If not given, will be calculated
            with squared Euclidean distance
        :return: Members instance
        """
        return cls(stacked_vectors, weights=None, pmatrix=pmatrix)
