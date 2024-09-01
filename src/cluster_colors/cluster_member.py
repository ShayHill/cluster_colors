"""One member of a cluster.

:author: Shay Hill
:created: 2024-09-01
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cluster_colors.type_hints import StackedVectors, Vector


class Member:
    """A member of a cluster.

    This will work with any weighted vector (any vector with one extra value on the
    last axis for weight).

    When clustering initial image arrays returned from `stack_image_colors`, the
    weight axis will only represent the number of times the color appears in the
    image. After removing some color or adding an alpha channel, the weight will also
    reflect the alpha channel, with transparent colors weighing less.
    """

    def __init__(self, weighted_vector: Vector) -> None:
        """Create a new Member instance.

        :param weighted_vector: a vector with a weight in the last axis
            (r, g, b, w)
        :param ancestors: sets of ancestors to merge
        """
        self.as_array = weighted_vector

    @property
    def vs(self) -> tuple[float, ...]:
        """All value axes of the Member as a tuple.

        :return: tuple of values that are not the weight
            the (r, g, b) in (r, g, b, w)
        """
        return tuple(self.as_array[:-1])

    @property
    def w(self) -> float:
        """Weight of the Member.

        :return: weight of the Member
            the w in (r, g, b, w)
        """
        return self.as_array[-1]

    @classmethod
    def new_members(cls, stacked_vectors: StackedVectors) -> set[Member]:
        """Transform an array of vectors into a set of Member instances.

        :param stacked_vectors: (-1, n + 1) a list of vectors with weight channels in
            the last axis
        :return: set of Member instances
        """
        return {Member(v) for v in stacked_vectors if v[-1]}
