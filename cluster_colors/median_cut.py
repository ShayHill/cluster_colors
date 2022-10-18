import numpy as np
import numpy.typing as npt
from typing import Annotated, Any, Iterable, cast, Sequence, TypeAlias


from cluster_colors.rgb_members_and_clusters import Member, Cluster

_FPArray: TypeAlias = npt.NDArray[np.floating[Any]]


def reduce_colors(colors: _FPArray, recursions: int = 8) -> set[Member]:
    """Reduce the number of colors in a numpy array of RGB values.

    Args:
        colors (np.ndarray): An array of RGB values.
        recursions (int, optional): The number of times to recurse. Defaults to 8.

    Returns:
        set[Member]: A set of Member objects.
    """
    clusters: set[Cluster] = {Cluster(colors)}
    for _ in range(recursions):
        for cluster in iter(clusters):
            clusters.remove(cluster)
            clusters.update(cluster.split())
    # TODO: create a property for Cluster that is a weighted Member
    return {Member(c.center) for c in clusters}
