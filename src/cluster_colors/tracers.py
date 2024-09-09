# """Color clusters that can be safely added to a Supercluster object.

# The cluster_colors library can break if two Member instances have the same rgb value.
# This allows you yo add one-cluster "tracer" clusters to a Supercluster object
# without worrying about a collision.

# You can use these tracers to identify, e.g., the
# whitest cluster (the cluster where the white tracer shows up). Member instances are
# not copied, created, or destroyed once a cluster is initialized,  so you can use `is`
# and `in` comparisons to identify the tracer.

#     # where `supercluster` is a Supercluster instance

#     WHITE = new_tracer(255, 255, 255)
#     white_cluster = Cluster({WHITE, [valid memebers with weights]})
#     supercluster.add(Clusters({WHITE}))
#     whitest_cluster = next(c for c in clusters if WHITE in c)


# :author: Shay Hill
# :created: 2023-04-14
# """

# import numpy as np

# from cluster_colors.clusters import Member

# # Color members are created by averaging pixels where each pixel is a 3-tuple of
# # 8-bit integers. So, there are some values that CANNOT exist in a color member.
# # (Unless you average > 68 billion pixels.) Any integer +- _DX is guaranteed to *not*
# # exist as any pixel in an existing color member.

# _DX = 1 / 2**36

# if 1 - _DX == 1:
#     _MSG = "This library is not compatible with your system."
#     raise ValueError(_MSG)


# def new_tracer(rgb: tuple[int, int, int]) -> Member:
#     """Create a new tracer with the given rgb value.

#     :param rgb: a 3-tuple of 8-bit integers
#     :return: a new tracer

#     These can be inserted into a Clusters object without worrying about collisions
#     with other Members. Tracers have a small weight which should not make a
#     meaningful difference in exemplar selection.
#     """
#     r, g, b = rgb
#     close_to_r = r - _DX if r else r + _DX
#     return Member(np.array([close_to_r, g, b, 0]))
