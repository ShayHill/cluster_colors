Divisive (but not hierarchical) clustering.

Slow, but clustering exactly how I want it. Iteratively split cluster with highest SSE. Splits are used to find new exemplars, which are thrown into k-medians with existing exemplars. Takes pretty extreme measures to avoid not only non-determinism but also non-arbitrary-ism:

* a Supercluster instance, when asked to split, will split the cluster with the highest SSE. If there is a tie, the Clusters instance will not arbitrarily decide between the tied clusters, but will instead split all clusters tied for max SSE. This means there is a small chance you will not be able to split a group of colors into exactly n clusters.
* delta-e is non-commutative, so delta-e is computed *twice* for each distance (a -> b and b -> a). The maximum of these two is used. This doubles the time of an already slow calculation, but delta-e is only used for distances between clusters, and this module is designed to work with small numbers of clusters (Agglomerative clustering may be a better bet if you want to use small clusters.)

Advantages:
* finds big clusters
* deterministic and non-arbitrary
* robust to outliers
* fast for what it is, can easily split a few thousand members into a small number of clusters
* decisions made early on do not effect the result as much as they would in true hierarchical clustering
* has strategies to avoid ties or arbitrary (even if deterministic) decisions with small member sets. This is important when dealing with questions like "which of these five colors is most unlike the others?"

Disadvantages:
* child clusters will not necessarily contain (or only contain) the members of the parent, so this is not hierarchical, though you can "merge" split clusters by regressing to previous states.
* sloooows down as the number of clusters grows, not the best way to de-cluster all the way back to constituent members.
* uses Euclidean distance (sum squared error) for many steps. Delta e is used for final splitting criteria.

This clustering is designed for questions like "what are the five dominant colors in this image (respecting transparency)?"

## Three large steps in the background

### Average colors by n-bit representation

`pool_colors`: reduce 8-bit image colors (potentially 16_777_216 colors) to a maximum of 262_144 by averaging. The ouput of `pool_colors` will also contain a weight axis for each color, representing the combined opacity of all pixels of that color.

### Median cut along longest axis

`cut_colors`: reduce colors to around 512 by recursively splitting along longest axis (longest actual axis. Not constrained to x, y, or, z axes).

### k-medians clustering

`KMedSupercluster`: split and merge (undo split) clusters.

* start with one cluster with 100 members
* split this cluster recursively into five clusters (30, 30, 20, 10, 10)
* ask for the largest cluster, and there's a tie
* KMedSupercluster will recursively unsplit clusters until all ties are broken. This will *rarely* be needed.


## Installation

    pip install cluster_colors

## Basic usage

~~~python
from cluster_colors import get_image_clusters

clusters = get_image_clusters(image_filename) # one cluster at this point
clusters.split_to_delta_e(16)
split_clusters = clusters.get_rsorted_clusters()

colors: list[tuple[float, float, float]] = [c.exemplar for c in split_clusters]

# to save the cluster exemplars as an image file

show_clusters(split_clusters, "open_file_to_see_clusters")
~~~
